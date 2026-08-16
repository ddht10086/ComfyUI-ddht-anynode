# SPDX-License-Identifier: GPL-3.0-only
"""Small, focused video utilities for ComfyUI.

The save node intentionally creates exactly one persistent output file. It does
not save a metadata PNG and it does not keep a silent intermediate video when
audio is connected.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import folder_paths
from comfy.utils import ProgressBar

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    try:
        from comfy_execution.graph_utils import ExecutionBlocker
    except ImportError:
        ExecutionBlocker = None


CATEGORY = "DDHT/Video"

VIDEO_FORMATS = {
    "video/h264-mp4": {
        "encoder": "libx264",
        "pix_fmts": ("yuv420p", "yuv420p10le"),
        "default_crf": 19,
    },
    "video/h265-mp4": {
        "encoder": "libx265",
        "pix_fmts": ("yuv420p10le", "yuv420p"),
        "default_crf": 22,
    },
}


def _find_ffmpeg() -> Optional[str]:
    """Return an FFmpeg executable supplied by PATH or imageio-ffmpeg."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _frame_to_rgb24(frame: torch.Tensor) -> bytes:
    """Convert one ComfyUI HWC float frame to packed RGB24 bytes."""
    while frame.ndim > 3:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {tuple(frame.shape)}")

    channels = frame.shape[-1]
    if channels == 1:
        frame = frame.repeat(1, 1, 3)
    elif channels >= 3:
        frame = frame[..., :3]
    else:
        raise ValueError(f"Expected 1, 3, or 4 channels, got {channels}")

    array = (
        frame.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp_(0.0, 1.0)
        .mul_(255.0)
        .round_()
        .to(dtype=torch.uint8)
        .numpy()
    )
    # VHS pads formats to their required dimension alignment (2 by default)
    # using edge replication. H.264/H.265 4:2:0 formats require even sizes.
    pad_height = -array.shape[0] % 2
    pad_width = -array.shape[1] % 2
    if pad_height or pad_width:
        array = np.pad(
            array,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="edge",
        )
    return np.ascontiguousarray(array).tobytes()


def _prepare_audio(audio: Optional[Mapping], temp_dir: str) -> Tuple[Optional[str], int, int]:
    """Write ComfyUI AUDIO data to a temporary interleaved float32 file."""
    if audio is None:
        return None, 0, 0
    if not isinstance(audio, Mapping):
        raise TypeError(
            "audio must be a ComfyUI AUDIO mapping containing waveform and sample_rate"
        )

    try:
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"] or 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Connected audio is missing a valid waveform or sample_rate."
        ) from exc
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        raise ValueError("Connected audio waveform is empty or invalid.")
    if sample_rate <= 0:
        raise ValueError("Connected audio sample_rate must be greater than zero.")

    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        # AUDIO is normally [batch, channels, samples]. Multiple batches are
        # concatenated in time instead of silently discarding them.
        waveform = waveform.permute(1, 0, 2).reshape(waveform.shape[1], -1)
    elif waveform.ndim != 2:
        raise ValueError(f"Unsupported AUDIO waveform shape: {tuple(waveform.shape)}")

    channels = int(waveform.shape[0])
    interleaved = (
        waveform.clamp(-1.0, 1.0)
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )

    os.makedirs(temp_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".f32", prefix="ddht_audio_", dir=temp_dir, delete=False
    )
    try:
        handle.write(interleaved.tobytes())
        return handle.name, sample_rate, channels
    finally:
        handle.close()


def _next_output_path(filename_prefix: str, width: int, height: int) -> Tuple[str, str, str]:
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, width, height
    )
    os.makedirs(full_output_folder, exist_ok=True)

    while True:
        output_name = f"{filename}_{counter:05}.mp4"
        output_path = os.path.join(full_output_folder, output_name)
        if not os.path.exists(output_path):
            return output_path, output_name, subfolder
        counter += 1


class DDHTSaveVideoSingleFile:
    """Encode an IMAGE batch, plus optional AUDIO, into exactly one MP4."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 24, "min": 1, "max": 240, "step": 1},
                ),
                "filename_prefix": ("STRING", {"default": "DDHT/video"}),
                "format": (
                    list(VIDEO_FORMATS),
                    {"default": "video/h264-mp4"},
                ),
                "pix_fmt": (
                    ["yuv420p", "yuv420p10le"],
                    {"default": "yuv420p"},
                ),
                "quality": (
                    "INT",
                    {
                        "default": 19,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "与 VHS 相同的 FFmpeg CRF；H.264 默认 19，H.265 默认 22。",
                    },
                ),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_video_path",)
    FUNCTION = "save_video"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def save_video(
        self,
        images: torch.Tensor,
        frame_rate: int,
        filename_prefix: str,
        format: str,
        pix_fmt: str,
        quality: int,
        audio: Optional[Mapping] = None,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("images must be a non-empty ComfyUI IMAGE batch [frames, height, width, channels]")

        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg was not found. Install requirements.txt or make ffmpeg available on PATH."
            )

        format_config = VIDEO_FORMATS.get(format)
        if format_config is None:
            raise ValueError(f"Unsupported video format: {format}")
        if pix_fmt not in format_config["pix_fmts"]:
            allowed = ", ".join(format_config["pix_fmts"])
            raise ValueError(f"pix_fmt {pix_fmt} is invalid for {format}; choose {allowed}")

        frame_count, height, width, _ = images.shape
        encoded_width = int(width) + (-int(width) % 2)
        encoded_height = int(height) + (-int(height) % 2)
        output_path, output_name, subfolder = _next_output_path(filename_prefix, width, height)
        temp_dir = folder_paths.get_temp_directory()
        audio_path = None
        process = None

        try:
            audio_path, sample_rate, channels = _prepare_audio(audio, temp_dir)
            fps_text = str(int(frame_rate))
            command = [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-color_range",
                "pc",
                "-colorspace",
                "rgb",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-s",
                f"{encoded_width}x{encoded_height}",
                "-r",
                fps_text,
                "-i",
                "-",
            ]

            if audio_path:
                command += [
                    "-f",
                    "f32le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    str(channels),
                    "-i",
                    audio_path,
                ]

            command += [
                "-map",
                "0:v:0",
                "-n",
                "-c:v",
                format_config["encoder"],
            ]
            if format == "video/h265-mp4":
                command += ["-vtag", "hvc1"]
            command += [
                "-pix_fmt",
                pix_fmt,
                "-crf",
                str(int(quality)),
                "-vf",
                "scale=out_color_matrix=bt709",
                "-color_range",
                "tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
            ]
            if format == "video/h265-mp4":
                command += ["-preset", "medium", "-x265-params", "log-level=quiet"]

            if audio_path:
                minimum_audio_duration = int(frame_count) / int(frame_rate) + 1
                command += [
                    "-map",
                    "1:a:0",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "use_metadata_tags",
                    "-af",
                    f"apad=whole_dur={minimum_audio_duration}",
                    "-shortest",
                ]
            else:
                command += ["-an"]

            command += [output_path]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            progress = ProgressBar(int(frame_count))
            try:
                for frame in images:
                    process.stdin.write(_frame_to_rgb24(frame))
                    progress.update(1)
            except BrokenPipeError:
                # FFmpeg's actual error is reported from stderr below.
                pass
            finally:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()

            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code != 0:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"FFmpeg failed with exit code {return_code}:\n{message}")

        except Exception:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

        relative_path = Path(subfolder, output_name).as_posix() if subfolder else output_name
        return {
            "ui": {"text": [f"Saved one video file: {relative_path}"]},
            "result": (output_path,),
        }


class DDHTExtractFramesByFPS:
    """Uniformly sample a video-like IMAGE batch at a requested FPS."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01},
                ),
                "extract_fps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "How many frames to keep per second.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "extract_frames"
    CATEGORY = CATEGORY

    def extract_frames(self, images: torch.Tensor, source_fps: float, extract_fps: int):
        if not isinstance(images, torch.Tensor) or images.ndim < 1 or images.shape[0] == 0:
            raise ValueError("images must contain at least one frame")
        if source_fps <= 0 or extract_fps <= 0:
            raise ValueError("source_fps and extract_fps must be greater than zero")

        frame_count = int(images.shape[0])
        if frame_count == 1 or extract_fps >= source_fps:
            return (images, frame_count)

        # Samples are placed at 0, 1/extract_fps, 2/extract_fps, ... and
        # mapped to the nearest source-frame timestamp.
        sample_count = math.floor((frame_count - 1) * extract_fps / source_fps) + 1
        step = source_fps / extract_fps
        index_values = []
        for sample_number in range(sample_count):
            index = min(frame_count - 1, math.floor(sample_number * step + 0.5))
            if not index_values or index != index_values[-1]:
                index_values.append(index)
        indices = torch.tensor(index_values, device=images.device, dtype=torch.long)
        selected = torch.index_select(images, 0, indices)
        return (selected, int(selected.shape[0]))


class DDHTTextLengthGate:
    """Pass text through only when its character count is inside a range."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                    },
                ),
                "min_length": (
                    "INT",
                    {"default": 1, "min": 0, "max": 10000000, "step": 1},
                ),
                "max_length": (
                    "INT",
                    {"default": 4000, "min": 0, "max": 10000000, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "character_count")
    FUNCTION = "check_length"
    CATEGORY = "DDHT/Text"

    def check_length(self, text: str, min_length: int, max_length: int):
        if min_length > max_length:
            raise ValueError(
                f"Invalid text length range: min_length ({min_length}) is greater than "
                f"max_length ({max_length})."
            )

        if not isinstance(text, str):
            text = str(text)
        character_count = len(text)

        if min_length <= character_count <= max_length:
            return (text, character_count)

        reason = (
            f"Text length gate stopped downstream execution: {character_count} characters "
            f"is outside the inclusive range {min_length} to {max_length}."
        )
        if ExecutionBlocker is not None:
            blocker = ExecutionBlocker(reason)
            return (blocker, blocker)
        raise RuntimeError(reason)


NODE_CLASS_MAPPINGS = {
    "DDHT_SaveVideoSingleFile": DDHTSaveVideoSingleFile,
    "DDHT_ExtractFramesByFPS": DDHTExtractFramesByFPS,
    "DDHT_TextLengthGate": DDHTTextLengthGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDHT_SaveVideoSingleFile": "保存视频（仅一个文件）- DDHT",
    "DDHT_ExtractFramesByFPS": "按每秒帧数抽帧 - DDHT",
    "DDHT_TextLengthGate": "文本长度门控 - DDHT",
}
