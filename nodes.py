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
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import folder_paths
from comfy.utils import ProgressBar


CATEGORY = "DDHT/Video"


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
    return np.ascontiguousarray(array).tobytes()


def _prepare_audio(audio: Optional[dict], temp_dir: str) -> Tuple[Optional[str], int, int]:
    """Write ComfyUI AUDIO data to a temporary interleaved float32 file."""
    if not isinstance(audio, dict):
        return None, 0, 0

    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0 or sample_rate <= 0:
        return None, 0, 0

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
                    "FLOAT",
                    {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01},
                ),
                "filename_prefix": ("STRING", {"default": "DDHT/video"}),
                "video_codec": (["h264", "h265"], {"default": "h264"}),
                "quality": (
                    "INT",
                    {
                        "default": 18,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "FFmpeg CRF: lower is higher quality and larger file size.",
                    },
                ),
                "preset": (
                    [
                        "ultrafast",
                        "superfast",
                        "veryfast",
                        "faster",
                        "fast",
                        "medium",
                        "slow",
                        "slower",
                        "veryslow",
                    ],
                    {"default": "medium"},
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
        frame_rate: float,
        filename_prefix: str,
        video_codec: str,
        quality: int,
        preset: str,
        audio: Optional[dict] = None,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("images must be a non-empty ComfyUI IMAGE batch [frames, height, width, channels]")

        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg was not found. Install requirements.txt or make ffmpeg available on PATH."
            )

        frame_count, height, width, _ = images.shape
        output_path, output_name, subfolder = _next_output_path(filename_prefix, width, height)
        temp_dir = folder_paths.get_temp_directory()
        audio_path = None
        process = None

        try:
            audio_path, sample_rate, channels = _prepare_audio(audio, temp_dir)
            fps_text = f"{float(frame_rate):.6f}".rstrip("0").rstrip(".")
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                fps_text,
                "-i",
                "pipe:0",
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

            encoder = "libx264" if video_codec == "h264" else "libx265"
            command += [
                "-map",
                "0:v:0",
                "-c:v",
                encoder,
                "-preset",
                preset,
                "-crf",
                str(int(quality)),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
            ]

            if audio_path:
                # apad + shortest keeps the full video when audio is shorter,
                # and trims excess audio when audio is longer.
                command += [
                    "-map",
                    "1:a:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-af",
                    "apad",
                    "-shortest",
                ]
            else:
                command += ["-an"]

            command += ["-movflags", "+faststart", output_path]
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
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "How many frames to keep per second.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "extract_frames"
    CATEGORY = CATEGORY

    def extract_frames(self, images: torch.Tensor, source_fps: float, extract_fps: float):
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


NODE_CLASS_MAPPINGS = {
    "DDHT_SaveVideoSingleFile": DDHTSaveVideoSingleFile,
    "DDHT_ExtractFramesByFPS": DDHTExtractFramesByFPS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDHT_SaveVideoSingleFile": "保存视频（仅一个文件）- DDHT",
    "DDHT_ExtractFramesByFPS": "按每秒帧数抽帧 - DDHT",
}
