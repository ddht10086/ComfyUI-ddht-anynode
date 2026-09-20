# SPDX-License-Identifier: GPL-3.0-only
"""Read one local image per execution, with seeded selection and native preview."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import random
import re
import tempfile
import threading

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".apng", ".webp", ".bmp", ".dib",
    ".gif", ".tif", ".tiff", ".ico", ".tga", ".ppm", ".pgm", ".pbm", ".pnm",
    ".pcx", ".dds", ".jp2", ".j2k", ".jpf", ".jpx", ".jpc", ".j2c",
    ".avif", ".heic", ".heif",
})
PREVIEW_SUBFOLDER = "ddht_folder_images"


def resolve_folder(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请填写图片文件夹路径；路径必须位于运行 ComfyUI 的机器上。")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value or "\x00" in value:
        raise ValueError("图片文件夹路径无效。")
    path = Path(os.path.expandvars(value)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"图片文件夹不存在或不是文件夹：{path}。请填写 ComfyUI 所在机器的路径。")
    return path


def _natural_key(path, folder):
    relative = path.relative_to(folder).as_posix()
    parts = tuple((1, int(part)) if part.isascii() and part.isdecimal() else (0, part.casefold())
                  for part in re.split(r"([0-9]+)", relative))
    return parts, relative


def scan_images(folder, recursive=False, check_interrupt=lambda: None):
    """Rescan names only; decode only the selected file. Do not follow directory links."""
    def on_error(error):
        raise error

    files = []
    try:
        for directory, _, names in os.walk(folder, followlinks=False, onerror=on_error):
            check_interrupt()
            for name in names:
                path = Path(directory) / name
                if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                    files.append(path)
            if not recursive:
                break
    except OSError as error:
        raise RuntimeError(f"无法读取图片文件夹，请检查访问权限：{folder}") from error
    files.sort(key=lambda path: _natural_key(path, folder))
    if not files:
        raise ValueError("文件夹中没有支持的图片；如图片位于下级目录，请开启“包含子文件夹”。")
    return files


def parse_color(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}", value.strip()):
        raise ValueError("填充颜色必须为 #RRGGBB 或 #RGB，例如白色 #FFFFFF、黑色 #000000。")
    value = value.strip()[1:]
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def _prepare_decoder(path):
    Image.init()
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"} and "HEIF" not in Image.OPEN:
        try:
            importlib.import_module("pillow_heif").register_heif_opener()
        except ImportError:
            raise RuntimeError("读取 HEIC/HEIF 需要在 ComfyUI 的 Python 环境安装 pillow-heif。") from None
    elif suffix == ".avif" and "AVIF" not in Image.OPEN:
        try:
            importlib.import_module("pillow_avif")
        except ImportError:
            raise RuntimeError("当前 Pillow 不支持 AVIF，请安装支持 AVIF 的 Pillow 或 pillow-avif-plugin。") from None


def read_image(path, fill_transparency=False, color=(255, 255, 255)):
    """Return HWC RGB float32, inverse-alpha mask, and the matching PIL preview."""
    _prepare_decoder(path)
    try:
        with Image.open(path) as source:
            source.seek(0)  # One selected file produces one image, including animated formats.
            image = ImageOps.exif_transpose(source)
            if image.mode.startswith("I;16") or (image.mode == "I" and source.format == "PNG"):
                # convert('RGB') clips 16-bit grayscale at 255; normalize it first.
                samples = np.asarray(image, dtype=np.float32)
                gray = np.clip(samples / 65535.0, 0.0, 1.0)
                rgb = np.repeat(gray[..., None], 3, axis=-1)
                transparent_sample = image.info.get("transparency")
                mask = (samples == transparent_sample).astype(np.float32) if isinstance(transparent_sample, int) else np.zeros(gray.shape, dtype=np.float32)
                if fill_transparency:
                    rgb = rgb * (1.0 - mask[..., None]) + np.asarray(color, dtype=np.float32) / 255.0 * mask[..., None]
                    mask = np.zeros(gray.shape, dtype=np.float32)
                preview = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
                if not fill_transparency:
                    preview.putalpha(Image.fromarray(np.rint((1.0 - mask) * 255).astype(np.uint8)))
            else:
                rgba = image.convert("RGBA")  # Also expands palette/tRNS and grayscale alpha.
                if fill_transparency:
                    background = Image.new("RGBA", rgba.size, (*color, 255))
                    preview = Image.alpha_composite(background, rgba).convert("RGB")
                    rgb = np.asarray(preview, dtype=np.float32) / 255.0
                    mask = np.zeros((preview.height, preview.width), dtype=np.float32)
                else:
                    pixels = np.asarray(rgba, dtype=np.float32) / 255.0
                    rgb = np.ascontiguousarray(pixels[..., :3])
                    mask = 1.0 - pixels[..., 3]
                    preview = rgba
            # Do not copy source EXIF, text metadata, or workflow data to the thumbnail.
            preview.info.clear()
            return rgb, mask, preview
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise RuntimeError(f"图片无法解码，可能已损坏或缺少解码器：{path.name}") from error


def save_preview(image):
    import folder_paths

    directory = Path(folder_paths.get_temp_directory()) / PREVIEW_SUBFOLDER
    thumbnail = image.copy()
    thumbnail.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    thumbnail.info.clear()
    path = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="ddht_", suffix=".png", dir=directory, delete=False) as file:
            path = Path(file.name)
            thumbnail.save(file, format="PNG", compress_level=1)
    except OSError as error:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError("无法创建节点预览，请检查 ComfyUI 临时目录的权限和剩余空间。") from error
    return {"filename": path.name, "subfolder": PREVIEW_SUBFOLDER, "type": "temp"}


def _nonnegative_integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} 必须是 0～{maximum} 之间的整数。")


class DDHTFolderImage:
    def __init__(self):
        self._sequence_signature = None
        self._next_index = 0
        self._lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "文件夹路径": ("STRING", {"default": "", "tooltip": "运行 ComfyUI 的机器上的文件夹；支持中文、空格和带引号的路径。"}),
            "读取方式": (["随机", "顺序"], {"default": "随机"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                             "tooltip": "仅随机模式使用。相同文件列表与种子选中同一张图片；randomize 可每次更新种子。"}),
            "包含子文件夹": ("BOOLEAN", {"default": False}),
            "顺序起始索引": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff,
                                "tooltip": "从 0 开始，按文件名自然排序逐次读取，读完后循环。超出图片数时取余。"}),
            "顺序重置标记": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff,
                                "tooltip": "修改此数值，下一次顺序读取就从起始索引重新开始；保持不变则继续。"}),
            "填充透明区域": ("BOOLEAN", {"default": False, "tooltip": "开启时将透明/半透明区域与填充颜色合成；关闭时透明度输出到 MASK。"}),
            "填充颜色": ("STRING", {"default": "#FFFFFF", "tooltip": "#RRGGBB 或 #RGB；例如白色 #FFFFFF、黑色 #000000、绿色 #00FF00。"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("图片", "透明遮罩", "文件名", "完整路径", "图片索引", "图片总数")
    FUNCTION = "load_image"
    CATEGORY = "DDHT/Image"
    OUTPUT_NODE = True  # Can also run on its own to preview the selected file.
    DESCRIPTION = "每次执行从本地文件夹读取一张图片；支持种子随机、顺序循环和透明区域填色。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Folder contents and sequential position change outside inputs.

    def load_image(self, 文件夹路径, 读取方式="随机", seed=0, 包含子文件夹=False,
                   顺序起始索引=0, 顺序重置标记=0, 填充透明区域=False, 填充颜色="#FFFFFF"):
        import torch
        from comfy import model_management

        check_interrupt = model_management.throw_exception_if_processing_interrupted
        check_interrupt()
        if 读取方式 not in {"随机", "顺序"}:
            raise ValueError("读取方式必须为随机或顺序。")
        if not isinstance(包含子文件夹, bool) or not isinstance(填充透明区域, bool):
            raise ValueError("包含子文件夹和填充透明区域必须是布尔值。")
        _nonnegative_integer(seed, "种子", 0xffffffffffffffff)
        _nonnegative_integer(顺序起始索引, "顺序起始索引", 0x7fffffff)
        _nonnegative_integer(顺序重置标记, "顺序重置标记", 0x7fffffff)
        color = parse_color(填充颜色) if 填充透明区域 else (255, 255, 255)
        folder = resolve_folder(文件夹路径)
        with self._lock:
            files = scan_images(folder, 包含子文件夹, check_interrupt)
            signature = (str(folder), 包含子文件夹, 顺序起始索引, 顺序重置标记,
                         tuple(path.relative_to(folder).as_posix() for path in files))
            if 读取方式 == "随机":
                index = random.Random(seed).randrange(len(files))
            else:
                index = (self._next_index if signature == self._sequence_signature else 顺序起始索引) % len(files)
            check_interrupt()
            path = files[index]
            rgb, mask, preview = read_image(path, 填充透明区域, color)
            check_interrupt()
            image_tensor = torch.from_numpy(rgb).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            ui_image = save_preview(preview)
            # Advance only after reading, conversion and preview have all succeeded.
            if 读取方式 == "顺序":
                self._sequence_signature = signature
                self._next_index = (index + 1) % len(files)
            else:
                self._sequence_signature = None
            return {"ui": {"images": [ui_image]},
                    "result": (image_tensor, mask_tensor, path.name, str(path), index, len(files))}


NODE_CLASS_MAPPINGS = {"DDHT_FolderImage": DDHTFolderImage}
NODE_DISPLAY_NAME_MAPPINGS = {"DDHT_FolderImage": "文件夹图片读取 - DDHT"}
