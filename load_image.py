# SPDX-License-Identifier: GPL-3.0-only
"""ComfyUI's native image loader with optional alpha compositing."""

from copy import deepcopy

import numpy as np
from PIL import Image

from .folder_images import parse_color, save_preview


def _native_loader():
    # `nodes` is ComfyUI's core module, not this package's `.nodes` module.
    # Resolve lazily so package registration does not load models or tensors.
    from nodes import LoadImage
    return LoadImage


def _fill_options(enabled, color):
    if not isinstance(enabled, bool):
        raise ValueError("填充透明区域必须是布尔值。")
    return parse_color(color) if enabled else None


def _composite(images, masks, color):
    if masks.shape[-2:] != images.shape[1:3]:
        # Native LoadImage returns a 64x64 zero mask for opaque images.
        # Keep that convention; it must not be broadcast over an arbitrary size.
        if masks.any().item():
            raise ValueError("图片与透明遮罩尺寸不一致，无法填充透明区域。")
        return images, masks
    transparency = masks.to(device=images.device, dtype=images.dtype).unsqueeze(-1)
    background = images.new_tensor([channel / 255.0 for channel in color])
    filled = images * (1.0 - transparency) + background * transparency
    return filled, masks.new_zeros(masks.shape)


def _preview(images, masks, filled):
    # Preview only the first frame; preserve every frame and original size below.
    pixels = images[0].detach().float().cpu().numpy()
    preview = Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))
    if not filled and masks.shape[-2:] == images.shape[1:3]:
        mask = masks[0].detach().float().cpu().numpy()
        alpha = np.rint(np.clip(1.0 - mask, 0, 1) * 255).astype(np.uint8)
        preview.putalpha(Image.fromarray(alpha))
    return save_preview(preview)


class DDHTLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = deepcopy(_native_loader().INPUT_TYPES())
        inputs["required"].update({
            "填充透明区域": ("BOOLEAN", {"default": False,
                "tooltip": "开启后按 Alpha 将透明和半透明区域合成到指定颜色，输出遮罩为全 0。"}),
            "填充颜色": ("STRING", {"default": "#FFFFFF",
                "tooltip": "支持 #RRGGBB 或 #RGB；白色 #FFFFFF，黑色 #000000，绿色 #00FF00。"}),
        })
        return inputs

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图片", "透明遮罩")
    FUNCTION = "load_image"
    CATEGORY = "DDHT/Image"
    DESCRIPTION = "使用 ComfyUI 原版方式上传和读取图片，可选填充透明区域及颜色；执行后预览输出结果。"

    def load_image(self, image, 填充透明区域=False, 填充颜色="#FFFFFF"):
        from comfy import model_management

        color = _fill_options(填充透明区域, 填充颜色)
        model_management.throw_exception_if_processing_interrupted()
        images, masks = _native_loader()().load_image(image)
        model_management.throw_exception_if_processing_interrupted()
        if color is not None:
            images, masks = _composite(images, masks, color)
        preview = _preview(images, masks, 填充透明区域)
        model_management.throw_exception_if_processing_interrupted()
        return {"ui": {"images": [preview]}, "result": (images, masks)}

    @classmethod
    def IS_CHANGED(cls, image, 填充透明区域=False, 填充颜色="#FFFFFF"):
        # ComfyUI also keys execution caches on the widget input values.
        return _native_loader().IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image, 填充透明区域=False, 填充颜色="#FFFFFF"):
        try:
            _fill_options(填充透明区域, 填充颜色)
        except ValueError as error:
            return str(error)
        return _native_loader().VALIDATE_INPUTS(image)


NODE_CLASS_MAPPINGS = {"DDHT_LoadImage": DDHTLoadImage}
NODE_DISPLAY_NAME_MAPPINGS = {"DDHT_LoadImage": "读取图片（透明填充）- DDHT"}
