# SPDX-License-Identifier: GPL-3.0-only
"""ComfyUI node for text and image input to the official DeepSeek API."""

from __future__ import annotations

import json
import time

from .deepseek_client import (
    MODELS, MAX_BODY_BYTES, build_payload, execute_request, resolve_api_key,
    resolve_model, serialize_payload, validate_image_model, _integer,
)


IMAGE_NAMES = tuple(f"图片{i}" for i in range(1, 9))


def encode_images(inputs, model, maximum, max_edge, quality, check_interrupt):
    connected = [(name, inputs[name]) for name in IMAGE_NAMES if inputs.get(name) is not None]
    validate_image_model(model, bool(connected))
    _integer(maximum, "最大图片数", 1, 600)
    _integer(max_edge, "图片最大边长", 64, 4096)
    _integer(quality, "JPEG 质量", 1, 100)
    if not connected:
        return [], 0

    # Reuse the existing IMAGE/JPEG path without requiring Torch merely to
    # inspect this node's schema or run text-only protocol tests.
    from .local_llm import _as_image_batch, _frame_to_jpeg_base64
    from comfy.utils import ProgressBar

    batches = [(name, _as_image_batch(value, name)) for name, value in connected]
    count = sum(int(batch.shape[0]) for _, batch in batches)
    if count > maximum:
        raise ValueError(f"已连接 {count} 张图片，超过最大图片数 {maximum}；请先抽帧或提高上限。不会自动丢弃图片。")
    progress = ProgressBar(count)
    images, encoded_size = [], 0
    for _, batch in batches:
        for frame in batch:
            check_interrupt()
            encoded = _frame_to_jpeg_base64(frame, max_edge, quality)
            encoded_size += len(encoded)
            if encoded_size > MAX_BODY_BYTES:
                raise ValueError("图片编码已超过 48 MiB 请求限制，请减少数量、尺寸或 JPEG 质量。")
            images.append(encoded)
            progress.update(1)
    return images, len(connected)


class DDHTDeepSeekAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "系统提示词": ("STRING", {"default": "", "multiline": True}),
                "模型": (list(MODELS), {"default": "deepseek-flash", "tooltip": "deepseek-flash 支持文字和图片；deepseek-v4-pro 仅文本。"}),
                "API_Key": ("STRING", {"default": "", "tooltip": "直接填写密钥，或留空读取环境变量。直接输入可能随工作流、历史或错误日志保存；分享前请清空。"}),
                "最大生成token": ("INT", {"default": 8192, "min": 1, "max": 393216}),
                "思考模式": (["自动", "开启", "关闭"], {"default": "自动"}),
                "推理强度": (["自动", "low", "high", "max"], {"default": "自动"}),
                "温度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "仅在思考模式关闭时发送。"}),
                "最大图片数": ("INT", {"default": 24, "min": 1, "max": 600, "tooltip": "图片批次全部展开，超限报错，不会自动抽样。"}),
                "图片最大边长": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
                "JPEG质量": ("INT", {"default": 90, "min": 1, "max": 100}),
                "图片细节": (["auto", "low", "high", "original"], {"default": "auto"}),
                "最大输出字符数": ("INT", {"default": 100000, "min": 1, "max": 10000000, "tooltip": "回答与思考字符数合计，超限断流报错。"}),
                "生成超时秒": ("INT", {"default": 600, "min": 1, "max": 86400}),
            },
            "optional": {
                "自定义模型名称": ("STRING", {"default": "", "tooltip": "模型选择自定义时生效，填写官方模型 ID。"}),
                "API密钥环境变量": ("STRING", {"default": "DEEPSEEK_API_KEY", "tooltip": "API_Key 留空时读取；环境变量在运行 ComfyUI 的机器上设置。"}),
                **{name: ("IMAGE",) for name in IMAGE_NAMES},
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("文本", "字符数", "思考内容", "用量JSON")
    FUNCTION = "infer"
    CATEGORY = "DDHT/LLM"
    DESCRIPTION = "调用 DeepSeek 官方 API；文本和可选图片发送到云端，返回最终回答。需有效 API Key；按官方规则计费。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def infer(
        self, 提示词, 系统提示词="", 模型="deepseek-flash", API_Key="",
        最大生成token=8192, 思考模式="自动", 推理强度="自动", 温度=1.0,
        最大图片数=24, 图片最大边长=1024, JPEG质量=90, 图片细节="auto",
        最大输出字符数=100000, 生成超时秒=600, 自定义模型名称="",
        API密钥环境变量="DEEPSEEK_API_KEY", **kwargs,
    ):
        from comfy import model_management

        def check_interrupt():
            if model_management.processing_interrupted():
                raise model_management.InterruptProcessingException()

        check_interrupt()
        model = resolve_model(模型, 自定义模型名称)
        key = resolve_api_key(API_Key, API密钥环境变量)
        options = dict(
            model=model, prompt=提示词, system_prompt=系统提示词,
            max_tokens=最大生成token, thinking=思考模式, reasoning_effort=推理强度,
            temperature=温度, image_detail=图片细节,
        )
        build_payload(**options)  # Validate text/settings before converting images.
        start = time.monotonic()
        images, connected = encode_images(kwargs, model, 最大图片数, 图片最大边长, JPEG质量, check_interrupt)
        payload = build_payload(**options, images=images)
        request_size = len(serialize_payload(payload))
        result = execute_request(
            payload, key, timeout_seconds=生成超时秒,
            max_output_characters=最大输出字符数, check_interrupt=check_interrupt,
        )
        usage = {
            "provider": "DeepSeek", "requested_model": model, "response_model": result.model,
            "request_id": result.request_id, "usage": result.usage,
            "finish_reason": result.finish_reason, "connected_image_inputs": connected,
            "sent_image_count": len(images), "request_bytes": request_size,
            "returned_character_count": len(result.text), "reasoning_character_count": len(result.reasoning),
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        return (result.text, len(result.text), result.reasoning, json.dumps(usage, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {"DDHT_DeepSeekAPI": DDHTDeepSeekAPI}
NODE_DISPLAY_NAME_MAPPINGS = {"DDHT_DeepSeekAPI": "DeepSeek API 推理 - DDHT"}
