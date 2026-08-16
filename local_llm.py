# SPDX-License-Identifier: GPL-3.0-only
"""ComfyUI node for calling multimodal LLM servers on the local network."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management
from comfy.utils import ProgressBar

from .llm_adapters import build_request, execute_request, serialized_payload_size


FRAMEWORKS = ["llama.cpp", "Ollama", "vLLM", "SGLang", "OpenAI兼容"]
IMAGE_INPUT_NAMES = [f"图片{i}" for i in range(1, 9)]


def _as_image_batch(value: Any, input_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{input_name} 不是有效的 ComfyUI IMAGE。")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] == 0:
        raise ValueError(
            f"{input_name} 必须是非空 IMAGE 批次 [数量, 高, 宽, 通道]，当前形状为 {tuple(value.shape)}。"
        )
    return value


def _uniform_indices(total: int, maximum: int) -> List[int]:
    if total <= maximum:
        return list(range(total))
    if maximum <= 1:
        return [0]
    return [round(i * (total - 1) / (maximum - 1)) for i in range(maximum)]


def _select_frames(image_inputs: Sequence[Tuple[str, torch.Tensor]], maximum: int):
    frames: List[Tuple[str, torch.Tensor]] = []
    for input_name, batch in image_inputs:
        frames.extend((f"{input_name}[{index}]", frame) for index, frame in enumerate(batch))
    selected = [frames[index] for index in _uniform_indices(len(frames), maximum)]
    return frames, selected


def _frame_to_jpeg_base64(frame: torch.Tensor, max_edge: int, quality: int) -> str:
    if frame.ndim != 3:
        raise ValueError(f"图片帧必须是 HWC 三维张量，当前形状为 {tuple(frame.shape)}。")

    array = (
        frame.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .numpy()
    )
    channels = int(array.shape[-1])
    if channels == 1:
        rgb = np.repeat(array, 3, axis=-1)
    elif channels == 2:
        luminance = np.repeat(array[..., :1], 3, axis=-1)
        alpha = array[..., 1:2]
        rgb = luminance * alpha + (1.0 - alpha)
    elif channels >= 4:
        alpha = array[..., 3:4]
        rgb = array[..., :3] * alpha + (1.0 - alpha)
    elif channels == 3:
        rgb = array
    else:
        raise ValueError(f"不支持 {channels} 通道图片。")

    image = Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB")
    width, height = image.size
    if max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(target, resampling)

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=int(quality),
        optimize=True,
        progressive=True,
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_extra_parameters(value: str) -> Mapping[str, Any]:
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"高级参数 JSON 格式错误（第 {exc.lineno} 行，第 {exc.colno} 列）：{exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("高级参数 JSON 的最外层必须是对象，例如 {\"min_p\": 0.05}。")
    return parsed


def _interrupt_exception() -> BaseException:
    exception_class = getattr(model_management, "InterruptProcessingException", RuntimeError)
    return exception_class()


class DDHTLocalLLMInference:
    """Call llama.cpp, Ollama, vLLM, SGLang, or another OpenAI-like server."""

    @classmethod
    def INPUT_TYPES(cls):
        optional_images = {
            name: (
                "IMAGE",
                {
                    "forceInput": True,
                    "tooltip": "可输入一个图片批次；节点会按端口顺序展开所有帧。",
                },
            )
            for name in IMAGE_INPUT_NAMES
        }
        return {
            "required": {
                "提示词": ("STRING", {"default": "请描述这些图片。", "multiline": True}),
                "系统提示词": (
                    "STRING",
                    {"default": "你是一个准确、简洁的视觉分析助手。", "multiline": True},
                ),
                "框架": (FRAMEWORKS, {"default": "llama.cpp"}),
                "API地址": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "模型名称": ("STRING", {"default": "local-model"}),
                "最大生成token": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 131072, "step": 1},
                ),
                "最大输出字符数": (
                    "INT",
                    {
                        "default": 3000,
                        "min": 0,
                        "max": 10000000,
                        "step": 1,
                        "tooltip": "0 表示不限制；限制包含模型的思考内容。",
                    },
                ),
                "达到字符上限时": (
                    ["报错终止流程", "截断并输出"],
                    {"default": "报错终止流程"},
                ),
                "最大图片数": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "超出时在全部端口展开后的图片中均匀抽取。",
                    },
                ),
                "图片最大边长": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 8192, "step": 64},
                ),
                "JPEG质量": (
                    "INT",
                    {"default": 90, "min": 40, "max": 100, "step": 1},
                ),
                "温度": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {"default": 20, "min": 0, "max": 1000, "step": 1},
                ),
                "重复惩罚": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "step": 1,
                        "control_after_generate": True,
                    },
                ),
                "思考模式": (["自动", "开启", "关闭"], {"default": "自动"}),
                "推理强度": (
                    ["自动", "low", "medium", "high", "xhigh"],
                    {"default": "自动"},
                ),
                "输出思考内容": ("BOOLEAN", {"default": False}),
                "连接超时秒": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 300.0, "step": 0.1},
                ),
                "生成超时秒": (
                    "FLOAT",
                    {"default": 600.0, "min": 1.0, "max": 86400.0, "step": 1.0},
                ),
                "最大请求体MB": (
                    "FLOAT",
                    {"default": 64.0, "min": 1.0, "max": 1024.0, "step": 1.0},
                ),
                "API密钥环境变量": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "填写环境变量名称，不要在工作流里直接保存密钥。",
                    },
                ),
                "高级参数JSON": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "合并到请求顶层；messages 和 stream 不允许覆盖。",
                    },
                ),
            },
            "optional": optional_images,
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("文本", "字符数", "用量JSON")
    FUNCTION = "infer"
    CATEGORY = "DDHT/LLM"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Remote generation should run for every queued workflow execution.
        return float("NaN")

    def infer(
        self,
        提示词: str,
        系统提示词: str,
        框架: str,
        API地址: str,
        模型名称: str,
        最大生成token: int,
        最大输出字符数: int,
        达到字符上限时: str,
        最大图片数: int,
        图片最大边长: int,
        JPEG质量: int,
        温度: float,
        top_p: float,
        top_k: int,
        重复惩罚: float,
        seed: int,
        思考模式: str,
        推理强度: str,
        输出思考内容: bool,
        连接超时秒: float,
        生成超时秒: float,
        最大请求体MB: float,
        API密钥环境变量: str,
        高级参数JSON: str,
        **kwargs,
    ):
        prompt = str(提示词 or "")
        system_prompt = str(系统提示词 or "")
        if not prompt.strip():
            raise ValueError("提示词不能为空。")
        if not str(模型名称).strip():
            raise ValueError("模型名称不能为空。")

        image_inputs: List[Tuple[str, torch.Tensor]] = []
        for name in IMAGE_INPUT_NAMES:
            value = kwargs.get(name)
            if value is not None:
                image_inputs.append((name, _as_image_batch(value, name)))

        all_frames, selected_frames = _select_frames(image_inputs, int(最大图片数))
        progress = ProgressBar(len(selected_frames)) if selected_frames else None
        images_base64: List[str] = []
        for _frame_name, frame in selected_frames:
            if model_management.processing_interrupted():
                raise _interrupt_exception()
            images_base64.append(
                _frame_to_jpeg_base64(frame, int(图片最大边长), int(JPEG质量))
            )
            if progress:
                progress.update(1)

        environment_name = str(API密钥环境变量 or "").strip()
        api_key = ""
        if environment_name:
            api_key = os.environ.get(environment_name, "")
            if not api_key:
                raise ValueError(f"环境变量 {environment_name} 未设置或为空。")

        extra_parameters = _parse_extra_parameters(str(高级参数JSON or ""))
        spec = build_request(
            framework=框架,
            base_url=str(API地址),
            model=str(模型名称).strip(),
            prompt=prompt,
            system_prompt=system_prompt,
            images_base64=images_base64,
            max_tokens=int(最大生成token),
            temperature=float(温度),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(重复惩罚),
            seed=int(seed),
            thinking_mode=思考模式,
            reasoning_effort=推理强度,
            api_key=api_key,
            extra_parameters=extra_parameters,
        )
        request_size = serialized_payload_size(spec)
        request_limit = int(float(最大请求体MB) * 1024 * 1024)
        if request_size > request_limit:
            raise RuntimeError(
                f"请求体约 {request_size / 1024 / 1024:.1f} MB，超过设置的 {最大请求体MB:g} MB。"
                "请减少最大图片数、图片最大边长或 JPEG 质量。"
            )

        result = execute_request(
            spec,
            connect_timeout=float(连接超时秒),
            read_timeout=float(生成超时秒),
            max_output_characters=int(最大输出字符数),
            on_character_limit=达到字符上限时,
            interrupted=model_management.processing_interrupted,
            interrupt_exception=_interrupt_exception,
        )

        if 输出思考内容 and result.reasoning:
            text = f"<think>\n{result.reasoning}\n</think>\n\n{result.content}"
        else:
            text = result.content

        usage: Dict[str, Any] = dict(result.usage)
        usage.update(
            {
                "framework": spec.framework,
                "endpoint": spec.endpoint,
                "model": str(模型名称).strip(),
                "connected_image_inputs": len(image_inputs),
                "input_image_count": len(all_frames),
                "sent_image_count": len(selected_frames),
                "request_bytes": request_size,
                "generated_character_count": result.generated_character_count,
                "returned_character_count": len(text),
                "reached_character_limit": result.reached_character_limit,
                "finish_reason": result.finish_reason,
            }
        )
        return (text, len(text), json.dumps(usage, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {"DDHT_LocalLLMInference": DDHTLocalLLMInference}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDHT_LocalLLMInference": "局域网多模态 LLM 推理 - DDHT"
}

