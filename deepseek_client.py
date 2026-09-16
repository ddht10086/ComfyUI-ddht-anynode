# SPDX-License-Identifier: GPL-3.0-only
"""DeepSeek official Chat Completions client; independent of ComfyUI/Torch."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass


ENDPOINT = "https://api.deepseek.com/chat/completions"
MAX_BODY_BYTES = 48 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MODELS = ("deepseek-flash", "deepseek-v4-pro", "自定义")
HTTP_ERRORS = {
    400: "请求格式错误，请检查模型与图片参数。",
    401: "认证失败，请检查 API Key。",
    402: "账户余额不足，请检查 DeepSeek 账户余额。",
    403: "当前密钥没有访问权限。",
    404: "模型或接口不存在，请检查模型名称。",
    413: "请求体过大，请减少图片数量或尺寸。",
    422: "参数不被模型支持，请检查模型名称和生成设置。",
    429: "请求过于频繁或超过并发限制，请稍后重试。",
    500: "DeepSeek 服务器出错，请稍后重试。",
    503: "DeepSeek 服务繁忙，请稍后重试。",
}


@dataclass
class DeepSeekResult:
    text: str
    reasoning: str
    usage: dict
    finish_reason: str
    request_id: str
    model: str


def resolve_api_key(direct_key: str, environment_name: str) -> str:
    if not isinstance(direct_key, str) or not isinstance(environment_name, str):
        raise ValueError("API Key 和环境变量名称必须是字符串。")
    key = direct_key.strip()
    if not key and environment_name.strip():
        key = os.environ.get(environment_name.strip(), "").strip()
    if not key:
        raise ValueError("请填写 API Key，或在启动 ComfyUI 的环境中设置 DEEPSEEK_API_KEY。")
    if any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError("API Key 包含空白或无效字符，请检查复制内容。")
    return key


def resolve_model(selection: str, custom: str) -> str:
    if selection not in MODELS:
        raise ValueError("模型选项无效；其他模型请选择“自定义”并填写名称。")
    model = custom.strip() if selection == "自定义" and isinstance(custom, str) else selection
    if not model or model == "自定义":
        raise ValueError("选择自定义模型时，必须填写自定义模型名称。")
    return model


def validate_image_model(model: str, has_images: bool) -> None:
    if has_images and model.lower() in {"deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"}:
        raise ValueError("所选模型不支持图片；请改选 deepseek-flash，或断开图片输入。")


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须是 {minimum}～{maximum} 范围内的整数。")
    return value


def build_payload(
    *, model, prompt, system_prompt="", images=(), max_tokens=8192,
    thinking="自动", reasoning_effort="自动", temperature=1.0, image_detail="auto",
):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("提示词不能为空。")
    if not isinstance(system_prompt, str):
        raise ValueError("系统提示词必须是文本。")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("模型名称不能为空。")
    _integer(max_tokens, "最大生成 token", 1, 393216)
    if thinking not in {"自动", "开启", "关闭"}:
        raise ValueError("思考模式无效。")
    if reasoning_effort not in {"自动", "low", "high", "max"}:
        raise ValueError("推理强度无效。")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("温度必须在 0～2 之间。")
    if image_detail not in {"auto", "low", "high", "original"}:
        raise ValueError("图片细节选项无效。")
    images = list(images)
    validate_image_model(model, bool(images))
    if len(images) > 600:
        raise ValueError("每次请求最多可发送 600 张图片。")

    content = [{"type": "text", "text": prompt}]
    for index, encoded in enumerate(images, 1):
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("图片编码为空或无效。")
        if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError("单张图片超过 32 MiB，请降低图片尺寸或 JPEG 质量。")
        content.extend([
            {"type": "text", "text": f"图{index}："},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}", "detail": image_detail,
            }},
        ])
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content if images else prompt})
    payload = {
        "model": model.strip(), "messages": messages, "stream": True,
        "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
    }
    if thinking != "自动":
        payload["thinking"] = {"type": "enabled" if thinking == "开启" else "disabled"}
    if thinking == "关闭":
        payload["temperature"] = float(temperature)
    elif reasoning_effort != "自动":
        payload["reasoning_effort"] = reasoning_effort
    return payload


def serialize_payload(payload):
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("DeepSeek 请求体超过 48 MiB，请减少图片数量、尺寸或 JPEG 质量。")
    return body


def _events(response, check):
    """Parse SSE boundaries, comments/keepalives, and multiline data fields."""
    parts = []
    size = 0
    for raw in response.iter_lines(chunk_size=64):
        check()
        try:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        except UnicodeDecodeError:
            raise RuntimeError("DeepSeek 返回了无效的 UTF-8 数据。") from None
        if not isinstance(line, str):
            raise RuntimeError("DeepSeek 返回了无效流数据。")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts, size = [], 0
        elif line.startswith("data:"):
            part = line[5:].lstrip(" ")
            size += len(part)
            if size > 1024 * 1024:
                raise RuntimeError("DeepSeek 单条流事件过大，已停止读取。")
            parts.append(part)
        elif line.startswith((":", "event:", "id:", "retry:")):
            continue
        else:
            raise RuntimeError("DeepSeek 未返回预期的 SSE 流，请稍后重试。")
    if parts:
        yield "\n".join(parts)


def execute_request(
    payload, api_key, *, timeout_seconds=600, max_output_characters=100000,
    check_interrupt=lambda: None,
):
    """Consume one paid request. No retries, redirects, or partial-success output."""
    import requests

    if not isinstance(timeout_seconds, (float, int)) or not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 86400:
        raise ValueError("生成超时秒必须在 1～86400 之间。")
    _integer(max_output_characters, "最大输出字符数", 1, 10000000)
    api_key = resolve_api_key(api_key, "")
    body = serialize_payload(payload)
    deadline = time.monotonic() + timeout_seconds

    def check():
        check_interrupt()
        if time.monotonic() >= deadline:
            raise RuntimeError("DeepSeek 生成超时，已停止读取；未返回不完整结果。")

    check()
    response = None
    texts, thoughts = [], []
    usage, finish_reason, request_id, response_model = {}, None, "", ""
    count, done = 0, False
    try:
        response = requests.post(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "text/event-stream"},
            stream=True, timeout=(min(10, timeout_seconds), min(30, timeout_seconds)),
            allow_redirects=False,
        )
        check()
        if response.status_code != 200:
            # Never echo an upstream response body: it may contain request data
            # or credentials. HTTP status gives actionable, stable diagnostics.
            detail = HTTP_ERRORS.get(response.status_code, "服务返回异常状态，请检查账户或稍后重试。")
            raise RuntimeError(f"DeepSeek API 请求失败（HTTP {response.status_code}）：{detail}")

        for data in _events(response, check):
            check()
            if data.strip() == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(data)
            except (ValueError, TypeError):
                raise RuntimeError("DeepSeek 返回了无法解析的流数据，已停止执行。") from None
            if not isinstance(chunk, dict):
                raise RuntimeError("DeepSeek 流数据结构无效。")
            if chunk.get("error"):
                raise RuntimeError("DeepSeek 在生成中返回错误，未输出不完整结果，请检查账户和模型设置。")
            if isinstance(chunk.get("usage"), dict):
                # Only numeric usage fields are exported, never arbitrary
                # error/request objects that a remote endpoint could echo.
                usage = {k: v for k, v in chunk["usage"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if isinstance(chunk.get("id"), str):
                request_id = chunk["id"][:200]
            if isinstance(chunk.get("model"), str):
                response_model = chunk["model"][:200]
            choices = chunk.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise RuntimeError("DeepSeek 返回了非预期的回答数量。")
            if not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
                raise RuntimeError("DeepSeek 回答数据结构无效。")
            delta = choice.get("delta", {})
            if delta.get("tool_calls"):
                raise RuntimeError("当前节点不执行工具调用，请使用普通文本回答。")
            for field, target in (("reasoning_content", thoughts), ("content", texts)):
                value = delta.get(field)
                if value is not None:
                    if not isinstance(value, str):
                        raise RuntimeError("DeepSeek 返回了非文本回答。")
                    count += len(value)
                    if count > max_output_characters:
                        raise RuntimeError("DeepSeek 输出（含思考）超过最大输出字符数，已断开连接并终止流程。")
                    target.append(value)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        check()
    except requests.Timeout:
        raise RuntimeError("DeepSeek 网络等待超时，请稍后重试；本次没有自动重试。") from None
    except requests.RequestException:
        raise RuntimeError("DeepSeek 网络连接失败或传输中断，请检查网络；本次没有自动重试。") from None
    finally:
        if response is not None:
            response.close()

    if finish_reason == "length":
        raise RuntimeError("DeepSeek 达到最大生成 token，回答不完整；请提高上限或缩短提示词。")
    if finish_reason == "content_filter":
        raise RuntimeError("DeepSeek 未完成回答（内容过滤），已终止流程。")
    if not done or finish_reason != "stop":
        raise RuntimeError("DeepSeek 流未正常完成，未向后续节点返回不完整结果。")
    text = "".join(texts)
    if not text.strip():
        raise RuntimeError("DeepSeek 没有返回最终回答，请调整提示词或增加生成 token。")
    return DeepSeekResult(text, "".join(thoughts), usage, finish_reason, request_id, response_model)
