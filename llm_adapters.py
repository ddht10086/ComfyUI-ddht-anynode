# SPDX-License-Identifier: GPL-3.0-only
"""HTTP adapters for local multimodal LLM servers.

The module deliberately has no ComfyUI imports so the protocol handling can be
tested independently.  It supports OpenAI-compatible SSE streams and Ollama's
native NDJSON stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

import requests


OPENAI_FRAMEWORKS = {"llama.cpp", "vllm", "sglang", "openai兼容"}


@dataclass
class RequestSpec:
    framework: str
    endpoint: str
    headers: Dict[str, str]
    payload: Dict[str, Any]
    stream_kind: str


@dataclass
class GenerationResult:
    content: str
    reasoning: str
    usage: Dict[str, Any]
    finish_reason: Optional[str]
    reached_character_limit: bool
    generated_character_count: int


class _CharacterBudget:
    def __init__(self, maximum: int):
        self.maximum = max(0, int(maximum))
        self.count = 0
        self.reached = False

    def take(self, text: Any) -> str:
        piece = _text_from_content(text)
        if not piece or self.reached:
            return ""
        if self.maximum == 0:
            self.count += len(piece)
            return piece

        remaining = self.maximum - self.count
        if remaining <= 0:
            self.reached = True
            return ""
        accepted = piece[:remaining]
        self.count += len(accepted)
        if len(piece) > remaining:
            self.reached = True
        return accepted


def _text_from_content(value: Any) -> str:
    """Extract text from string or OpenAI content-part arrays."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value)


def _normalized_framework(framework: str) -> str:
    value = framework.strip()
    aliases = {
        "ollama": "ollama",
        "vllm": "vllm",
        "sglang": "sglang",
        "openai-compatible": "openai兼容",
        "openai compatible": "openai兼容",
        "openai兼容": "openai兼容",
        "llama.cpp": "llama.cpp",
    }
    normalized = aliases.get(value.lower())
    if normalized is None:
        raise ValueError(f"不支持的本地 LLM 框架：{framework}")
    return normalized


def build_endpoint(framework: str, base_url: str) -> str:
    """Turn a server base URL (or a full endpoint URL) into an endpoint."""
    normalized = _normalized_framework(framework)
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API 地址必须是完整的 http:// 或 https:// 地址。")
    if parsed.username or parsed.password:
        raise ValueError("请不要把用户名或密码写进 API 地址；请改用 API 密钥环境变量。")
    if parsed.query or parsed.fragment:
        raise ValueError("API 地址不能包含查询参数或锚点。")

    path = parsed.path.rstrip("/")
    if normalized == "ollama":
        if path.endswith("/api/chat"):
            endpoint_path = path
        elif path.endswith("/api"):
            endpoint_path = f"{path}/chat"
        elif path.endswith("/v1"):
            endpoint_path = f"{path[:-3]}/api/chat"
        else:
            endpoint_path = f"{path}/api/chat"
    else:
        if path.endswith("/chat/completions"):
            endpoint_path = path
        elif path.endswith("/v1"):
            endpoint_path = f"{path}/chat/completions"
        else:
            endpoint_path = f"{path}/v1/chat/completions"

    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def _merged_parameters(defaults: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(defaults)
    for key, value in extra.items():
        if key in {"messages", "stream"}:
            continue
        if key == "options" and isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def build_request(
    *,
    framework: str,
    base_url: str,
    model: str,
    prompt: str,
    system_prompt: str,
    images_base64: Iterable[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    thinking_mode: str,
    reasoning_effort: str,
    api_key: str = "",
    extra_parameters: Optional[Mapping[str, Any]] = None,
) -> RequestSpec:
    """Build a framework-specific streaming request without sending it."""
    normalized = _normalized_framework(framework)
    endpoint = build_endpoint(normalized, base_url)
    images = list(images_base64)
    extra = dict(extra_parameters or {})
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if normalized == "ollama":
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_message: Dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images
        messages.append(user_message)
        defaults: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "repeat_penalty": float(repetition_penalty),
                "seed": int(seed),
            },
        }
        if thinking_mode != "自动":
            defaults["think"] = thinking_mode == "开启"
        payload = _merged_parameters(defaults, extra)
        payload["messages"] = messages
        payload["stream"] = True
        return RequestSpec(normalized, endpoint, headers, payload, "ollama_ndjson")

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    user_content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }
        for image in images
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    defaults = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": int(seed),
    }
    if normalized in {"llama.cpp", "vllm", "sglang"}:
        defaults["top_k"] = int(top_k)
    if normalized == "llama.cpp":
        defaults["repeat_penalty"] = float(repetition_penalty)
    elif normalized in {"vllm", "sglang"}:
        defaults["repetition_penalty"] = float(repetition_penalty)
    if normalized == "llama.cpp":
        if thinking_mode != "自动":
            defaults["chat_template_kwargs"] = {
                "enable_thinking": thinking_mode == "开启"
            }
        if reasoning_effort != "自动":
            defaults["reasoning_effort"] = reasoning_effort

    payload = _merged_parameters(defaults, extra)
    payload["messages"] = messages
    payload["stream"] = True
    return RequestSpec(normalized, endpoint, headers, payload, "openai_sse")


def serialized_payload_size(spec: RequestSpec) -> int:
    return len(json.dumps(spec.payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _error_message(response: requests.Response) -> str:
    text = response.text[:2000].strip()
    try:
        body = response.json()
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                return str(error.get("message") or error)
            if error:
                return str(error)
            if body.get("detail"):
                return str(body["detail"])
            if body.get("message"):
                return str(body["message"])
    except (ValueError, TypeError):
        pass
    return text or "服务器没有返回错误详情"


def _iter_decoded_lines(response: requests.Response):
    # A small chunk keeps cancellation and output-limit handling responsive
    # even when a server emits very small token events.
    for raw_line in response.iter_lines(chunk_size=64):
        if not raw_line:
            continue
        if isinstance(raw_line, bytes):
            yield raw_line.decode("utf-8", errors="replace").strip()
        else:
            yield str(raw_line).strip()


def execute_request(
    spec: RequestSpec,
    *,
    connect_timeout: float,
    read_timeout: float,
    max_output_characters: int,
    on_character_limit: str,
    interrupted: Optional[Callable[[], bool]] = None,
    interrupt_exception: Optional[Callable[[], BaseException]] = None,
) -> GenerationResult:
    """Send a request and consume its stream until completion or cancellation."""
    response: Optional[requests.Response] = None
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    usage: Dict[str, Any] = {}
    finish_reason: Optional[str] = None
    budget = _CharacterBudget(max_output_characters)

    def append(target: List[str], value: Any):
        accepted = budget.take(value)
        if accepted:
            target.append(accepted)

    try:
        try:
            response = requests.post(
                spec.endpoint,
                headers=spec.headers,
                json=spec.payload,
                stream=True,
                timeout=(float(connect_timeout), float(read_timeout)),
            )
        except requests.ConnectTimeout as exc:
            raise RuntimeError(f"连接本地 LLM 超时：{spec.endpoint}") from exc
        except requests.ReadTimeout as exc:
            raise RuntimeError(f"等待本地 LLM 生成结果超时：{spec.endpoint}") from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"无法连接本地 LLM：{spec.endpoint}。请检查服务是否启动、地址、端口和局域网防火墙。"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"本地 LLM 请求失败：{exc}") from exc

        if response.status_code >= 400:
            message = _error_message(response)
            raise RuntimeError(
                f"{spec.framework} API 请求失败（HTTP {response.status_code}）：{message}"
            )

        for line in _iter_decoded_lines(response):
            if interrupted and interrupted():
                if interrupt_exception:
                    raise interrupt_exception()
                raise RuntimeError("ComfyUI 已中断本地 LLM 推理。")

            if spec.stream_kind == "ollama_ndjson":
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Ollama 返回了无法解析的流数据：{line[:300]}") from exc
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama 推理失败：{chunk['error']}")
                message = chunk.get("message") or {}
                append(reasoning_parts, message.get("thinking") or chunk.get("thinking"))
                if not budget.reached:
                    append(content_parts, message.get("content"))
                if chunk.get("done"):
                    finish_reason = chunk.get("done_reason") or "stop"
                    usage = {
                        key: chunk[key]
                        for key in (
                            "total_duration",
                            "load_duration",
                            "prompt_eval_count",
                            "prompt_eval_duration",
                            "eval_count",
                            "eval_duration",
                        )
                        if key in chunk
                    }
            else:
                if line.startswith(":"):
                    continue
                data = line[5:].strip() if line.startswith("data:") else line
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"OpenAI 兼容接口返回了无法解析的流数据：{data[:300]}") from exc
                if isinstance(chunk.get("error"), Mapping):
                    raise RuntimeError(
                        f"{spec.framework} 推理失败：{chunk['error'].get('message') or chunk['error']}"
                    )
                if isinstance(chunk.get("usage"), Mapping):
                    usage = dict(chunk["usage"])
                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta") or choice.get("message") or {}
                    append(
                        reasoning_parts,
                        delta.get("reasoning_content") or delta.get("reasoning"),
                    )
                    if not budget.reached:
                        append(content_parts, delta.get("content"))
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])

            if budget.reached:
                break

    except requests.ReadTimeout as exc:
        raise RuntimeError(f"等待本地 LLM 生成结果超时：{spec.endpoint}") from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"读取本地 LLM 流时连接中断：{spec.endpoint}。请检查服务日志和局域网连接。"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"读取本地 LLM 流失败：{exc}") from exc
    finally:
        if response is not None:
            response.close()

    if budget.reached and on_character_limit == "报错终止流程":
        raise RuntimeError(
            f"本地 LLM 输出达到最大字符数 {max_output_characters}，已主动断开生成连接并终止流程。"
        )

    return GenerationResult(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        usage=usage,
        finish_reason=finish_reason,
        reached_character_limit=budget.reached,
        generated_character_count=budget.count,
    )
