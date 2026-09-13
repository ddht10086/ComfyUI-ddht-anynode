# SPDX-License-Identifier: GPL-3.0-only
"""Dependency-free text validation nodes for ComfyUI workflows."""

from __future__ import annotations


def _prefix_error(expected_prefix):
    if not isinstance(expected_prefix, str):
        return "预期开头 expected_prefix 必须是文本。"
    if not expected_prefix:
        return "请填写预期开头 expected_prefix；空字符串会匹配所有文本，因此不允许为空。"
    return None


def _preview(value: str) -> str:
    # Bound diagnostics and escape newlines/control characters. Do not embed
    # the full (possibly very long) LLM response in our exception message.
    limit = 80
    return repr(value[:limit]) + ("…" if len(value) > limit else "")


class DDHTTextPrefixGate:
    """Pass the original text or raise to fail the current prompt execution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "连接本地 LLM 的文本输出。通过校验后原样输出。",
                    },
                ),
                "expected_prefix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "要求文本以这些字符开头；区分大小写、标点及全半角，支持实际换行，不使用正则表达式。",
                    },
                ),
                "ignore_leading_whitespace": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "忽略开头空白",
                        "label_off": "严格匹配开头",
                        "tooltip": "仅在检查时忽略输入文本开头的空格、换行、制表符等空白；不会修改输出文本或预期开头。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "check_prefix"
    CATEGORY = "DDHT/Text"
    DESCRIPTION = (
        "校验文本开头是否符合指定字符。符合时原样输出文本，不符合时报错终止当前工作流。"
        "请把后续节点连接到本节点的 text 输出。只检查开头格式，不验证全文语义。"
    )

    @classmethod
    def VALIDATE_INPUTS(cls, expected_prefix=None):
        # ComfyUI passes only constant inputs here. A linked prefix is checked
        # at runtime; an empty widget can be rejected before expensive LLM work.
        if expected_prefix is None:
            return True
        return _prefix_error(expected_prefix) or True

    def check_prefix(
        self, text: str, expected_prefix: str, ignore_leading_whitespace: bool = False
    ):
        if not isinstance(text, str):
            raise TypeError("输入 text 必须是文本，请连接 LLM 的 STRING 输出。")
        error = _prefix_error(expected_prefix)
        if error:
            raise ValueError(error)
        if not isinstance(ignore_leading_whitespace, bool):
            raise TypeError("ignore_leading_whitespace 必须是布尔值。")

        candidate = text.lstrip() if ignore_leading_whitespace else text
        if not candidate.startswith(expected_prefix):
            # A normal exception fails this queued prompt in ComfyUI's executor.
            # ExecutionBlocker would only block descendants; a global interrupt
            # flag could accidentally affect the next queued prompt.
            mode = "忽略开头空白" if ignore_leading_whitespace else "严格匹配"
            raise RuntimeError(
                f"文本开头校验失败，已终止当前工作流（{mode}）。"
                f"预期开头：{_preview(expected_prefix)}；"
                f"实际开头：{_preview(candidate)}。"
            )
        return (text,)


NODE_CLASS_MAPPINGS = {"DDHT_TextPrefixGate": DDHTTextPrefixGate}
NODE_DISPLAY_NAME_MAPPINGS = {"DDHT_TextPrefixGate": "文本开头校验 - DDHT"}
