"""ComfyUI node for the DDHT MiniMax-H3 SLA attention patch.

The attention implementation is imported lazily so a missing or incompatible
Triton installation cannot prevent ComfyUI from starting.

Adapted in 2026 for ComfyUI-ddht-anynode from PlagueKind's
ComfyUI-H3-SLA-Attention (MIT). See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import logging


log = logging.getLogger("DDHT.SLA")


class DDHTH3SLAAttention:
    """Install block-sparse attention on a cloned MiniMax-H3 model."""

    DESCRIPTION = (
        "MiniMax-H3 专用 SLA 块稀疏注意力。请放在 LoRA 加载器之后、采样器之前；"
        "短序列和不兼容的注意力调用会自动使用原密集注意力。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sparsity_ratio": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 0.95,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": (
                            "跳过的键块比例。0.85 是 lightx2v 的原始设置；"
                            "0.90 更快但更稀疏。低于约 0.60 通常不会加速。"
                        ),
                    },
                ),
                "block_size": (
                    ["64", "128"],
                    {
                        "default": "64",
                        "tooltip": (
                            "查询块大小。64 对 H3 音频更友好；128 略快但可能降低语音质量。"
                        ),
                    },
                ),
                "min_seq_len": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 0,
                        "max": 1_000_000,
                        "step": 1024,
                        "tooltip": "短于此长度的序列自动使用原密集注意力。",
                    },
                ),
                "dense_last_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": "最后 N 个采样步骤恢复为密集注意力；0 表示全程 SLA。",
                    },
                ),
                "protect_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "保护音频",
                        "label_off": "均匀选择",
                        "tooltip": (
                            "始终保留文本、条件和音频前缀。建议有声视频保持开启。"
                        ),
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "启用 SLA",
                        "label_off": "密集旁路",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_sla"
    CATEGORY = "DDHT/Model Patches/MiniMax-H3"

    def apply_sla(
        self,
        model,
        sparsity_ratio=0.90,
        block_size="64",
        min_seq_len=8192,
        dense_last_steps=0,
        protect_audio=True,
        enabled=True,
    ):
        if not enabled:
            log.info("SLA 已禁用，MODEL 原样输出。")
            return (model,)

        try:
            from .ddht_sla import patch_h3_sla

            patched = patch_h3_sla(
                model,
                sparsity_ratio=float(sparsity_ratio),
                block_size=int(block_size),
                min_seq_len=int(min_seq_len),
                dense_last_steps=int(dense_last_steps),
                protect_audio=bool(protect_audio),
            )
        except Exception:  # Never make an optional acceleration block the run.
            log.exception(
                "SLA 补丁安装失败；将输出原模型并继续使用密集注意力。"
                "请检查 Triton、GPU 和 ComfyUI 版本。"
            )
            return (model,)

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "DDHTH3SLAAttention": DDHTH3SLAAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDHTH3SLAAttention": "MiniMax-H3 SLA Attention - DDHT",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
