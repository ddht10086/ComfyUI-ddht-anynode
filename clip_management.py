# SPDX-License-Identifier: GPL-3.0-only
"""Workflow-safe CLIP/text-encoder VRAM management for ComfyUI."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import comfy.model_management as model_management


CATEGORY = "DDHT/Memory"
MODE_CPU = "卸载到 CPU 内存"
MODE_FULL = "完全卸载 CLIP（释放显存缓存）"


def _is_cpu_device(device: Any) -> bool:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return device_type == "cpu"
    return str(device).split(":", 1)[0].lower() == "cpu"


def _get_patcher(clip: Any) -> Any:
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise TypeError(
            "The connected CLIP object has no ComfyUI ModelPatcher and cannot be unloaded safely."
        )
    return patcher


def _same_model_family(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_uuid = getattr(left, "clone_base_uuid", None)
    right_uuid = getattr(right, "clone_base_uuid", None)
    return left_uuid is not None and left_uuid == right_uuid


def _loaded_clip_clones(target: Any) -> list[Any]:
    """Return the target plus any currently registered patcher clones."""
    loaded_models = getattr(model_management, "loaded_models", None)
    candidates: Iterable[Any] = ()
    if callable(loaded_models):
        try:
            candidates = loaded_models()
        except Exception as exc:
            logging.debug("[DDHT] Could not enumerate loaded models: %s", exc)

    result = []
    seen = set()
    for patcher in (*tuple(candidates), target):
        if patcher is None or not _same_model_family(patcher, target):
            continue
        # Clones can share the same underlying module. Offload it only once.
        model_identity = id(getattr(patcher, "model", patcher))
        if model_identity in seen:
            continue
        seen.add(model_identity)
        result.append(patcher)
    return result


def _loaded_bytes(patcher: Any) -> int:
    loaded_size = getattr(patcher, "loaded_size", None)
    if not callable(loaded_size):
        return 0
    try:
        return max(0, int(loaded_size() or 0))
    except Exception:
        return 0


def _empty_accelerator_cache() -> None:
    empty_cache = getattr(model_management, "soft_empty_cache", None)
    if not callable(empty_cache):
        return
    try:
        empty_cache(force=True)
    except TypeError:
        # Older ComfyUI versions do not expose the force argument.
        empty_cache()


def _offload_to_cpu(target: Any) -> tuple[int, int]:
    """Move resident CLIP weights to CPU while keeping its managed state reusable."""
    patchers = _loaded_clip_clones(target)
    for patcher in patchers:
        offload_device = getattr(patcher, "offload_device", None)
        if not _is_cpu_device(offload_device):
            raise RuntimeError(
                "CLIP cannot be offloaded to CPU because its offload_device is not CPU. "
                "If ComfyUI was started with --gpu-only, remove that option and restart ComfyUI."
            )

    before = sum(_loaded_bytes(patcher) for patcher in patchers)
    for patcher in patchers:
        offload_device = patcher.offload_device
        move_patches = getattr(patcher, "model_patches_to", None)
        if callable(move_patches):
            move_patches(offload_device)

        partial_unload = getattr(patcher, "partially_unload", None)
        if not callable(partial_unload):
            # Very old/custom patchers need the official full-unload fallback.
            return _fully_unload(target, before_override=before)
        partial_unload(offload_device, 1e32)

    after = sum(_loaded_bytes(patcher) for patcher in patchers)
    _empty_accelerator_cache()
    return max(0, before - after), len(patchers)


def _fully_unload(target: Any, before_override: int | None = None) -> tuple[int, int]:
    """Detach a CLIP family from ComfyUI's loaded-model registry."""
    patchers = _loaded_clip_clones(target)
    before = (
        sum(_loaded_bytes(patcher) for patcher in patchers)
        if before_override is None
        else before_override
    )

    unload = getattr(model_management, "unload_model_and_clones", None)
    if callable(unload):
        try:
            unload(target, unload_additional_models=True, all_devices=True)
        except TypeError:
            # Compatibility with ComfyUI versions before all_devices was added.
            try:
                unload(target, unload_additional_models=True)
            except TypeError:
                unload(target)
    else:
        legacy_unload = getattr(model_management, "unload_model_clones", None)
        if callable(legacy_unload):
            legacy_unload(target)
        else:
            # Last-resort compatibility for old/custom ComfyUI builds.
            detach = getattr(target, "detach", None)
            if not callable(detach):
                raise RuntimeError(
                    "This ComfyUI version does not expose a supported CLIP unload API."
                )
            detach()

    _empty_accelerator_cache()
    return before, len(patchers)


class DDHTUnloadCLIP:
    """Unload CLIP only after all connected conditioning has been encoded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "conditioning_1": (
                    "CONDITIONING",
                    {
                        "tooltip": "连接正向条件，确保文本编码完成后才卸载 CLIP。",
                    },
                ),
                "mode": (
                    [MODE_CPU, MODE_FULL],
                    {
                        "default": MODE_CPU,
                        "tooltip": (
                            "CPU 模式保留可快速重新加载的管理状态；完全卸载模式会从 "
                            "ComfyUI 已加载模型列表分离 CLIP 并清理显存缓存。"
                        ),
                    },
                ),
            },
            "optional": {
                "conditioning_2": (
                    "CONDITIONING",
                    {
                        "tooltip": "可连接负向条件；连接后会等待正、负条件都编码完成。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning_1", "conditioning_2", "unload_info")
    OUTPUT_TOOLTIPS = (
        "原样输出的第一组条件。",
        "原样输出的第二组条件；未连接输入时为 None。",
        "本次 CLIP 卸载模式及估算释放量。",
    )
    FUNCTION = "unload_clip"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "在已连接的条件完成编码后卸载 CLIP/text encoder，再把条件原样传给采样流程。"
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # The CLIP may have been reloaded elsewhere even when conditioning is cached.
        # Force this side-effect node to run for every queued workflow execution.
        return float("nan")

    def unload_clip(
        self,
        clip: Any,
        conditioning_1: Any,
        mode: str,
        conditioning_2: Any = None,
    ):
        patcher = _get_patcher(clip)
        if mode == MODE_CPU:
            freed_bytes, patcher_count = _offload_to_cpu(patcher)
        elif mode == MODE_FULL:
            freed_bytes, patcher_count = _fully_unload(patcher)
        else:
            raise ValueError(f"Unsupported CLIP unload mode: {mode}")

        freed_mb = freed_bytes / (1024 * 1024)
        info = (
            f"{mode}：处理 {patcher_count} 个 CLIP 模型实例，"
            f"估算释放 {freed_mb:.1f} MB 显存。"
        )
        logging.info("[DDHT] %s", info)
        return (conditioning_1, conditioning_2, info)


NODE_CLASS_MAPPINGS = {
    "DDHT_UnloadCLIP": DDHTUnloadCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DDHT_UnloadCLIP": "流程中卸载 CLIP - DDHT",
}
