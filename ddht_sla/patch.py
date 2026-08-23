"""Wire SLA block-sparse attention into MiniMax-H3 at inference time.

Adapted in 2026 for ComfyUI-ddht-anynode from PlagueKind's
ComfyUI-H3-SLA-Attention (MIT). Package names, logging names and wrapper keys
were changed for DDHT integration. See THIRD_PARTY_NOTICES.md.

The hook is ``transformer_options["optimized_attention_override"]``, read by
``wrap_attn`` in ``comfy/ldm/modules/attention.py``. MiniMax-H3 reaches it from
its self-attention call with q/k/v shaped ``[1, H, S, 128]``.
"""

from __future__ import annotations

import logging

import torch

from .block_map import get_block_map
from .kernel import block_sparse_attention


log = logging.getLogger("DDHT.SLA")

_H3_HEAD_DIM = 128
_OK_DTYPES = (torch.bfloat16, torch.float16)


def _new_state():
    return {
        "calls": 0,
        "dense": 0,
        "step": 0,
        "n_steps": 0,
        "last_step_index": None,
        "summarized": False,
        "seq": 0,
        "kept": 0,
        "blocks": 0,
        "pinned": 0,
        "backend": None,
        "failed": None,
    }


def _reset_run_state(state):
    """Reset per-run counters while preserving the displaced backend name."""
    state["calls"] = 0
    state["dense"] = 0
    state["step"] = 0
    state["n_steps"] = 0
    state["last_step_index"] = None
    state["summarized"] = False
    state["seq"] = 0
    state["kept"] = 0
    state["blocks"] = 0
    state["pinned"] = 0
    state["failed"] = None


def _summarise(state, sparsity, blkq, blkk):
    """Log one summary per sampling run."""
    if state["calls"] == 0:
        log.warning(
            "SLA 补丁已安装但未触发；注意力没有稀疏化（%d 次密集回退）。"
            "请确认采样器使用的是本节点输出的 MODEL。",
            state["dense"],
        )
        return

    real = 1.0 - (state["kept"] / state["blocks"]) if state["blocks"] else 0.0
    log.info(
        "SLA: %d calls | S=%d | blocks %d/%d kept (%.1f%% sparse, asked %.0f%%) "
        "| %d pinned | BLK=%dx%d | %d dense fall-throughs | displaced %s",
        state["calls"],
        state["seq"],
        state["kept"],
        state["blocks"],
        real * 100.0,
        sparsity * 100.0,
        state["pinned"],
        blkq,
        blkk,
        state["dense"],
        state["backend"] or "?",
    )
    if state["failed"] is not None:
        log.warning("SLA 内核至少一次回退到密集注意力：%s", state["failed"])


def _make_override(
    state,
    sparsity_ratio,
    blkq,
    blkk,
    min_seq_len,
    protect_audio=True,
):
    topk_ratio = 1.0 - sparsity_ratio

    def override(
        func,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        def dense():
            state["dense"] += 1
            return func(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        if state["backend"] is None:
            state["backend"] = getattr(func, "__name__", repr(func))

        transformer_options = kwargs.get("transformer_options") or {}

        # Keep every attention call that does not match MiniMax-H3's packed
        # self-attention layout on ComfyUI's original dense backend.
        if (
            not skip_reshape
            or mask is not None
            or q.ndim != 4
            or q.shape[-1] != _H3_HEAD_DIM
            or q.dtype not in _OK_DTYPES
            or q.shape[2] < min_seq_len
            or transformer_options.get("_ddht_h3sla_dense", False)
        ):
            return dense()

        try:
            batch, num_heads, sequence, head_dim = q.shape

            # H3 creates [S, H, D] then transposes for attention. Transposing
            # back to BLHD generally restores its original contiguous layout.
            qb, kb, vb = (tensor.transpose(1, 2) for tensor in (q, k, v))
            if not qb.is_contiguous():
                qb, kb, vb = qb.contiguous(), kb.contiguous(), vb.contiguous()

            prefix = (
                int(transformer_options.get("_ddht_h3sla_prefix", 0) or 0)
                if protect_audio
                else 0
            )
            if prefix >= sequence:
                prefix = 0

            lut, topk = get_block_map(
                qb,
                kb,
                topk_ratio,
                blkq,
                blkk,
                protect_upto=prefix,
            )
            out = block_sparse_attention(qb, kb, vb, lut, topk, blkq, blkk)

            state["calls"] += 1
            state["seq"] = sequence
            state["kept"] = topk
            state["blocks"] = (sequence + blkk - 1) // blkk
            state["pinned"] = (prefix + blkk - 1) // blkk

            if skip_output_reshape:
                return out.transpose(1, 2)
            return out.reshape(batch, sequence, num_heads * head_dim)
        except Exception as exc:  # A kernel problem must not kill the run.
            if state["failed"] is None:
                state["failed"] = f"{exc.__class__.__name__}: {exc}"
                log.debug("SLA 内核失败", exc_info=True)
            return dense()

    return override


def _call_next_wrapper(executor, *args, **kwargs):
    """Advance ComfyUI's wrapper chain without bypassing later wrappers."""
    if not isinstance(executor, type) and callable(executor):
        return executor(*args, **kwargs)
    # Compatibility with older test/executor shims.
    return executor.original(*args, **kwargs)


def _sequence_scalar(value):
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return float(value[0])
    return None


def _resolve_sampler_step(transformer_options):
    """Resolve the logical sampler step from the schedule and current sigma."""
    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigmas = transformer_options.get("sigmas")
    if sample_sigmas is None or current_sigmas is None:
        return None

    try:
        n_steps = len(sample_sigmas) - 1
    except TypeError:
        return None
    if n_steps < 1:
        return None

    if isinstance(sample_sigmas, (list, tuple)):
        current = _sequence_scalar(current_sigmas)
        if current is None:
            try:
                current = float(current_sigmas)
            except (TypeError, ValueError):
                return None
        try:
            values = [float(value) for value in sample_sigmas[:-1]]
        except (TypeError, ValueError):
            return None
        if not values:
            return None
        step_index = min(
            range(len(values)),
            key=lambda index: abs(values[index] - current),
        )
        return step_index, n_steps

    try:
        schedule = sample_sigmas.reshape(-1)
        current = current_sigmas.reshape(-1)
        if schedule.numel() < 2 or current.numel() == 0:
            return None
        current_value = current[0].to(device=schedule.device, dtype=schedule.dtype)
        step_index = int(torch.argmin(torch.abs(schedule[:-1] - current_value)).item())
        return step_index, int(schedule.numel()) - 1
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _prepare_run_state(state, transformer_options, sparsity_ratio, blkq, blkk):
    """Synchronize SLA's run state with the real sampler schedule."""
    resolved = _resolve_sampler_step(transformer_options)
    if resolved is None:
        n_steps = max(1, len(transformer_options.get("sample_sigmas", [])) - 1)
        if state["step"] >= n_steps:
            if not state["summarized"] and (state["calls"] or state["dense"]):
                _summarise(state, sparsity_ratio, blkq, blkk)
            _reset_run_state(state)
        state["n_steps"] = n_steps
        state["step"] += 1
        return n_steps

    step_index, n_steps = resolved
    last_step_index = state["last_step_index"]
    new_run = (
        state["n_steps"] not in (0, n_steps)
        or (
            last_step_index is not None
            and (
                step_index < last_step_index
                or (state["summarized"] and step_index <= last_step_index)
            )
        )
    )

    if new_run:
        if not state["summarized"] and (state["calls"] or state["dense"]):
            _summarise(state, sparsity_ratio, blkq, blkk)
        _reset_run_state(state)

    state["n_steps"] = n_steps
    state["step"] = step_index + 1
    state["last_step_index"] = step_index
    return n_steps


def _make_wrapper(state, sparsity_ratio, blkq, blkk, dense_last_steps):
    """Track sampler state and expose H3's protected prefix to attention."""

    def wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        minimax_payload=None,
        **kwargs,
    ):
        if transformer_options is None:
            transformer_options = {}

        n_steps = _prepare_run_state(
            state,
            transformer_options,
            sparsity_ratio,
            blkq,
            blkk,
        )

        # Packed H3 order is [text | cond/ref | audio | video]. Keep everything
        # before the video segment available to each sparse query block.
        prefix = 0
        layout = minimax_payload.get("layout") if minimax_payload else None
        for segment in getattr(layout, "segments", ()) or ():
            if len(segment) == 3 and segment[2] == "video":
                prefix = int(segment[0])
                break
        transformer_options["_ddht_h3sla_prefix"] = prefix
        transformer_options["_ddht_h3sla_dense"] = bool(
            dense_last_steps > 0
            and state["step"] > n_steps - dense_last_steps
        )

        # Forward the H3-only payload only when the model supplied it. This
        # keeps wiring the node to a non-H3 model a graceful dense no-op.
        if minimax_payload is not None:
            kwargs["minimax_payload"] = minimax_payload

        out = _call_next_wrapper(
            executor,
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            **kwargs,
        )

        if state["step"] >= n_steps and not state["summarized"]:
            _summarise(state, sparsity_ratio, blkq, blkk)
            state["summarized"] = True
        return out

    return wrapper


def patch_h3_sla(
    model,
    sparsity_ratio=0.90,
    block_size=64,
    min_seq_len=8192,
    dense_last_steps=0,
    protect_audio=True,
):
    """Return a clone whose compatible MiniMax-H3 attention runs sparse."""
    blkq = int(block_size)
    if blkq not in (64, 128):
        raise ValueError("block_size must be 64 or 128")

    # 128x64 fits consumer Blackwell shared-memory limits; 128x128 may not.
    blkk = 64 if blkq == 128 else blkq

    state = _new_state()
    patched = model.clone()

    transformer_options = patched.model_options.get("transformer_options", {}).copy()
    transformer_options["optimized_attention_override"] = _make_override(
        state,
        float(sparsity_ratio),
        blkq,
        blkk,
        int(min_seq_len),
        bool(protect_audio),
    )
    patched.model_options["transformer_options"] = transformer_options

    patched.add_wrapper_with_key(
        "diffusion_model",
        "ddht_h3_sla_state",
        _make_wrapper(
            state,
            float(sparsity_ratio),
            blkq,
            blkk,
            int(dense_last_steps),
        ),
    )

    log.info(
        "SLA 已安装 | sparsity=%.2f | BLK=%dx%d | min_seq_len=%d | "
        "dense_last_steps=%d | protect_audio=%s",
        sparsity_ratio,
        blkq,
        blkk,
        min_seq_len,
        dense_last_steps,
        protect_audio,
    )
    return patched
