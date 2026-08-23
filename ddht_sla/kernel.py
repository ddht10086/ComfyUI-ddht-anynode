"""Triton forward kernel for MiniMax-H3 SLA block-sparse attention.

Adapted in 2026 for ComfyUI-ddht-anynode from PlagueKind's MIT-licensed
ComfyUI-H3-SLA-Attention. The original kernel was vendored and reduced from
ModelTC/LightX2V under Apache-2.0. This version retains PlagueKind's BLHD layout,
masked-load fixes and GPU-resource launch ladder. DDHT changed package
integration, names and documentation. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_forward(
    q_pointer,
    k_pointer,
    v_pointer,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    lookup_pointer,
    output_pointer,
    heads: tl.constexpr,
    query_length: tl.constexpr,
    key_length: tl.constexpr,
    query_blocks: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    query_block_index = tl.program_id(0).to(tl.int64)
    batch_head_index = tl.program_id(1).to(tl.int64)

    batch_index = batch_head_index // heads
    head_index = batch_head_index % heads
    hidden_dim: tl.constexpr = heads * head_dim

    q_offset = batch_index * query_length * hidden_dim + head_index * head_dim
    kv_offset = batch_index * key_length * hidden_dim + head_index * head_dim
    lookup_offset = (
        batch_head_index * query_blocks + query_block_index
    ) * topk

    offsets_m = query_block_index * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)

    q_pointers = (
        q_pointer
        + q_offset
        + offsets_m[:, None] * hidden_dim
        + offsets_d[None, :]
    )
    output_pointers = (
        output_pointer
        + q_offset
        + offsets_m[:, None] * hidden_dim
        + offsets_d[None, :]
    )
    lookup_row = lookup_pointer + lookup_offset

    running_max = tl.full([block_m], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([block_m], dtype=tl.float32)
    output_accumulator = tl.zeros([block_m, head_dim], dtype=tl.float32)

    # Masked values must be zero because undefined lanes can otherwise turn a
    # sequence-tail row into NaN during the dot products.
    q = tl.load(
        q_pointers,
        mask=offsets_m[:, None] < query_length,
        other=0.0,
    )

    for lookup_index in tl.range(topk):
        key_block_index = tl.load(lookup_row + lookup_index).to(tl.int64)
        key_start = key_block_index * block_n
        key_mask = (key_start + offsets_n) < key_length

        k_pointers = (
            k_pointer
            + kv_offset
            + (key_start + offsets_n)[None, :] * hidden_dim
            + offsets_d[:, None]
        )
        v_pointers = (
            v_pointer
            + kv_offset
            + (key_start + offsets_n)[:, None] * hidden_dim
            + offsets_d[None, :]
        )

        k = tl.load(k_pointers, mask=key_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        qk = tl.where(key_mask[None, :], qk, float("-inf"))

        v = tl.load(v_pointers, mask=key_mask[:, None], other=0.0)
        local_max = tl.max(qk, 1)
        new_max = tl.maximum(running_max, local_max)
        qk = qk - new_max[:, None]

        probabilities = tl.math.exp2(qk)
        local_sum = tl.sum(probabilities, 1)
        correction = tl.math.exp2(running_max - new_max)
        output_accumulator = output_accumulator * correction[:, None]
        output_accumulator += tl.dot(probabilities.to(v.dtype), v)

        running_sum = running_sum * correction + local_sum
        running_max = new_max

    output_accumulator = output_accumulator / running_sum[:, None]
    tl.store(
        output_pointers,
        output_accumulator.to(output_pointer.type.element_ty),
        mask=offsets_m[:, None] < query_length,
    )


# Consumer Blackwell has less shared memory than the datacenter GPUs the
# upstream kernel targeted. Probe launch configurations and remember the first
# one that fits each tile/head dimension.
_LAUNCH_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
}
_CHOSEN_CONFIG = {}


def block_sparse_attention(
    q,
    k,
    v,
    lookup,
    topk,
    block_m,
    block_n,
    qk_scale=None,
):
    """Attend each query block only to key blocks named by ``lookup``."""
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert lookup.is_contiguous()
    assert block_m in (64, 128) and block_n in (64, 128)

    batch, query_length, heads, head_dim = q.shape
    key_length = k.shape[1]
    if qk_scale is None:
        qk_scale = head_dim**-0.5

    query_blocks = triton.cdiv(query_length, block_m)
    output = torch.empty_like(q)
    grid = (query_blocks, batch * heads)

    config_key = (block_m, block_n, head_dim)
    if config_key in _CHOSEN_CONFIG:
        ladder = (_CHOSEN_CONFIG[config_key],)
    else:
        ladder = _LAUNCH_LADDER[(block_m, block_n)]

    last_error = None
    for num_warps, num_stages in ladder:
        try:
            _attention_forward[grid](
                q,
                k,
                v,
                qk_scale,
                topk,
                lookup,
                output,
                heads,
                query_length,
                key_length,
                query_blocks,
                head_dim,
                block_m,
                block_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        except triton.runtime.errors.OutOfResources as exc:
            last_error = exc
            continue

        _CHOSEN_CONFIG[config_key] = (num_warps, num_stages)
        return output

    if last_error is not None:
        raise last_error
    raise RuntimeError("No viable Triton SLA launch configuration")
