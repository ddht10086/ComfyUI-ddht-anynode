"""Choose which key blocks each SLA query block may attend to.

Adapted in 2026 for ComfyUI-ddht-anynode from PlagueKind's MIT-licensed
ComfyUI-H3-SLA-Attention. The original implementation was vendored from
ModelTC/LightX2V under Apache-2.0. This version retains PlagueKind's masked-load,
minimum-top-k, fp32-pooling, protected-prefix and allocation fixes. DDHT changed
package integration and documentation. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_kernel(
    x_pointer,
    mean_pointer,
    length: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_length: tl.constexpr,
):
    block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)

    batch_index = batch_head_index // heads
    head_index = batch_head_index - batch_index * heads

    offsets_l = block_index * block_length + tl.arange(0, block_length)
    offsets_d = tl.arange(0, head_dim)

    x_offset = batch_index * length * heads * head_dim + head_index * head_dim
    mean_offset = (
        batch_head_index
        * ((length + block_length - 1) // block_length)
        * head_dim
    )
    # Masked lanes feed a sum and must be explicitly zeroed.
    x = tl.load(
        x_pointer
        + x_offset
        + offsets_l[:, None] * (heads * head_dim)
        + offsets_d[None, :],
        mask=offsets_l[:, None] < length,
        other=0.0,
    )

    valid_length = min(block_length, length - block_index * block_length)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / valid_length
    tl.store(
        mean_pointer + mean_offset + block_index * head_dim + offsets_d,
        x_mean.to(mean_pointer.dtype.element_ty),
    )


def mean_pool(x, block_length):
    """Map contiguous ``(B, L, H, D)`` to fp32 block means ``(B,H,N,D)``."""
    assert x.is_contiguous()
    batch, length, heads, head_dim = x.shape
    length_blocks = (length + block_length - 1) // block_length

    # Triton accumulates the reduction in fp32. Keeping the result in fp32
    # avoids quantizing routing scores, which matters at high sparsity.
    x_mean = torch.empty(
        (batch, heads, length_blocks, head_dim),
        device=x.device,
        dtype=torch.float32,
    )

    grid = (length_blocks, batch * heads)
    _compress_kernel[grid](
        x,
        x_mean,
        length,
        heads,
        head_dim,
        block_length,
    )
    return x_mean


def get_block_map(q, k, topk_ratio, blkq=128, blkk=128, protect_upto=0):
    """Return an int32 lookup table and number of kept key blocks.

    ``q`` and ``k`` must be contiguous ``(B, L, H, D)`` tensors. The lookup
    table is ``(B, H, ceil(LQ/blkq), topk)``. ``protect_upto`` pins the key
    blocks covering the leading N tokens into every query block without
    evicting the video blocks selected by ordinary top-k routing.
    """
    pooled_q = mean_pool(q, blkq)

    # Fold SageAttention-style smooth-k into the pooled value so no full-size
    # smoothed K allocation is needed.
    mean_k = k.mean(dim=1, dtype=torch.float32)
    pooled_k = mean_pool(k, blkk) - mean_k[:, :, None, :]

    num_q_heads = pooled_q.shape[1]
    num_kv_heads = pooled_k.shape[1]
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        pooled_k = pooled_k.repeat_interleave(
            num_q_heads // num_kv_heads,
            dim=1,
        )

    pooled_score = pooled_q @ pooled_k.transpose(-1, -2)

    key_blocks = pooled_score.shape[-1]
    topk = max(1, min(key_blocks, int(topk_ratio * key_blocks)))

    if protect_upto > 0:
        pinned = min(
            (int(protect_upto) + blkk - 1) // blkk,
            key_blocks,
        )
        if pinned > 0:
            pooled_score[..., :pinned] = float("inf")
            topk = min(key_blocks, topk + pinned)

    lookup = torch.topk(
        pooled_score,
        topk,
        dim=-1,
        sorted=False,
    ).indices
    return lookup.to(torch.int32).contiguous(), topk
