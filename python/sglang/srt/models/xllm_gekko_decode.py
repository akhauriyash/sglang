# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Fixed-shape Gekko decode over the serving allocator's recurrent state.

The functional recurrence remains responsible for prefill and block commits.
This path appends one token per request without copying retained histories.
Slots and absolute positions are device inputs, including during graph replay.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .xllm_rmsnorm import native_rms_norm
from .xllm_sca import native_sca_decode
from .xllm_tdn import native_tdn_decode


@triton.jit
def _decode_metadata(
    Slots, SeqLens, Positions, Lengths, B: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row, row < B, -1)
    position = tl.load(SeqLens + row, row < B, 1) - 1
    retained = tl.where(position < 4096, position, 4096 + position % 4096)
    tl.store(Positions + row, tl.where(slot >= 0, position, 0), row < B)
    tl.store(Lengths + row, tl.where(slot >= 0, retained + 1, 0), row < B)


def prepare_decode_metadata(slots, seq_lens, positions, lengths):
    _decode_metadata[(1,)](
        slots,
        seq_lens,
        positions,
        lengths,
        len(slots),
        triton.next_power_of_2(len(slots)),
    )


@triton.jit
def _append_histories(
    K,
    V,
    Features,
    Raw,
    KPool,
    VPool,
    FeaturePool,
    RawPool,
    Slots,
    Lengths,
    DIM: tl.constexpr,
    CAPACITY: tl.constexpr,
    NSLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    index = tl.load(Lengths + row) - 1
    valid = (slot >= 0) & (slot < NSLOTS) & (index >= 0) & (index < CAPACITY)
    source = row * DIM + channel
    destination = (slot * CAPACITY + index) * DIM + channel
    mask = valid & (channel < DIM)
    tl.store(KPool + destination, tl.load(K + source, channel < DIM, 0), mask)
    tl.store(VPool + destination, tl.load(V + source, channel < DIM, 0), mask)
    tl.store(
        FeaturePool + destination, tl.load(Features + source, channel < DIM, 0), mask
    )
    tl.store(RawPool + destination, tl.load(Raw + source, channel < DIM, 0), mask)


@triton.jit
def _gather_memories(
    Pool,
    Slots,
    Output,
    STRIDE: tl.constexpr,
    NSLOTS: tl.constexpr,
    QHEADS: tl.constexpr,
    KVHEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + row)
    kvhead = head // (QHEADS // KVHEADS)
    value = tl.load(
        Pool + slot * STRIDE + kvhead * DIM * DIM + offset,
        (slot >= 0) & (slot < NSLOTS) & (offset < DIM * DIM),
        0,
    )
    tl.store(
        Output + (row * QHEADS + head) * DIM * DIM + offset, value, offset < DIM * DIM
    )


def _project_rows(projection, value):
    # Preserve the qualified per-request GEMM shape. Larger batch GEMMs may
    # choose different BF16 accumulation algorithms and need separate parity.
    if len(value) == 1:
        return projection(value)
    return torch.cat([projection(value[row : row + 1]) for row in range(len(value))])


@torch.no_grad()
def pooled_gekko_decode(layer, hidden, cache, slots, positions, lengths):
    """Advance ordinary decode rows in place; negative slots are padding.

    Pools use the existing Gekko cache layout. Every real row must have a
    distinct slot and initialized state; positions divisible by 256 minus one
    must use the functional boundary path instead. No state is allocated here.
    """
    if (
        layer.model_dim,
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        layer.v_head_dim,
        layer.chunk_size,
        layer.timenorm.num_groups,
    ) != (1536, 32, 8, 64, 64, 4096, 32):
        raise ValueError("Pooled Gekko decode requires the native 1536-channel layout")
    heads, kvheads, dim = layer.num_heads, layer.num_kv_heads, layer.head_dim
    memory_size = kvheads * dim * dim
    mean_offset = memory_size + kvheads * dim
    z = native_tdn_decode(
        hidden,
        layer.timenorm.weight,
        layer.timenorm.bias,
        cache.temporal,
        slots,
        positions,
        packed_offset=mean_offset + 64,
        mean_offset=mean_offset,
        var_offset=mean_offset + 32,
        beta1=layer.timenorm.beta1,
        beta2=layer.timenorm.beta2,
        eps=layer.timenorm.eps,
    )
    raw_q = layer.q_conv.decode_pooled(_project_rows(layer.wq, z), cache.conv[0], slots)
    raw_k = layer.k_conv.decode_pooled(_project_rows(layer.wk, z), cache.conv[1], slots)
    value = layer.v_conv.decode_pooled(_project_rows(layer.wv, z), cache.conv[2], slots)
    if layer.value_activation == "silu":
        value = F.silu(value)
    query_features = raw_q.view(-1, heads, dim).float().softmax(-1).to(hidden.dtype)
    key_features = raw_k.view(-1, kvheads, dim).float().softmax(-1).to(hidden.dtype)
    q = layer.query_norm(raw_q).view(-1, heads, dim)
    k = layer.key_norm(raw_k).view(-1, kvheads, dim)
    q, k = layer.rotary(q, k, positions)
    capacity = cache.conv[3].shape[1]
    _append_histories[(len(hidden),)](
        k,
        value,
        key_features,
        raw_k,
        *cache.conv[3:7],
        slots,
        lengths,
        kvheads * dim,
        capacity,
        cache.temporal.shape[0],
        triton.next_power_of_2(kvheads * dim),
    )
    sca = native_sca_decode(
        q,
        cache.conv[3].view(-1, capacity, kvheads, dim),
        cache.conv[4].view(-1, capacity, kvheads, dim),
        slots,
        lengths,
        dim**-0.5,
    )
    memories = hidden.new_empty((len(hidden), heads, dim, dim))
    _gather_memories[(len(hidden), heads, triton.cdiv(dim * dim, 256))](
        cache.temporal,
        slots,
        memories,
        cache.temporal.numel() // cache.temporal.shape[0],
        cache.temporal.shape[0],
        heads,
        kvheads,
        dim,
        256,
    )
    awm = torch.bmm(query_features.view(-1, 1, dim), memories.view(-1, dim, dim)).view(
        -1, heads, dim
    )
    combined = (sca + awm).flatten(1)
    normalized = native_rms_norm(combined, None, layer.norm_eps, heads)
    gate = F.softplus(_project_rows(layer.wr, z), beta=math.log(2.0))
    return _project_rows(layer.wo, normalized * gate)
