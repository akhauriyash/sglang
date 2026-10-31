# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native-order BF16 Gekko short convolution for NVIDIA CUDA inference.

Weight softmax and BF16 rounding remain Torch operations. Prefill returns an
independently owned next history; one-token decode updates request-owned pools.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _shortconv(
    X,
    HISTORY,
    TAPS,
    Y,
    NEXT,
    LENGTH: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    time, feature = index // DIM, index % DIM
    output_valid = index < LENGTH * DIM
    accumulator = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(WIDTH):
        joined_time = time + tap
        from_history = joined_time < WIDTH - 1
        old = tl.load(
            HISTORY + joined_time * HS0 + feature * HS1,
            mask=output_valid & from_history,
            other=0,
        ).to(tl.float32)
        fresh = tl.load(
            X + (joined_time - WIDTH + 1) * XS0 + feature * XS1,
            mask=output_valid & ~from_history,
            other=0,
        ).to(tl.float32)
        value = tl.where(from_history, old, fresh)
        coefficient = tl.load(
            TAPS + feature * WS0 + tap * WS1,
            mask=output_valid,
            other=0,
        ).to(tl.float32)
        # Keep native chronological FP32 additions; the launcher disables FMA.
        accumulator = accumulator + value * coefficient
    tl.store(Y + index, accumulator.to(tl.bfloat16), mask=output_valid)
    if WIDTH > 1:
        history_valid = index < (WIDTH - 1) * DIM
        joined_time = LENGTH + time
        from_history = joined_time < WIDTH - 1
        old = tl.load(
            HISTORY + joined_time * HS0 + feature * HS1,
            mask=history_valid & from_history,
            other=0,
        )
        fresh = tl.load(
            X + (joined_time - WIDTH + 1) * XS0 + feature * XS1,
            mask=history_valid & ~from_history,
            other=0,
        )
        tl.store(NEXT + index, tl.where(from_history, old, fresh), mask=history_valid)


def native_shortconv_forward(x, weight, history, *, normalized_weight=None):
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("Gekko ShortConv requires autocast to be disabled")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, history)):
        raise ValueError("Gekko ShortConv is inference-only")
    if not x.is_cuda or torch.version.hip is not None or x.dtype != torch.bfloat16:
        raise ValueError("Gekko ShortConv requires NVIDIA CUDA BF16 activations")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected x[T,D] and weight[D,W]")
    length, dimension = x.shape
    width = weight.shape[1]
    if dimension <= 0 or not 1 <= width <= 4 or weight.shape[0] != dimension:
        raise ValueError("Expected positive channels and width in [1,4]")
    if history.shape != (width - 1, dimension):
        raise ValueError("History must have shape [W-1,D]")
    if any(t.device != x.device or t.dtype != x.dtype for t in (weight, history)):
        raise ValueError("Input, history, and learned weights must share BF16 device")
    if length == 0:
        return x, history
    if normalized_weight is None:
        taps = weight.float().softmax(-1).to(x.dtype)
    else:
        taps = normalized_weight
        if (
            taps.shape != weight.shape
            or taps.device != x.device
            or taps.dtype != x.dtype
            or (torch.is_grad_enabled() and taps.requires_grad)
        ):
            raise ValueError("Normalized weights must match the learned BF16 weights")
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    next_history = x.new_empty((width - 1, dimension))
    _shortconv[(triton.cdiv(max(length, width - 1) * dimension, 256),)](
        x,
        history,
        taps,
        output,
        next_history,
        length,
        dimension,
        width,
        *x.stride(),
        *history.stride(),
        *taps.stride(),
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, next_history


@triton.jit
def _shortconv_decode_pooled(
    X,
    TAPS,
    HISTORY,
    SLOTS,
    Y,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    SLOT_COUNT: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    HS0: tl.constexpr,
    HS1: tl.constexpr,
    SS0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    feature = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row * SS0)
    slot_valid = (slot >= 0) & (slot < SLOT_COUNT)
    slot = tl.where(slot_valid, slot, 0)
    valid = slot_valid & (feature < DIM)
    history = HISTORY + slot * HS0 + feature
    fresh = tl.load(X + row * XS0 + feature * XS1, mask=valid, other=0).to(tl.float32)
    # Each lane owns one feature of one distinct request slot. Load every old
    # value before any write; no program reads another program's history cells.
    if WIDTH > 1:
        old0 = tl.load(history, mask=valid, other=0).to(tl.float32)
    if WIDTH > 2:
        old1 = tl.load(history + HS1, mask=valid, other=0).to(tl.float32)
    if WIDTH > 3:
        old2 = tl.load(history + 2 * HS1, mask=valid, other=0).to(tl.float32)

    accumulator = tl.full((BLOCK,), 0, tl.float32)
    if WIDTH > 1:
        tap0 = tl.load(TAPS + feature * WS0, mask=valid, other=0).to(tl.float32)
        accumulator = accumulator + old0 * tap0
    if WIDTH > 2:
        tap1 = tl.load(TAPS + feature * WS0 + WS1, mask=valid, other=0).to(tl.float32)
        accumulator = accumulator + old1 * tap1
    if WIDTH > 3:
        tap2 = tl.load(TAPS + feature * WS0 + 2 * WS1, mask=valid, other=0).to(
            tl.float32
        )
        accumulator = accumulator + old2 * tap2
    last = tl.load(TAPS + feature * WS0 + (WIDTH - 1) * WS1, mask=valid, other=0).to(
        tl.float32
    )
    accumulator = accumulator + fresh * last
    tl.store(Y + row * DIM + feature, accumulator.to(tl.bfloat16), mask=feature < DIM)

    if WIDTH > 1:
        tl.store(history, old1 if WIDTH > 2 else fresh, mask=valid)
    if WIDTH > 2:
        tl.store(history + HS1, old2 if WIDTH > 3 else fresh, mask=valid)
    if WIDTH > 3:
        tl.store(history + 2 * HS1, fresh, mask=valid)


def native_shortconv_decode_pooled(x, normalized_weight, history_pool, cache_slots):
    """Apply one token per row and shift valid histories in place.

    ``history_pool[S,W-1,D]`` must be contiguous. Valid slots in the device
    vector ``cache_slots[B]`` must be distinct; the serving allocator enforces
    that invariant before capture. Negative/out-of-range slots produce zero
    output and never access or modify a history. No slot values reach the CPU.
    ``normalized_weight[D,W]`` is the native FP32 softmax rounded to BF16.
    """
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("Gekko ShortConv requires autocast to be disabled")
    operands = (x, normalized_weight, history_pool)
    if torch.is_grad_enabled() and any(t.requires_grad for t in operands):
        raise ValueError("Gekko ShortConv is inference-only")
    if not x.is_cuda or torch.version.hip is not None or x.dtype != torch.bfloat16:
        raise ValueError("Gekko ShortConv requires NVIDIA CUDA BF16 activations")
    if x.ndim != 2 or normalized_weight.ndim != 2 or history_pool.ndim != 3:
        raise ValueError("Expected x[B,D], normalized_weight[D,W], and pool[S,W-1,D]")
    batch, dimension = x.shape
    width = normalized_weight.shape[1]
    if (
        dimension <= 0
        or not 1 <= width <= 4
        or normalized_weight.shape[0] != dimension
        or history_pool.shape[0] <= 0
        or history_pool.shape[1:] != (width - 1, dimension)
        or not history_pool.is_contiguous()
    ):
        raise ValueError(
            "Expected positive channels/slots, width in [1,4], contiguous pool"
        )
    if any(t.device != x.device or t.dtype != x.dtype for t in operands[1:]):
        raise ValueError(
            "Input, history, and normalized weights must share BF16 device"
        )
    if (
        cache_slots.shape != (batch,)
        or cache_slots.device != x.device
        or cache_slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Cache slots must be a matching device int32/int64 vector")
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    if batch:
        _shortconv_decode_pooled[(batch, triton.cdiv(dimension, 256))](
            x,
            normalized_weight,
            history_pool,
            cache_slots,
            output,
            dimension,
            width,
            history_pool.shape[0],
            *x.stride(),
            *normalized_weight.stride(),
            *history_pool.stride()[:2],
            cache_slots.stride(0),
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
