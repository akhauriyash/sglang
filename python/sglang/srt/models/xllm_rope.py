# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Fused SM90 BF16 adjacent-pair RoPE with native xLLM arithmetic.

Inverse frequencies come from NativeRotaryEmbedding, including its YaRN
construction and graph lifetime handling. CUDA libdevice trigonometry and
explicit complex-multiply FMAs preserve the native Torch rounding sequence.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rotate_decode(
    Q,
    K,
    Positions,
    Frequencies,
    OQ,
    OK,
    QH: tl.constexpr,
    KH: tl.constexpr,
    DIM: tl.constexpr,
    ROTARY: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    AMPLITUDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    pair = tl.arange(0, BLOCK)
    live = pair < DIM // 2
    rotate = pair < ROTARY // 2
    if head < QH:
        pointer = Q + row * QS0 + head * QS1 + (2 * pair) * QS2
        real = tl.load(pointer, live, 0).to(tl.float32)
        imaginary = tl.load(pointer + QS2, live, 0).to(tl.float32)
    else:
        pointer = K + row * KS0 + (head - QH) * KS1 + (2 * pair) * KS2
        real = tl.load(pointer, live, 0).to(tl.float32)
        imaginary = tl.load(pointer + KS2, live, 0).to(tl.float32)
    frequency = tl.load(Frequencies + pair, rotate, 0)
    position = tl.load(Positions + row).to(tl.float32)
    phase = position * frequency
    cosine = libdevice.cos(phase) * AMPLITUDE
    sine = libdevice.sin(phase) * AMPLITUDE
    # Match the CUDA c10::complex multiplication: each second product rounds
    # before the first product is fused into the final addition/subtraction.
    result_real = tl.fma(real, cosine, -(imaginary * sine))
    result_imaginary = tl.fma(real, sine, imaginary * cosine)
    result_real = tl.where(rotate, result_real, real)
    result_imaginary = tl.where(rotate, result_imaginary, imaginary)
    if head < QH:
        destination = OQ + (row * QH + head) * DIM + 2 * pair
    else:
        destination = OK + (row * KH + head - QH) * DIM + 2 * pair
    tl.store(destination, result_real, live)
    tl.store(destination + 1, result_imaginary, live)


def native_rope_decode(q, k, positions, inverse_frequencies, attention_factor=1.0):
    """Rotate BF16 Q/K[B,H,64] using device positions and fixed FP32 frequencies.

    The serving caller limits this path to at most four rows. Inputs and their
    strides stay fixed across graph replay; positions and values may change.
    A zero-dimensional rotary prefix returns the original Q/K tensors.
    """
    if (
        q.ndim != 3
        or k.ndim != 3
        or len(q) == 0
        or len(q) != len(k)
        or q.shape[-1] != 64
        or k.shape[-1] != 64
        or not q.shape[1]
        or not k.shape[1]
    ):
        raise ValueError("Native RoPE requires Q/K[B,H,64] with equal nonempty batch")
    if (
        any(
            value.dtype != torch.bfloat16
            or not value.is_cuda
            or value.device != q.device
            for value in (q, k)
        )
        or torch.version.hip is not None
    ):
        raise ValueError("Native RoPE requires BF16 Q/K on one NVIDIA CUDA device")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("Native RoPE decode is currently qualified only for SM90")
    if (
        positions.shape != (len(q),)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != q.device
        or not positions.is_contiguous()
        or inverse_frequencies.ndim != 1
        or len(inverse_frequencies) > 32
        or inverse_frequencies.dtype != torch.float32
        or inverse_frequencies.device != q.device
        or not inverse_frequencies.is_contiguous()
    ):
        raise ValueError(
            "Native RoPE requires device int32/int64 positions[B] and FP32 inverse frequencies"
        )
    if any(value.requires_grad for value in (q, k, inverse_frequencies)):
        raise ValueError("Native RoPE decode supports inference without autograd only")
    if not math.isfinite(attention_factor) or attention_factor <= 0:
        raise ValueError("Native RoPE attention factor must be finite and positive")
    if not len(inverse_frequencies):
        return q, k
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _rotate_decode[(len(q), q.shape[1] + k.shape[1])](
        q,
        k,
        positions,
        inverse_frequencies,
        oq,
        ok,
        q.shape[1],
        k.shape[1],
        64,
        len(inverse_frequencies) * 2,
        *q.stride(),
        *k.stride(),
        float(attention_factor),
        32,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return oq, ok
