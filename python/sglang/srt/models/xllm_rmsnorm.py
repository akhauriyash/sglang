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

"""Native xLLM BF16 RMSNorm arithmetic for CUDA inference.

The native reduction groups channels by vectorized CUDA register loads before
reducing warp partials. Its mean and epsilon are combined by an FP32 FMA. Both
details affect BF16 rounding; a conventional RMS reduction is not equivalent.
This helper imports only Torch and Triton so reference qualification can load
it independently of the serving runtime.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rms_norm_kernel(
    X,
    Gain,
    Y,
    DIM: tl.constexpr,
    CHANNELS: tl.constexpr,
    THREADS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    VECTOR: tl.constexpr,
    EPS: tl.constexpr,
    AFFINE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    thread = tl.arange(0, THREADS)
    lanes = thread % 32
    total = tl.full((THREADS,), 0.0, tl.float32)
    for index in tl.static_range(ELEMENTS):
        channel = (
            (index // VECTOR) * THREADS * VECTOR + thread * VECTOR + index % VECTOR
        )
        value = tl.load(
            X + row * DIM + group * CHANNELS + channel, channel < CHANNELS, 0
        ).to(tl.float32)
        total = tl.fma(value, value, total)

    # Native WarpReduce uses descending shfl.down, then saves each warp's lane0.
    for offset in tl.static_range(5):
        distance = 16 >> offset
        other_index = (thread // 32) * 32 + ((lanes + distance) % 32)
        other = tl.gather(total, other_index, axis=0)
        total = _add_rn(total, tl.where(lanes + distance < 32, other, total))
    first_lanes = tl.minimum(lanes * 32, THREADS - 32)
    total = tl.where(lanes < THREADS // 32, tl.gather(total, first_lanes, axis=0), 0.0)
    # Native BlockAllReduce broadcasts the warp partials with descending xor.
    for offset in tl.static_range(5):
        distance = 16 >> offset
        total = _add_rn(total, tl.gather(total, thread ^ distance, axis=0))
    scale = tl.full((), 1.0 / CHANNELS, tl.float32)
    inverse = tl.rsqrt(tl.fma(total, scale, tl.full((), EPS, tl.float32)))
    for index in tl.static_range(ELEMENTS):
        channel = (
            (index // VECTOR) * THREADS * VECTOR + thread * VECTOR + index % VECTOR
        )
        address = row * DIM + group * CHANNELS + channel
        value = tl.load(X + address, channel < CHANNELS, 0).to(tl.float32)
        output = _mul_rn(value, inverse)
        if AFFINE:
            gain = tl.load(Gain + group * CHANNELS + channel, channel < CHANNELS, 0).to(
                tl.float32
            )
            # Native offset affine rounds weight+1 to BF16 before applying it.
            # Keep that rounding inside the norm instead of launching an add.
            gain = _add_rn(gain, 1.0).to(tl.bfloat16).to(tl.float32)
            output = _mul_rn(output, gain)
        tl.store(Y + address, output, channel < CHANNELS)


def native_rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float, num_groups: int = 1
) -> torch.Tensor:
    """Native BF16 inference; optional ``weight`` stores the offset from one."""
    if (
        not x.is_cuda
        or torch.version.hip is not None
        or x.dtype != torch.bfloat16
        or (weight is not None and weight.dtype != torch.bfloat16)
    ):
        raise ValueError("Native CUDA RMSNorm requires BF16 input and weight")
    dimension = x.shape[-1]
    if (
        num_groups < 1
        or dimension < 1
        or (weight is not None and weight.shape != (dimension,))
        or dimension % num_groups
    ):
        raise ValueError("RMSNorm weight and groups must match the input channels")
    channels = dimension // num_groups
    padded = max(128, triton.next_power_of_2(channels))
    if padded > 8192:
        raise ValueError("Native RMSNorm supports at most 8192 channels per group")
    x = x.contiguous()
    output = torch.empty_like(x)
    if x.numel() == 0:
        return output
    threads = max(32, padded // 8)
    elements = padded // threads
    vector = next(
        (
            width
            for width in (8, 4, 2)
            if channels % width == 0
            and elements % width == 0
            and x.data_ptr() % (2 * width) == 0
        ),
        1,
    )
    # The kernel preserves the native BF16 rounding of the offset addition.
    gain = weight
    _rms_norm_kernel[(x.numel() // dimension, num_groups)](
        x,
        gain,
        output,
        dimension,
        channels,
        threads,
        elements,
        vector,
        eps,
        weight is not None,
        num_warps=threads // 32,
        enable_fp_fusion=False,
    )
    return output
