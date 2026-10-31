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

"""xLLM grouped timestep-decay normalization for NVIDIA BF16 inference.

This specialization implements native CUDA Welford and CUB affine scan
arithmetic for 32 groups of 48 channels. Inputs and learned parameters must
be finite. Other layouts, dtypes, devices, and training are outside its
contract; the reference retains an independent Torch fallback. Pooled decode
additionally supports negative request slots as masked padding.

Requests retain current 256-token block moments plus block-start FP32 EMA
values and beta powers. Replaying a partial block preserves full-prefix
rounding across arbitrary chunks. Blocks start at request position zero,
independently of the saved prior count. No native extension is required.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Two partial blocks, two block-start EMA vectors, and two beta powers.
TDN_STATE_SIZE = 2 * 32 * 256 + 2 * 32 + 2


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], tl.float32, True, 1
    )


@triton.jit
def _sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], tl.float32, True, 1
    )


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], tl.float32, True, 1
    )


@triton.jit
def _row_moments(X, Mean, Var, L: tl.constexpr):
    token, group = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    base = token * 1536 + group * 48
    first = tl.load(X + base + lane).to(tl.float32)
    second = tl.load(X + base + lane + 32, lane < 16, 0).to(tl.float32)
    delta = _sub(second, first)
    updated_mean = tl.fma(0.5, delta, first)
    updated_var = tl.fma(0.5, tl.fma(delta, _sub(second, updated_mean), 0.0), 0.0)
    count = tl.where(lane < 16, 2, 1)
    mean = tl.where(lane < 16, updated_mean, first)
    var = tl.where(lane < 16, updated_var, 0.0)
    # The native Welford population-moment operator, descending shfl.down.
    for step in tl.static_range(5):
        distance = 16 >> step
        peer = tl.minimum(lane + distance, 31)
        other_count = tl.gather(count, peer, 0)
        other_mean = tl.gather(mean, peer, 0)
        other_var = tl.gather(var, peer, 0)
        combined_count = count + other_count
        c2 = tl.div_rn(other_count.to(tl.float32), combined_count.to(tl.float32))
        c1 = _sub(1.0, c2)
        difference = _sub(other_mean, mean)
        mean = tl.fma(c2, other_mean, _mul(c1, mean))
        var = tl.fma(
            _mul(c1, difference),
            _mul(c2, difference),
            tl.fma(c2, other_var, _mul(c1, var)),
        )
        count = combined_count
    tl.store(Mean + group * L + token, tl.sum(tl.where(lane == 0, mean, 0.0), 0))
    tl.store(Var + group * L + token, tl.sum(tl.where(lane == 0, var, 0.0), 0))


@triton.jit
def _cub_warp_scans(addend, coefficient):
    thread = tl.arange(0, 256)
    lane, warp = thread % 32, thread // 32
    # CUB WarpScanShfl performs ascending shfl.up at distances 1,2,4,8,16.
    for step in tl.static_range(5):
        distance = 1 << step
        peer = warp * 32 + tl.maximum(lane - distance, 0)
        prefix_add = tl.gather(addend, peer, 0)
        prefix_mul = tl.gather(coefficient, peer, 0)
        added = tl.fma(coefficient, prefix_add, addend)
        multiplied = _mul(coefficient, prefix_mul)
        addend = tl.where(lane >= distance, added, addend)
        coefficient = tl.where(lane >= distance, multiplied, coefficient)
    aggregate_add = tl.gather(addend, tl.full((256,), 31, tl.int32), 0)
    aggregate_mul = tl.gather(coefficient, tl.full((256,), 31, tl.int32), 0)
    prefix_add = tl.zeros((256,), tl.float32)
    prefix_mul = tl.full((256,), 1.0, tl.float32)
    # CUB ComputeWarpPrefix folds warp aggregates sequentially, not by a tree.
    for index in tl.static_range(1, 8):
        prefix_add = tl.where(warp == index, aggregate_add, prefix_add)
        prefix_mul = tl.where(warp == index, aggregate_mul, prefix_mul)
        location = tl.full((256,), index * 32 + 31, tl.int32)
        next_add = tl.gather(addend, location, 0)
        next_mul = tl.gather(coefficient, location, 0)
        aggregate_add = tl.fma(next_mul, aggregate_add, next_add)
        aggregate_mul = _mul(next_mul, aggregate_mul)
    return (
        tl.where(warp > 0, tl.fma(coefficient, prefix_add, addend), addend),
        tl.where(warp > 0, _mul(coefficient, prefix_mul), coefficient),
    )


@triton.jit
def _initial_stream_powers(Count, Powers, B1: tl.constexpr, B2: tl.constexpr):
    count = tl.load(Count).to(tl.float32)
    tl.store(Powers, libdevice.pow(tl.full((), B1, tl.float32), count))
    tl.store(Powers + 1, libdevice.pow(tl.full((), B2, tl.float32), count))


@triton.jit(do_not_specialize=["length"])
def _scan_stream_block(
    Mean,
    Var,
    StartMean,
    StartVar,
    StartPowers,
    CumMean,
    CumRstd,
    FinalMean,
    FinalVar,
    FinalPowers,
    length,
    B1: tl.constexpr,
    B2: tl.constexpr,
    EPS: tl.constexpr,
):
    group = tl.program_id(0)
    thread = tl.arange(0, 256)
    valid = thread < length
    b1, b2 = tl.full((), B1, tl.float32), tl.full((), B2, tl.float32)
    mean = tl.load(Mean + group * 256 + thread, valid, 0)
    var = tl.load(Var + group * 256 + thread, valid, 0)
    mean_add, mean_mul = _cub_warp_scans(
        _mul(_sub(1.0, b1), mean), tl.where(valid, b1, 1.0)
    )
    var_add, var_mul = _cub_warp_scans(
        _mul(_sub(1.0, b2), var), tl.where(valid, b2, 1.0)
    )
    total_mean = tl.fma(mean_mul, tl.load(StartMean + group), mean_add)
    total_var = tl.fma(var_mul, tl.load(StartVar + group), var_add)
    total_b1 = tl.fma(mean_mul, tl.load(StartPowers), 0.0)
    total_b2 = tl.fma(var_mul, tl.load(StartPowers + 1), 0.0)
    den1, den2 = _sub(1.0, total_b1), _sub(1.0, total_b2)
    corrected_mean = tl.where((den1 > 0) & (den2 > 0), tl.div_rn(total_mean, den1), 0.0)
    corrected_rstd = tl.rsqrt(_add(tl.div_rn(total_var, den2), EPS))
    corrected_rstd = tl.where((den1 > 0) & (den2 > 0), corrected_rstd, tl.rsqrt(EPS))
    tl.store(CumMean + group * 256 + thread, corrected_mean, valid)
    tl.store(CumRstd + group * 256 + thread, corrected_rstd, valid)
    tl.store(FinalMean + group, tl.sum(tl.where(thread == 255, total_mean, 0.0), 0))
    tl.store(FinalVar + group, tl.sum(tl.where(thread == 255, total_var, 0.0), 0))
    if group == 0:
        tl.store(FinalPowers, tl.sum(tl.where(thread == 255, total_b1, 0.0), 0))
        tl.store(FinalPowers + 1, tl.sum(tl.where(thread == 255, total_b2, 0.0), 0))


@triton.jit(do_not_specialize=["cursor", "prefix"])
def _apply_stream_chunk(X, Gain, Bias, CumMean, CumRstd, Y, cursor, prefix):
    token, group = tl.program_id(0), tl.program_id(1)
    feature = tl.arange(0, 64)
    channel = group * 48 + feature
    offset = (cursor + token) * 1536 + channel
    value = tl.load(X + offset, feature < 48, 0).to(tl.float32)
    gain = tl.load(Gain + channel, feature < 48, 0).to(tl.float32)
    bias = tl.load(Bias + channel, feature < 48, 0).to(tl.float32)
    mean = tl.load(CumMean + group * 256 + prefix + token)
    rstd = tl.load(CumRstd + group * 256 + prefix + token)
    output = tl.fma(_mul(_sub(value, mean), rstd), gain, bias)
    tl.store(Y + offset, output, feature < 48)


@triton.jit
def _decode_pooled(
    X,
    Weight,
    Bias,
    Pool,
    Slots,
    Positions,
    Y,
    POOL_ROWS: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    PACKED_OFFSET: tl.constexpr,
    MEAN_OFFSET: tl.constexpr,
    VAR_OFFSET: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    EPS: tl.constexpr,
):
    row, group = tl.program_id(0), tl.program_id(1)
    slot, position = tl.load(Slots + row), tl.load(Positions + row)
    prefix = (position % 256).to(tl.int32)
    valid = (slot >= 0) & (slot < POOL_ROWS) & (position >= 0) & (prefix != 255)
    pool = Pool + slot.to(tl.int64) * POOL_STRIDE
    packed = pool + PACKED_OFFSET

    # Identical native Welford population moments to _row_moments. Keeping
    # the 32-lane reduction separate from the 256-lane scan preserves its
    # arithmetic even though both now execute in one program.
    lane = tl.arange(0, 32)
    base = row * 1536 + group * 48
    first = tl.load(X + base + lane, valid, 0).to(tl.float32)
    second = tl.load(X + base + lane + 32, valid & (lane < 16), 0).to(tl.float32)
    delta = _sub(second, first)
    updated_mean = tl.fma(0.5, delta, first)
    updated_var = tl.fma(0.5, tl.fma(delta, _sub(second, updated_mean), 0.0), 0.0)
    count = tl.where(lane < 16, 2, 1)
    mean = tl.where(lane < 16, updated_mean, first)
    var = tl.where(lane < 16, updated_var, 0.0)
    for step in tl.static_range(5):
        distance = 16 >> step
        peer = tl.minimum(lane + distance, 31)
        other_count = tl.gather(count, peer, 0)
        other_mean = tl.gather(mean, peer, 0)
        other_var = tl.gather(var, peer, 0)
        combined_count = count + other_count
        c2 = tl.div_rn(other_count.to(tl.float32), combined_count.to(tl.float32))
        c1 = _sub(1.0, c2)
        difference = _sub(other_mean, mean)
        mean = tl.fma(c2, other_mean, _mul(c1, mean))
        var = tl.fma(
            _mul(c1, difference),
            _mul(c2, difference),
            tl.fma(c2, other_var, _mul(c1, var)),
        )
        count = combined_count
    row_mean = tl.sum(tl.where(lane == 0, mean, 0.0), 0)
    row_var = tl.sum(tl.where(lane == 0, var, 0.0), 0)

    thread = tl.arange(0, 256)
    live = valid & (thread <= prefix)
    mean = tl.load(packed + 66 + group * 256 + thread, valid & (thread < prefix), 0)
    var = tl.load(packed + 8258 + group * 256 + thread, valid & (thread < prefix), 0)
    mean = tl.where(thread == prefix, row_mean, mean)
    var = tl.where(thread == prefix, row_var, var)
    tl.store(packed + 66 + group * 256 + prefix, row_mean, valid)
    tl.store(packed + 8258 + group * 256 + prefix, row_var, valid)
    b1, b2 = tl.full((), B1, tl.float32), tl.full((), B2, tl.float32)
    mean_add, mean_mul = _cub_warp_scans(
        _mul(_sub(1.0, b1), mean), tl.where(live, b1, 1.0)
    )
    var_add, var_mul = _cub_warp_scans(
        _mul(_sub(1.0, b2), var), tl.where(live, b2, 1.0)
    )
    total_mean = tl.fma(mean_mul, tl.load(packed + group, valid, 0), mean_add)
    total_var = tl.fma(var_mul, tl.load(packed + 32 + group, valid, 0), var_add)
    total_b1 = tl.fma(mean_mul, tl.load(packed + 64, valid, 0), 0.0)
    total_b2 = tl.fma(var_mul, tl.load(packed + 65, valid, 0), 0.0)
    den1, den2 = _sub(1.0, total_b1), _sub(1.0, total_b2)
    corrected_mean = tl.where((den1 > 0) & (den2 > 0), tl.div_rn(total_mean, den1), 0.0)
    corrected_rstd = tl.rsqrt(_add(tl.div_rn(total_var, den2), EPS))
    corrected_rstd = tl.where((den1 > 0) & (den2 > 0), corrected_rstd, tl.rsqrt(EPS))
    tl.store(
        pool + MEAN_OFFSET + group,
        tl.sum(tl.where(thread == 255, total_mean, 0.0), 0),
        valid,
    )
    tl.store(
        pool + VAR_OFFSET + group,
        tl.sum(tl.where(thread == 255, total_var, 0.0), 0),
        valid,
    )
    mean = tl.sum(tl.where(thread == prefix, corrected_mean, 0.0), 0)
    rstd = tl.sum(tl.where(thread == prefix, corrected_rstd, 0.0), 0)

    feature = tl.arange(0, 64)
    channel = group * 48 + feature
    value = tl.load(X + row * 1536 + channel, valid & (feature < 48), 0).to(tl.float32)
    weight = tl.load(Weight + channel, feature < 48, 0).to(tl.float32)
    # The native PyTorch weight + 1 expression rounds to BF16 before the
    # apply kernel promotes it to FP32. Do not fuse away that cast.
    gain = _add(weight, 1.0).to(tl.bfloat16).to(tl.float32)
    bias = tl.load(Bias + channel, feature < 48, 0).to(tl.float32)
    output = tl.fma(_mul(_sub(value, mean), rstd), gain, bias)
    output = tl.where(valid, output, tl.where(slot < 0, 0.0, float("nan")))
    tl.store(Y + row * 1536 + channel, output, feature < 48)


@dataclass(frozen=True)
class PackedTDNState:
    """Request-owned FP32 scan state and number of processed request tokens.

    Storage may be a contiguous state-pool view. Construction does not copy it.
    Forward clones it before writing, preserving snapshots and other requests.
    The caller keeps the running count separately from scan-block alignment.
    """

    storage: torch.Tensor
    position: int = 0

    def __post_init__(self):
        if (
            self.storage.dtype != torch.float32
            or self.storage.shape != (TDN_STATE_SIZE,)
            or not self.storage.is_contiguous()
            or not self.storage.is_cuda
            or torch.version.hip is not None
        ):
            raise ValueError("TDN storage must be contiguous NVIDIA CUDA FP32[16450]")
        if not isinstance(self.position, int) or self.position < 0:
            raise ValueError("TDN request position must be a nonnegative integer")

    @classmethod
    def from_storage(cls, storage: torch.Tensor, position: int):
        """Restore a pool view without copying; validate layout and position."""
        return cls(storage=storage, position=position)

    def clone(self):
        return type(self)(self.storage.clone(), self.position)

    @property
    def block_mean(self):
        return self.storage[:32]

    @property
    def block_var(self):
        return self.storage[32:64]

    @property
    def block_powers(self):
        return self.storage[64:66]

    @property
    def partial_mean(self):
        return self.storage[66:8258].view(32, 256)

    @property
    def partial_var(self):
        return self.storage[8258:].view(32, 256)


def _validate_parameters(beta1: float, beta2: float, eps: float):
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("TDN beta1 and beta2 must lie in [0, 1)")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("TDN epsilon must be finite and positive")


@torch.no_grad()
def init_tdn_state(
    prior_count: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_var: torch.Tensor,
    beta1: float = 0.999,
    beta2: float = 0.9999,
) -> PackedTDNState:
    """Initialize a request from finite FP32 priors and a nonnegative count.

    Native inference rounds FP32 mean and variance priors to BF16 before
    promoting them to accumulator precision. Count initializes beta powers;
    the returned request position is always zero.
    """
    _validate_parameters(beta1, beta2, 1e-5)
    if (
        not prior_mean.is_cuda
        or torch.version.hip is not None
        or prior_mean.numel() != 32
        or prior_var.numel() != 32
        or prior_mean.dtype != torch.float32
        or prior_var.dtype != torch.float32
        or prior_var.device != prior_mean.device
        or prior_count.device != prior_mean.device
        or prior_count.dtype != torch.int64
        or prior_count.numel() != 1
    ):
        raise ValueError("TDN requires NVIDIA CUDA FP32 priors[32] and int64 count[1]")
    state = PackedTDNState(
        # Unused partial-block entries are masked by the kernel. Initialize
        # them as well so snapshots and pooled state have deterministic bytes.
        torch.zeros(TDN_STATE_SIZE, device=prior_mean.device, dtype=torch.float32)
    )
    state.block_mean.copy_(prior_mean.to(torch.bfloat16).float().flatten())
    state.block_var.copy_(prior_var.to(torch.bfloat16).float().flatten())
    _initial_stream_powers[(1,)](
        prior_count.reshape(1), state.block_powers, beta1, beta2, num_warps=1
    )
    return state


@torch.no_grad()
def native_tdn_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    state: PackedTDNState,
    beta1: float = 0.999,
    beta2: float = 0.9999,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PackedTDNState]:
    """Return output, final FP32 mean/variance, and an independent next state.

    Input is nonempty BF16[tokens, 1536]. The BF16 weight stores an offset
    from one. Parameters must stay fixed over a request. The caller increments
    its own int64 count by the input length. No input state is modified.
    """
    _validate_parameters(beta1, beta2, eps)
    if (
        x.ndim != 2
        or x.shape[1] != 1536
        or len(x) == 0
        or not x.is_cuda
        or torch.version.hip is not None
        or x.dtype != torch.bfloat16
    ):
        raise ValueError("TDN requires nonempty NVIDIA CUDA BF16 x[tokens, 1536]")
    if state.storage.device != x.device:
        raise ValueError("TDN input and state must be on the same CUDA device")
    for name, value in (("weight", weight), ("bias", bias)):
        if (
            value.shape != (1536,)
            or value.dtype != torch.bfloat16
            or value.device != x.device
        ):
            raise ValueError(f"TDN {name} must be BF16[1536] on the input device")
    x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
    next_state = PackedTDNState(state.storage.clone(), state.position + len(x))
    position = state.position
    length = len(x)
    group_mean = torch.empty((32, length), dtype=torch.float32, device=x.device)
    group_var = torch.empty_like(group_mean)
    _row_moments[(length, 32)](
        x, group_mean, group_var, length, num_warps=1, enable_fp_fusion=False
    )
    scratch_mean = torch.empty_like(next_state.partial_mean)
    scratch_rstd = torch.empty_like(next_state.partial_mean)
    final_mean = torch.empty_like(next_state.block_mean)
    final_var = torch.empty_like(next_state.block_var)
    final_powers = torch.empty_like(next_state.block_powers)
    output = torch.empty_like(x)
    gain = weight + 1.0
    cursor = 0
    while cursor < length:
        prefix = position % 256
        taken = min(256 - prefix, length - cursor)
        end = prefix + taken
        next_state.partial_mean[:, prefix:end].copy_(
            group_mean[:, cursor : cursor + taken]
        )
        next_state.partial_var[:, prefix:end].copy_(
            group_var[:, cursor : cursor + taken]
        )
        _scan_stream_block[(32,)](
            next_state.partial_mean,
            next_state.partial_var,
            next_state.block_mean,
            next_state.block_var,
            next_state.block_powers,
            scratch_mean,
            scratch_rstd,
            final_mean,
            final_var,
            final_powers,
            end,
            beta1,
            beta2,
            eps,
            num_warps=8,
            enable_fp_fusion=False,
        )
        _apply_stream_chunk[(taken, 32)](
            x,
            gain,
            bias,
            scratch_mean,
            scratch_rstd,
            output,
            cursor,
            prefix,
            num_warps=2,
            enable_fp_fusion=False,
        )
        if end == 256:
            next_state.block_mean.copy_(final_mean)
            next_state.block_var.copy_(final_var)
            next_state.block_powers.copy_(final_powers)
        position += taken
        cursor += taken
    return output, final_mean, final_var, next_state


@torch.no_grad()
def native_tdn_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    temporal_pool: torch.Tensor,
    cache_slots: torch.Tensor,
    positions: torch.Tensor,
    *,
    packed_offset: int,
    mean_offset: int,
    var_offset: int,
    beta1: float = 0.999,
    beta2: float = 0.9999,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Normalize one token per pooled request without host state inspection.

    ``positions`` gives the number of tokens already processed, independently
    of the saved prior count. Pool rows must contain initialized packed state
    compatible with :func:`native_tdn_forward`; offsets count FP32 elements
    within one contiguous pool row. Real rows must own distinct slots.

    Negative slots are padding: their output is zero and no pool data is read
    or written. Other invalid device metadata, including a 256-token block
    boundary (``position % 256 == 255``), produces NaNs without changing state.
    The caller must route block boundaries through the eager operator, which
    commits block-start statistics and powers before the next pooled decode.
    This restriction avoids a cross-program race on those shared powers.

    Tensor addresses and shapes remain fixed for CUDA graph replay; slot and
    position tensor contents may change. The operator preserves the native
    Welford/256-lane scan order and BF16 gain rounding. It updates just the new
    partial moments and the externally stored final FP32 mean and variance.
    """
    _validate_parameters(beta1, beta2, eps)
    if (
        x.ndim != 2
        or x.shape[1] != 1536
        or len(x) == 0
        or not x.is_cuda
        or torch.version.hip is not None
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise ValueError(
            "TDN decode requires contiguous NVIDIA CUDA BF16 x[batch,1536]"
        )
    for name, value in (("weight", weight), ("bias", bias)):
        if (
            value.shape != (1536,)
            or value.dtype != torch.bfloat16
            or value.device != x.device
            or not value.is_contiguous()
        ):
            raise ValueError(f"TDN decode {name} must be contiguous BF16[1536]")
    if (
        temporal_pool.ndim < 2
        or temporal_pool.shape[0] == 0
        or temporal_pool.dtype != torch.float32
        or temporal_pool.device != x.device
        or not temporal_pool.is_contiguous()
    ):
        raise ValueError("TDN decode requires a contiguous FP32 pool[slots,...]")
    stride = temporal_pool.numel() // temporal_pool.shape[0]
    fields = ((packed_offset, TDN_STATE_SIZE), (mean_offset, 32), (var_offset, 32))
    if any(
        not isinstance(offset, int) or offset < 0 or offset + size > stride
        for offset, size in fields
    ):
        raise ValueError("TDN decode state offsets must fit within one pool row")
    if any(
        max(first, second) < min(first + first_size, second + second_size)
        for index, (first, first_size) in enumerate(fields)
        for second, second_size in fields[index + 1 :]
    ):
        raise ValueError("TDN decode state fields must not overlap")
    for name, value in (("cache_slots", cache_slots), ("positions", positions)):
        if (
            value.shape != (len(x),)
            or value.dtype not in (torch.int32, torch.int64)
            or value.device != x.device
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"TDN decode {name} must be contiguous device integers[batch]"
            )
    output = torch.empty_like(x)
    _decode_pooled[(len(x), 32)](
        x,
        weight,
        bias,
        temporal_pool,
        cache_slots,
        positions,
        output,
        temporal_pool.shape[0],
        stride,
        packed_offset,
        mean_offset,
        var_offset,
        beta1,
        beta2,
        eps,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return output
