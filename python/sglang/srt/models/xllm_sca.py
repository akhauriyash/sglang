# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Native xLLM SCA softmax arithmetic on the vendored SM90 FA4 kernel.

Native SCA sums each thread's probabilities sequentially, fuses the first
addition with the previous tile's rescaling, reduces the four-lane group in
descending XOR order, and uses a correctly rounded reciprocal. These choices
affect BF16 outputs. The remaining attention algorithm is shared with FA4;
the custom class is passed explicitly and has its own compiled-cache identity.

These helpers support finite BF16 inputs with 64-channel Q/K/V heads on SM90.
Prefill implements one request with bottom-right causal alignment. Decode reads
one live prefix per request directly from a fixed pool using device slots and
lengths. Neither supports dropout, split KV, or autograd, or changes full attention.
"""

import math

import cutlass
import torch
from cutlass import Float32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from sglang.kernels.ops.attention.flash_attn.cute.interface import _flash_attn_fwd
from sglang.kernels.ops.attention.flash_attn.cute.softmax import Softmax


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip) for value in (a, b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _first_add_fma(previous, scale, probability, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(value).ir_value(loc=loc, ip=ip)
                for value in (previous, scale, probability)
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _reciprocal_rn(value, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(item).ir_value(loc=loc, ip=ip) for item in (1.0, value)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class NativeXllmSoftmax(Softmax):
    """SM90 softmax hooks preserving the tested native SCA arithmetic."""

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return NativeXllmSoftmax(
            scale_log2, num_rows, row_max, row_sum, arch, softmax_scale
        )

    @cute.jit
    def _update_row_sum(self, x: cute.TensorSSA, prior_sum=None, scale=None):
        assert self.arch < 100
        values = cute.make_rmem_tensor(x.shape, Float32)
        values.store(x)
        if cutlass.const_expr(prior_sum is None):
            total = values[0]
        else:
            total = _first_add_fma(prior_sum, scale, values[0])
        for index in cutlass.range_constexpr(1, cute.size(x.shape)):
            total = _add_rn(total, values[index])
        return total

    @cute.jit
    def _finalize_row_sum(self, x: cute.TensorSSA):
        values = cute.make_rmem_tensor(x.shape, Float32)
        values.store(x)
        for index in cutlass.range_constexpr(cute.size(x.shape)):
            total = values[index]
            total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, offset=2))
            total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, offset=1))
            values[index] = total
        return values.load()

    def _reciprocal(self, value: Float32):
        return _reciprocal_rn(value)


def native_sca_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = 0.125,
) -> torch.Tensor:
    """Attend one query suffix to its causal prefix; return BF16[tokens,H,64].

    Query row ``i`` may attend key rows through ``Klen - Qlen + i``. Callers
    select the architectural SCA window; this helper adds no sliding mask.
    Token/head strides are supported and input tensors are never modified.
    """
    if any(value.ndim != 3 or value.shape[-1] != 64 for value in (q, k, v)):
        raise ValueError("Native SCA requires Q/K/V[tokens, heads, 64]")
    if not 0 < len(q) <= len(k) or len(k) != len(v):
        raise ValueError("Native SCA requires 0 < query length <= equal K/V length")
    if (
        not q.shape[1]
        or not k.shape[1]
        or k.shape[1] != v.shape[1]
        or q.shape[1] % k.shape[1]
    ):
        raise ValueError("Native SCA requires compatible grouped-query head counts")
    if (
        any(
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or value.device != q.device
            for value in (q, k, v)
        )
        or torch.version.hip is not None
    ):
        raise ValueError("Native SCA requires BF16 Q/K/V on one NVIDIA CUDA device")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("Native SCA is currently qualified only for SM90")
    if any(value.requires_grad for value in (q, k, v)):
        raise ValueError("Native SCA supports inference without autograd only")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Native SCA softmax scale must be finite and positive")
    output, _ = _flash_attn_fwd(
        q=q[None],
        k=k[None],
        v=v[None],
        softmax_scale=softmax_scale,
        causal=True,
        num_splits=1,
        return_lse=False,
        softmax_impl=NativeXllmSoftmax,
    )
    return output[0]


def native_sca_decode(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    softmax_scale: float = 0.125,
) -> torch.Tensor:
    """Attend one token per request without gathering or copying its history.

    Q is BF16[B,H,64]; K/V pools are BF16[slots,capacity,KV,64]. Contiguous
    CUDA int32 ``slots[B]`` and ``lengths[B]`` select each chronological live
    prefix, including the incoming token. Their addresses remain fixed during
    CUDA graph replay while their values may change. Capacity must be at least
    256 and divisible by 128 (the serving pool uses 8192).

    Callers ensure 0 <= length <= capacity and valid slots for nonempty rows.
    Length-zero rows return zeros and may use slot -1. Values beyond a live
    prefix, including all rows of an empty slot, are never read. Device values
    are deliberately not synchronized to the host for validation.
    """
    if (
        q.ndim != 3
        or q.shape[0] == 0
        or q.shape[-1] != 64
        or k_pool.ndim != 4
        or k_pool.shape != v_pool.shape
        or k_pool.shape[0] == 0
        or k_pool.shape[-1] != 64
        or not k_pool.shape[2]
        or not q.shape[1]
        or q.shape[1] % k_pool.shape[2]
    ):
        raise ValueError(
            "Native SCA decode requires Q[B,H,64], equal K/V[slots,capacity,KV,64], "
            "and compatible grouped-query heads"
        )
    if (
        any(
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or value.device != q.device
            or value.stride(-1) != 1
            for value in (q, k_pool, v_pool)
        )
        or torch.version.hip is not None
    ):
        raise ValueError(
            "Native SCA decode requires last-dimension-contiguous BF16 Q/K/V "
            "on one NVIDIA CUDA device"
        )
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("Native SCA decode is currently qualified only for SM90")
    if any(value.requires_grad for value in (q, k_pool, v_pool)):
        raise ValueError("Native SCA decode supports inference without autograd only")
    if slots.shape != (len(q),) or lengths.shape != (len(q),):
        raise ValueError("Native SCA decode requires slots[B] and lengths[B]")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Native SCA softmax scale must be finite and positive")
    output, _ = _flash_attn_fwd(
        q=q[:, None],
        k=k_pool,
        v=v_pool,
        page_table=slots[:, None],
        seqused_k=lengths,
        max_seqlen_k=k_pool.shape[1],
        softmax_scale=softmax_scale,
        causal=True,
        num_splits=1,
        return_lse=False,
        softmax_impl=NativeXllmSoftmax,
    )
    return output[:, 0]
