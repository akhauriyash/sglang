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

"""Readable, single-request xLLM Gekko inference state machine.

This is a mathematical PyTorch reference, not an optimized attention backend.
Its independent-request semantics follow xLLM's stage4 GatedDeltaAttention and
the *full-chunk* adaptive_working_memory forward, rather than the unqualified
native cached inference path. Native CUDA/BF16 numerical parity is unqualified.
CUDA BF16 TDN at the checkpoint's 1536-channel, 32-group shape uses the native
scan arithmetic with request-local partial-block state. Other shapes use the
mathematical FP32 recurrence. Activation, convolution, and memory states use
the input dtype. Packed document resets are unsupported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def supports_pooled_gekko_decode(config: Mapping) -> bool:
    """Geometry supported by the native fixed-shape decode kernels."""
    heads, dim = config.get("num_heads"), config.get("head_dim")
    return (
        config.get("model_dim"),
        heads,
        config.get("num_kv_heads") or heads,
        dim,
        config.get("v_head_dim") or dim,
        config.get("chunk_size", 4096),
        config.get("timenorm_num_groups") or heads,
    ) == (1536, 32, 8, 64, 64, 4096, 32)


@dataclass
class GekkoState:
    """All state for one layer of one request; tensors never belong to the model.

    K/V streams have layout [retained_tokens, KV_heads, features]. At a chunk
    boundary they retain the last chunk, whose delayed memory update has not
    yet happened. Between boundaries they also retain the current prefix.
    """

    position: int
    tdn_count: Tensor
    tdn_mean: Tensor
    tdn_var: Tensor
    q_history: Tensor
    k_history: Tensor
    v_history: Tensor
    sca_k: Tensor
    values: Tensor
    feature_k: Tensor
    raw_k: Tensor
    memory: Tensor
    log_normalizer: Tensor
    tdn_packed: Tensor | None = None

    def clone(self) -> GekkoState:
        """An independent snapshot, including convolution and delayed memory."""
        return GekkoState(
            **{
                f.name: (
                    getattr(self, f.name).clone()
                    if isinstance(getattr(self, f.name), Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )


class TimestepDecayNorm(nn.Module):
    """Native offset affine and learned priors with persistent FP32 statistics.

    Decay/bias correction use the mathematical config constants. Native CUB
    rounds beta and ``1-beta`` in FP32 and computes powers with a parallel scan;
    its rounding and the reference's expm1 correction are not bit-equivalent.
    """

    def __init__(
        self,
        num_features: int,
        num_groups: int,
        beta1: float = 0.999,
        beta2: float = 0.9999,
        eps: float = 1e-5,
    ):
        super().__init__()
        if not 0 < num_groups < num_features or num_features % num_groups:
            raise ValueError("TDN groups must evenly divide the model width")
        if not (0 < beta1 < 1 and 0 < beta2 < 1 and eps > 0):
            raise ValueError("TDN needs betas in (0, 1) and positive epsilon")
        self.num_features = num_features
        self.num_groups = num_groups
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight = nn.Parameter(torch.zeros(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("prior_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("prior_mean", torch.zeros(num_groups, dtype=torch.float32))
        self.register_buffer("prior_var", torch.zeros(num_groups, dtype=torch.float32))

    def forward(
        self, x: Tensor, count: Tensor, mean: Tensor, variance: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Accumulation and count advance are independent of serving partitions.
        output = []
        n = int(count.item())
        for token in x:
            grouped = token.float().view(self.num_groups, -1)
            token_var, token_mean = torch.var_mean(grouped, dim=-1, correction=0)
            n += 1
            mean = self.beta1 * mean + (1.0 - self.beta1) * token_mean
            variance = self.beta2 * variance + (1.0 - self.beta2) * token_var
            correction1 = -math.expm1(n * math.log(self.beta1))
            correction2 = -math.expm1(n * math.log(self.beta2))
            normalized = (grouped - (mean / correction1)[:, None]) * torch.rsqrt(
                variance[:, None] / correction2 + self.eps
            )
            # The native offset addition is performed in the parameter dtype.
            normalized = normalized.flatten() * (self.weight + 1).float()
            output.append((normalized + self.bias.float()).to(x.dtype))
        y = torch.stack(output) if output else x.new_empty(x.shape)
        return y, count.new_tensor(n), mean, variance


class NormalizedShortConv(nn.Module):
    """Depthwise causal convolution with checkpoint *logits*, without residual."""

    def __init__(self, num_features: int, width: int = 4):
        super().__init__()
        if not 1 <= width <= 4:
            raise ValueError("Native Gekko ShortConv width must be in [1, 4]")
        self.width = width
        self.weight = nn.Parameter(torch.zeros(num_features, width))
        self._normalized_weight_cache = None
        # CUDA graphs reference external constants by address. Keep captured
        # versions alive even if a later eager call refreshes the active cache.
        self._captured_normalized_weights = {}

    def _normalized_weight(self) -> Tensor:
        """Detached inference taps; ordinary tensor mutations invalidate them.

        Inference-created parameters have no version counter and stay uncached.
        Weight updates require graph recapture: replay does not execute Python
        cache validation. Mutating ``weight.data`` bypasses PyTorch's version
        tracking and is unsupported, as it is for autograd's mutation checks.
        """
        weight = self.weight
        key = None
        if not weight.is_inference():
            key = (
                id(weight),
                weight._version,
                weight.data_ptr(),
                weight.device,
                weight.dtype,
                weight.shape,
                weight.stride(),
            )
        cached = self._normalized_weight_cache
        if key is None or cached is None or cached[0] != key:
            with torch.no_grad():
                normalized = weight.float().softmax(-1).to(weight.dtype)
            self._normalized_weight_cache = (
                (key, normalized) if key is not None else None
            )
        else:
            normalized = cached[1]
        if normalized.is_cuda:
            with torch.cuda.device(normalized.device):
                if torch.cuda.is_current_stream_capturing():
                    self._captured_normalized_weights[id(normalized)] = normalized
        return normalized

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._normalized_weight_cache = None
        return result

    def decode_pooled(
        self, x: Tensor, history_pool: Tensor, cache_slots: Tensor
    ) -> Tensor:
        """One token per request; update distinct valid device slots in place."""
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (x, self.weight, history_pool)
        ):
            raise ValueError("Gekko ShortConv pooled decode is inference-only")
        from .xllm_shortconv import native_shortconv_decode_pooled

        return native_shortconv_decode_pooled(
            x, self._normalized_weight(), history_pool, cache_slots
        )

    def forward(self, x: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        if history.shape != (self.width - 1, x.shape[-1]):
            raise ValueError("ShortConv history has an incompatible shape")
        if x.shape[0] == 0:
            return x, history
        if (
            x.is_cuda
            and torch.version.hip is None
            and x.dtype == self.weight.dtype == history.dtype == torch.bfloat16
            and x.device == self.weight.device == history.device
            and not torch.is_autocast_enabled("cuda")
            and not (
                torch.is_grad_enabled()
                and any(t.requires_grad for t in (x, self.weight, history))
            )
        ):
            from .xllm_shortconv import native_shortconv_forward

            return native_shortconv_forward(
                x, self.weight, history, normalized_weight=self._normalized_weight()
            )
        taps = self.weight.float().softmax(dim=-1).to(x.dtype)
        stream = torch.cat((history, x), dim=0)
        # FP32 accumulation, followed by one activation-dtype cast. Histories
        # contain the projected inputs, including any projection bias.
        y = sum(
            stream[i : i + len(x)].float() * taps[:, i].float()
            for i in range(self.width)
        ).to(x.dtype)
        next_history = stream[-(self.width - 1) :] if self.width > 1 else stream[:0]
        return y, next_history.clone()


def adaptive_working_memory_update(
    memory: Tensor,
    log_normalizer: Tensor,
    feature_keys: Tensor,
    raw_keys: Tensor,
    values: Tensor,
) -> tuple[Tensor, Tensor]:
    """Incorporate one completed older chunk using native normalization axes.

    ``memory`` is [KV_heads, key_dim, value_dim], normalizers are
    [KV_heads, key_dim], and chunk tensors are [tokens, KV_heads, features].
    Callers must delay this update until the following chunk finishes emitting.
    """
    if len(raw_keys) == 0:
        raise ValueError("AWM cannot incorporate an empty chunk")
    # Native AWM reduces contiguous time rows. A strided FP32 logsumexp changes
    # reduction rounding, which can change the BF16 memory update ratio.
    keys_by_feature = raw_keys.permute(1, 2, 0).float().contiguous()
    time_weights = keys_by_feature.softmax(dim=-1).to(memory.dtype)
    chunk_normalizer = torch.logsumexp(keys_by_feature, dim=-1)
    updated_normalizer = torch.logaddexp(log_normalizer, chunk_normalizer)
    ratio = (chunk_normalizer - updated_normalizer).exp().to(memory.dtype)
    residual = torch.baddbmm(
        values.transpose(0, 1), feature_keys.transpose(0, 1), memory, alpha=-1.0
    )
    update = torch.bmm(time_weights, residual)
    return (
        torch.addcmul(memory, update - memory, ratio.unsqueeze(-1)),
        updated_normalizer,
    )


class GekkoAttention(nn.Module):
    """Native-named projections and explicit per-request recurrent state.

    ``config`` is the saved native model dictionary or a resolved family config
    with ``model_dim``. A ``gekko_attention`` dictionary takes precedence over
    generic fields. ``forward`` takes [tokens, model_dim] and returns the
    attention delta and a fresh state object; the caller adds the residual.
    """

    def __init__(self, config: Mapping[str, Any] | Any):
        super().__init__()
        # These helpers are kept torch-only. The local import also permits the
        # reference backbone to instantiate this module without an import cycle.
        from .xllm_reference import (
            NativeRMSNorm,
            NativeRotaryEmbedding,
        )

        cfg = dict(config) if isinstance(config, Mapping) else vars(config).copy()
        cfg.update(cfg.get("gekko_attention") or {})
        for field in ("model_dim", "num_heads", "head_dim"):
            if cfg.get(field) is None:
                raise ValueError(f"Gekko requires explicit native {field}")
        self.model_dim = int(cfg["model_dim"])
        self.num_heads = int(cfg["num_heads"])
        self.num_kv_heads = int(cfg.get("num_kv_heads") or self.num_heads)
        self.head_dim = int(cfg["head_dim"])
        self.v_head_dim = int(cfg.get("v_head_dim") or self.head_dim)
        self.chunk_size = int(cfg.get("chunk_size", 4096))
        if (
            min(
                self.model_dim,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.v_head_dim,
                self.chunk_size,
            )
            <= 0
            or self.num_heads % self.num_kv_heads
        ):
            raise ValueError("Gekko dimensions and chunk size must be valid")
        supported = {
            "attn_act_func": "softmax",
            "attn_gate_func": "softplus",
            "awm_qknorm_func": "softmax",
            "causal_conv_weight_normalization": True,
            "residual_func": "base",
        }
        for key, expected in supported.items():
            if cfg.get(key, expected) != expected:
                raise ValueError(f"Gekko reference requires {key}={expected!r}")
        if any(
            cfg.get(key, 0)
            for key in ("dropout", "attention_dropout", "hidden_dropout")
        ):
            raise ValueError("Gekko inference reference requires zero dropout")
        self.value_activation = cfg.get("gekko_value_activation", "identity")
        if self.value_activation not in ("identity", "silu"):
            raise ValueError("Unsupported Gekko value activation")
        self.norm_eps = float(cfg.get("rmsnorm_eps", 1e-6))
        width = int(cfg.get("causal_conv_width", 4))
        qdim = self.num_heads * self.head_dim
        kdim = self.num_kv_heads * self.head_dim
        vdim = self.num_kv_heads * self.v_head_dim
        odim = self.num_heads * self.v_head_dim
        self.wq = nn.Linear(self.model_dim, qdim, bias=False)
        self.wk = nn.Linear(self.model_dim, kdim, bias=False)
        self.wv = nn.Linear(self.model_dim, vdim, bias=True)
        self.wr = nn.Linear(self.model_dim, odim, bias=cfg.get("apply_bias_term", True))
        self.wo = nn.Linear(odim, self.model_dim, bias=False)
        self.q_conv = NormalizedShortConv(qdim, width)
        self.k_conv = NormalizedShortConv(kdim, width)
        self.v_conv = NormalizedShortConv(vdim, width)
        self.query_norm = NativeRMSNorm(qdim, self.norm_eps, num_groups=self.num_heads)
        self.key_norm = NativeRMSNorm(kdim, self.norm_eps, num_groups=self.num_kv_heads)
        self.timenorm = TimestepDecayNorm(
            self.model_dim,
            int(cfg.get("timenorm_num_groups") or self.num_heads),
            float(cfg.get("timenorm_beta1", 0.999)),
            float(cfg.get("timenorm_beta2", 0.9999)),
            float(cfg.get("timenorm_eps", 1e-5)),
        )
        self.rotary = NativeRotaryEmbedding(
            self.head_dim,
            rope_head_dim=cfg.get("rope_head_dim"),
            rope_base=float(cfg.get("rope_base", 500000.0)),
            rope_scaling=cfg.get("rope_scaling"),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 131072)),
        )

    def uses_native_tdn(self, x: Tensor) -> bool:
        return (
            x.is_cuda
            and torch.version.hip is None
            and x.dtype == torch.bfloat16
            and self.model_dim == 1536
            and self.timenorm.num_groups == 32
        )

    def initial_state(self, x: Tensor) -> GekkoState:
        """Initialize request-local state from checkpoint buffers without aliasing."""
        nk, dk, dv = self.num_kv_heads, self.head_dim, self.v_head_dim
        width = self.q_conv.width - 1
        tdn_packed = None
        if self.uses_native_tdn(x):
            from .xllm_tdn import init_tdn_state

            tdn_packed = init_tdn_state(
                self.timenorm.prior_count.to(x.device),
                self.timenorm.prior_mean.to(x.device),
                self.timenorm.prior_var.to(x.device),
                self.timenorm.beta1,
                self.timenorm.beta2,
            ).storage
        return GekkoState(
            position=0,
            tdn_count=self.timenorm.prior_count.to(x.device).clone(),
            # Native TDN rounds initial checkpoint priors to the activation
            # dtype before promoting them into its persistent FP32 cache.
            tdn_mean=self.timenorm.prior_mean.to(x).float().clone(),
            tdn_var=self.timenorm.prior_var.to(x).float().clone(),
            q_history=x.new_zeros((width, self.num_heads * dk)),
            k_history=x.new_zeros((width, nk * dk)),
            v_history=x.new_zeros((width, nk * dv)),
            sca_k=x.new_empty((0, nk, dk)),
            values=x.new_empty((0, nk, dv)),
            feature_k=x.new_empty((0, nk, dk)),
            raw_k=x.new_empty((0, nk, dk)),
            memory=x.new_zeros((nk, dk, dv)),
            log_normalizer=torch.full(
                (nk, dk), -math.inf, device=x.device, dtype=torch.float32
            ),
            tdn_packed=tdn_packed,
        )

    def _attend(
        self, q: Tensor, k: Tensor, v: Tensor, positions: Tensor, key_start: int
    ) -> Tensor:
        repetitions = self.num_heads // self.num_kv_heads
        expanded_k = k.repeat_interleave(repetitions, dim=1)
        expanded_v = v.repeat_interleave(repetitions, dim=1)
        scores = torch.einsum("thd,shd->hts", q.float(), expanded_k.float())
        scores *= self.head_dim**-0.5
        key_positions = torch.arange(key_start, key_start + len(k), device=q.device)
        # Retained arrays start at the previous architectural chunk; the upper
        # causal bound handles a serving piece with several current tokens.
        scores.masked_fill_(key_positions[None, :] > positions[:, None], -math.inf)
        probabilities = scores.softmax(dim=-1)
        return torch.einsum("hts,shd->thd", probabilities, expanded_v.float()).to(
            v.dtype
        )

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        state: GekkoState | None = None,
        *,
        bos_mask: Tensor | None = None,
        segment_idx: Tensor | None = None,
    ) -> tuple[Tensor, GekkoState]:
        if bos_mask is not None or segment_idx is not None:
            raise ValueError(
                "Packed/reset Gekko state is unsupported; use independent requests"
            )
        if x.ndim != 2 or x.shape[-1] != self.model_dim:
            raise ValueError("Gekko expects one request with shape [tokens, model_dim]")
        state = self.initial_state(x) if state is None else state
        if state.memory.device != x.device or state.memory.dtype != x.dtype:
            raise ValueError("Gekko state device and activation dtype must match input")
        expected_retained = (
            state.position
            if state.position < self.chunk_size
            else self.chunk_size + state.position % self.chunk_size
        )
        if state.position < 0 or len(state.sca_k) != expected_retained:
            raise ValueError(
                "Gekko state retention does not match its logical position"
            )
        if not len(x):
            return x.new_empty(x.shape), state
        tdn_packed = state.tdn_packed
        if self.uses_native_tdn(x):
            from .xllm_tdn import PackedTDNState, native_tdn_forward

            if tdn_packed is None:
                raise ValueError("Native Gekko TDN requires its partial scan state")
            # The helper clones packed storage before advancing. A pooled view
            # or caller's previous request state must remain unchanged.
            z, mean, variance, next_tdn = native_tdn_forward(
                x,
                self.timenorm.weight,
                self.timenorm.bias,
                PackedTDNState.from_storage(tdn_packed, state.position),
                self.timenorm.beta1,
                self.timenorm.beta2,
                self.timenorm.eps,
            )
            count = state.tdn_count + len(x)
            tdn_packed = next_tdn.storage
        else:
            z, count, mean, variance = self.timenorm(
                x, state.tdn_count, state.tdn_mean, state.tdn_var
            )
        raw_q, q_history = self.q_conv(self.wq(z), state.q_history)
        raw_k, k_history = self.k_conv(self.wk(z), state.k_history)
        value, v_history = self.v_conv(self.wv(z), state.v_history)
        if self.value_activation == "silu":
            value = F.silu(value)
        query_features = (
            raw_q.view(-1, self.num_heads, self.head_dim)
            .float()
            .softmax(-1)
            .to(x.dtype)
        )
        raw_k_heads = raw_k.view(-1, self.num_kv_heads, self.head_dim)
        key_features = raw_k_heads.float().softmax(-1).to(x.dtype)
        q = self.query_norm(raw_q).view(-1, self.num_heads, self.head_dim)
        k = self.key_norm(raw_k).view(-1, self.num_kv_heads, self.head_dim)
        positions = torch.arange(
            state.position, state.position + len(x), device=x.device
        )
        q, k = self.rotary(q, k, positions)
        value = value.view(-1, self.num_kv_heads, self.v_head_dim)
        outputs = []
        offset = 0
        while offset < len(x):
            length = min(
                len(x) - offset, self.chunk_size - state.position % self.chunk_size
            )
            piece = slice(offset, offset + length)
            retained_start = state.position - len(state.sca_k)
            cached_k = torch.cat((state.sca_k, k[piece]), dim=0)
            cached_v = torch.cat((state.values, value[piece]), dim=0)
            cached_feature_k = torch.cat((state.feature_k, key_features[piece]), dim=0)
            cached_raw_k = torch.cat((state.raw_k, raw_k_heads[piece]), dim=0)
            sca = self._attend(
                q[piece], cached_k, cached_v, positions[piece], retained_start
            )
            memories = state.memory.repeat_interleave(
                self.num_heads // self.num_kv_heads, dim=0
            )
            awm = torch.einsum("thd,hdv->thv", query_features[piece], memories)
            outputs.append(sca + awm)
            next_position = state.position + length
            memory, log_normalizer = state.memory, state.log_normalizer
            if next_position % self.chunk_size == 0:
                # All outputs from the current chunk use the OLD memory. Only
                # now incorporate its predecessor, then retain the new chunk.
                if len(cached_k) == 2 * self.chunk_size:
                    old = slice(0, self.chunk_size)
                    memory, log_normalizer = adaptive_working_memory_update(
                        memory,
                        log_normalizer,
                        cached_feature_k[old],
                        cached_raw_k[old],
                        cached_v[old],
                    )
                cached_k = cached_k[-self.chunk_size :].clone()
                cached_v = cached_v[-self.chunk_size :].clone()
                cached_feature_k = cached_feature_k[-self.chunk_size :].clone()
                cached_raw_k = cached_raw_k[-self.chunk_size :].clone()
            state = replace(
                state,
                position=next_position,
                sca_k=cached_k,
                values=cached_v,
                feature_k=cached_feature_k,
                raw_k=cached_raw_k,
                memory=memory,
                log_normalizer=log_normalizer,
            )
            offset += length
        combined = torch.cat(outputs, dim=0)
        if (
            combined.is_cuda
            and torch.version.hip is None
            and combined.dtype == torch.bfloat16
            and not (torch.is_grad_enabled() and combined.requires_grad)
        ):
            from .xllm_rmsnorm import native_rms_norm

            # Native Gekko applies the same RMS reduction to each value head,
            # without an affine parameter. Its FP32 FMA affects BF16 rounding.
            normalized = native_rms_norm(
                combined.flatten(1), None, self.norm_eps, self.num_heads
            )
        else:
            normalized = combined.float() * torch.rsqrt(
                combined.float().square().mean(-1, keepdim=True) + self.norm_eps
            )
            normalized = normalized.to(x.dtype).flatten(1)
        gate = F.softplus(self.wr(z), beta=math.log(2.0))
        delta = self.wo(normalized * gate)
        state = replace(
            state,
            tdn_count=count,
            tdn_mean=mean,
            tdn_var=variance,
            tdn_packed=tdn_packed,
            q_history=q_history,
            k_history=k_history,
            v_history=v_history,
        )
        return delta, state
