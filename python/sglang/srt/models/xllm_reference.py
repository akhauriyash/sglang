"""Torch reference for native xLLM dense and hybrid checkpoints.

This module deliberately has no SGLang or native xLLM imports. It preserves
native parameter names and provides small request-owned states plus per-layer
entry points for serving adapters. It is a semantic oracle, not a replacement
for native learned-checkpoint BF16 qualification or a fast attention backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def native_model_config(config: Any) -> dict:
    """Extract the saved native model config from a serving config or mapping."""
    if not isinstance(config, Mapping):
        config = vars(config)
    config = dict(config)
    config = config.get("native_model_config", config)
    return dict(config)


def resolve_family(config: Mapping, kind: str) -> dict:
    """Apply the saved resolved attention-family override, without guessing it."""
    family = dict(config)
    override = config.get(f"{kind}_attention")
    if override is not None:
        family.update(dict(override))
    elif str(config.get("arch", "transformer")).startswith("hybrid_"):
        raise ValueError(f"Missing resolved {kind}_attention configuration")
    return family


def native_norm_eps(config: Mapping) -> float:
    """Current ModelConf spelling, with the legacy saved-config fallback."""
    return config.get("rmsnorm_eps", config.get("norm_eps", 1e-6))


class NativeRMSNorm(nn.Module):
    """Native zero-centered affine RMSNorm, including BF16 offset rounding."""

    def __init__(self, dim: int, eps: float = 1e-6, num_groups: int = 1):
        super().__init__()
        if dim <= 0 or num_groups <= 0 or dim % num_groups:
            raise ValueError("RMSNorm dimensions must divide into positive groups")
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: Tensor) -> Tensor:
        if (
            x.is_cuda
            and torch.version.hip is None
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
            and not (
                torch.is_grad_enabled()
                and (x.requires_grad or self.weight.requires_grad)
            )
        ):
            from .xllm_rmsnorm import native_rms_norm

            return native_rms_norm(x, self.weight, self.eps, self.num_groups)
        shape = x.shape
        grouped = x.float().reshape(*shape[:-1], self.num_groups, -1)
        inverse_rms = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + self.eps)
        normalized = (grouped * inverse_rms).reshape(shape)
        # Native Python adds one in parameter dtype *before* the CUDA norm
        # promotes the effective gain to its FP32 accumulation dtype.
        gain = (self.weight + 1.0).float()
        return (normalized * gain).to(x.dtype)


class NativeRotaryEmbedding(nn.Module):
    """Adjacent-pair RoPE with the native xLLM YaRN correction context."""

    def __init__(
        self,
        head_dim: int,
        rope_head_dim: int | None = None,
        rope_base: float = 500000.0,
        rope_scaling: Mapping | None = None,
        max_position_embeddings: int = 131072,
        *,
        cache_inverse_frequencies: bool = False,
    ):
        super().__init__()
        self.cache_inverse_frequencies = cache_inverse_frequencies
        # Keep this ordinary tensor attribute out of checkpoint buffers and
        # Module dtype conversions: inverse frequencies always remain FP32.
        self._inverse_frequency_cache = None
        # Captured graphs borrow these addresses after the eager stream cache
        # changes. Retain every captured tensor until the module is destroyed.
        self._captured_inverse_frequencies = {}
        self.head_dim = head_dim
        self.rope_head_dim = head_dim if rope_head_dim is None else rope_head_dim
        self.base = rope_base
        self.max_position_embeddings = max_position_embeddings
        self.scaling = None if rope_scaling is None else dict(rope_scaling)
        self.attention_factor = 1.0
        dim = self.rope_head_dim
        if dim < 0 or dim > head_dim or dim % 2:
            raise ValueError(
                "Rotary dimension must be even and no larger than head_dim"
            )
        if not math.isfinite(rope_base) or rope_base <= 1:
            raise ValueError("rope_base must be finite and greater than one")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.scaling is not None:
            allowed = {
                "rope_type",
                "factor",
                "beta_fast",
                "beta_slow",
                "attention_factor",
                "original_max_position_embeddings",
            }
            if set(self.scaling) - allowed:
                raise ValueError("Unsupported native RoPE scaling fields")
            if self.scaling.get("rope_type", "yarn") != "yarn":
                raise ValueError("Only native YaRN scaling is supported")
            for key, default in (
                ("factor", 1.0),
                ("beta_fast", 32.0),
                ("beta_slow", 1.0),
            ):
                value = self.scaling.get(key)
                value = default if value is None else float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid YaRN {key}")
                self.scaling[key] = value
            if (
                self.scaling["factor"] < 1
                or self.scaling["beta_fast"] < self.scaling["beta_slow"]
            ):
                raise ValueError("Invalid YaRN factor or correction bounds")
            amplitude = self.scaling.get("attention_factor")
            self.attention_factor = (
                1 + 0.1 * math.log(self.scaling["factor"])
                if amplitude is None
                else float(amplitude)
            )
            if not math.isfinite(self.attention_factor) or self.attention_factor <= 0:
                raise ValueError("Invalid YaRN attention_factor")
            original = self.scaling.get("original_max_position_embeddings")
            if original is not None and (type(original) is not int or original <= 0):
                raise ValueError("Invalid original_max_position_embeddings")

    def inverse_frequencies(self, device: torch.device) -> Tensor:
        if not self.cache_inverse_frequencies:
            return self._compute_inverse_frequencies(device)
        device = torch.device(device)
        stream = None
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            stream = torch.cuda.current_stream(device).cuda_stream
        key = (
            device,
            stream,
            self.head_dim,
            self.rope_head_dim,
            self.base,
            self.max_position_embeddings,
            self.attention_factor,
            None if self.scaling is None else tuple(sorted(self.scaling.items())),
        )
        cached = self._inverse_frequency_cache
        if cached is None or cached[0] != key:
            # Rebuild on a stream switch instead of assuming synchronization.
            # An inference-mode warmup must also permit later position gradients.
            with torch.inference_mode(False), torch.no_grad():
                frequencies = self._compute_inverse_frequencies(device)
            cached = (key, frequencies)
            self._inverse_frequency_cache = cached
        if device.type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    # Recapturing the same key can use a new tensor while older
                    # graphs still need the previous tensor at its original address.
                    self._captured_inverse_frequencies[id(cached[1])] = cached[1]
        return cached[1]

    def _compute_inverse_frequencies(self, device: torch.device) -> Tensor:
        dim = self.rope_head_dim
        if not dim:
            return torch.empty(0, dtype=torch.float32, device=device)
        if self.scaling is None:
            # This Python construction matches native non-YaRN frequency rounding.
            denominator = torch.tensor(
                [self.base ** (j / dim) for j in range(0, dim, 2)],
                dtype=torch.float32,
                device=device,
            )
            return 1.0 / denominator
        scaling = self.scaling
        context = scaling.get("original_max_position_embeddings")
        context = self.max_position_embeddings if context is None else context

        def correction_dim(rotations):
            return (
                dim
                * math.log(context / (rotations * 2 * math.pi))
                / (2 * math.log(self.base))
            )

        low = max(math.floor(correction_dim(scaling["beta_fast"])), 0)
        high = min(math.ceil(correction_dim(scaling["beta_slow"])), dim - 1)
        if low == high:
            high += 0.001
        pairs = torch.arange(dim // 2, dtype=torch.float32, device=device)
        ramp = ((pairs - low) / (high - low)).clamp(0.0, 1.0)
        channels = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        denominator = self.base ** (channels / dim)
        extrapolated = 1.0 / denominator
        interpolated = 1.0 / (scaling["factor"] * denominator)
        extrapolation_weight = 1.0 - ramp
        return (
            interpolated * (1.0 - extrapolation_weight)
            + extrapolated * extrapolation_weight
        )

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        if self.rope_head_dim == 0:
            return q, k
        if (
            self.head_dim == 64
            and q.ndim == k.ndim == 3
            and 0 < len(q) <= 4
            and q.dtype == k.dtype == torch.bfloat16
            and q.is_cuda
            and k.device == q.device
            and positions.device == q.device
            and positions.dtype in (torch.int32, torch.int64)
            and positions.is_contiguous()
            and not q.requires_grad
            and not k.requires_grad
            and torch.version.hip is None
            and torch.cuda.get_device_capability(q.device) == (9, 0)
        ):
            from sglang.srt.models.xllm_rope import native_rope_decode

            # This lookup also retains frequencies borrowed by captured graphs.
            return native_rope_decode(
                q, k, positions, self.inverse_frequencies(q.device), self.attention_factor
            )
        phases = torch.outer(positions.float(), self.inverse_frequencies(q.device))
        frequencies = torch.polar(
            torch.ones_like(phases) * self.attention_factor, phases
        )

        def rotate(x):
            dim = self.rope_head_dim
            pairs = torch.view_as_complex(
                x[..., :dim].float().reshape(*x.shape[:-1], -1, 2)
            )
            rotated = (
                torch.view_as_real(pairs * frequencies[:, None, :])
                .flatten(-2)
                .to(x.dtype)
            )
            return (
                rotated
                if dim == self.head_dim
                else torch.cat((rotated, x[..., dim:]), -1)
            )

        return rotate(q), rotate(k)


@dataclass
class LayerState:
    """All history for one layer of one request; never shared across requests."""

    key: Tensor | None = None
    value: Tensor | None = None
    key_start: int = 0
    conv: dict[str, Tensor] = field(default_factory=dict)
    extra: Any = None

    def clone(self) -> LayerState:
        return LayerState(
            key=None if self.key is None else self.key.clone(),
            value=None if self.value is None else self.value.clone(),
            key_start=self.key_start,
            conv={name: tensor.clone() for name, tensor in self.conv.items()},
            extra=None if self.extra is None else self.extra.clone(),
        )


@dataclass
class RequestState:
    position: int
    layers: list[LayerState]

    def clone(self) -> RequestState:
        return RequestState(self.position, [layer.clone() for layer in self.layers])

    def reset(self) -> None:
        self.position = 0
        self.layers = [LayerState() for _ in self.layers]


class ResidualShortConv(nn.Module):
    """Raw depthwise convolution plus input; the final tap is the current token."""

    def __init__(self, num_features: int, kernel_size: int = 4, init_std: float = 0.02):
        super().__init__()
        if num_features <= 0 or not 1 <= kernel_size <= 4:
            raise ValueError(
                "Residual ShortConv requires positive channels and width 1..4"
            )
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(num_features, kernel_size))
        nn.init.trunc_normal_(
            self.weight, std=init_std, a=-3 * init_std, b=3 * init_std
        )

    def forward(
        self, x: Tensor, state: LayerState | None = None, name: str = "conv"
    ) -> Tensor:
        if x.ndim != 2 or x.shape[-1] != self.weight.shape[0]:
            raise ValueError("Convolution input must be [tokens, channels]")
        width = self.kernel_size
        history = None if state is None else state.conv.get(name)
        if history is None:
            history = x.new_zeros((width - 1, x.shape[-1]))
        if history.shape != (width - 1, x.shape[-1]):
            raise ValueError("ShortConv state shape does not match its layer")
        joined = torch.cat((history, x), dim=0)
        # FLA accumulates taps in FP32 and casts before the residual addition.
        convolution = torch.zeros_like(x, dtype=torch.float32)
        for tap in range(width):
            convolution = (
                convolution
                + joined[tap : tap + len(x)].float() * self.weight[:, tap].float()
            )
        if state is not None:
            state.conv[name] = (
                joined[-(width - 1) :].clone() if width > 1 else joined[:0].clone()
            )
        return x + convolution.to(x.dtype)


def _validate_backbone(config: Mapping) -> None:
    required = ("model_dim", "ffn_hidden_dim", "num_heads", "head_dim")
    for name in required:
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"An explicit positive {name} is required")
    unsupported = {
        "qknorm": False,
        "apply_attn_gate": False,
        "scale_emb": False,
        "num_experts": 0,
        "num_values": 0,
        "dropout": 0.0,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
    }
    for name, expected in unsupported.items():
        if config.get(name, expected) != expected:
            raise ValueError(
                f"Unsupported native dense/Inkling setting {name}={config[name]}"
            )
    for name in ("apply_rmsnorm", "norm_affine", "swiglu"):
        if not config.get(name, True):
            raise ValueError(f"The native backbone requires {name}=True")
    if config.get("residual_func", "base") not in ("base", "add"):
        raise ValueError("Only additive residuals are supported")
    if config.get("attn_act_func", "softmax") != "softmax":
        raise ValueError("Only softmax full/local attention is supported")
    if config.get("v_head_dim") not in (None, config["head_dim"]):
        raise ValueError("Dense/Inkling query and value dimensions must match")


class ReferenceAttention(nn.Module):
    # Bound the temporary score matrix for long prompts. Each query still
    # attends to precisely the same keys with the same reduction order.
    max_score_elements = 4 * 1024 * 1024

    def __init__(self, config: Mapping, kind: str):
        super().__init__()
        _validate_backbone(config)
        self.kind = kind
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config.get("num_kv_heads") or self.num_heads
        self.head_dim = config["head_dim"]
        if self.num_heads % self.num_kv_heads:
            raise ValueError("Query head count must be divisible by KV head count")
        dim = config["model_dim"]
        self.norm = NativeRMSNorm(
            dim, native_norm_eps(config), config.get("layernorm_num_groups", 1)
        )
        self.wq = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.rope = NativeRotaryEmbedding(
            self.head_dim,
            config.get("rope_head_dim"),
            config.get("rope_base", 500000.0),
            config.get("rope_scaling"),
            config.get("max_position_embeddings") or 131072,
        )
        self.window = (
            config.get("local_attention_window", 8192) if kind == "inkling" else None
        )
        if self.window is not None and (
            type(self.window) is not int or self.window <= 0
        ):
            raise ValueError("Inkling window must be a positive integer")
        if kind == "inkling":
            if config.get("causal_conv_weight_normalization", False):
                raise ValueError("Inkling convolution weights must be unnormalized")
            width = config.get("causal_conv_width", 4)
            std = config.get("short_conv_init_std", 0.02)
            self.k_shortconv = ResidualShortConv(
                self.num_kv_heads * self.head_dim, width, std
            )
            self.v_shortconv = ResidualShortConv(
                self.num_kv_heads * self.head_dim, width, std
            )
            self.out_shortconv = ResidualShortConv(dim, width, std)

    def project_qkv(
        self, hidden: Tensor, state: LayerState | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized = self.norm(hidden)
        q, k, v = self.wq(normalized), self.wk(normalized), self.wv(normalized)
        if self.kind == "inkling":
            k = self.k_shortconv(k, state, "attention.k")
            v = self.v_shortconv(v, state, "attention.v")
        return (
            q.reshape(-1, self.num_heads, self.head_dim),
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            v.reshape(-1, self.num_kv_heads, self.head_dim),
        )

    def apply_rope(
        self, q: Tensor, k: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        return self.rope(q, k, positions)

    def finish_attention(
        self, attention: Tensor, residual: Tensor, state: LayerState | None = None
    ) -> Tensor:
        output = self.wo(attention.reshape(len(residual), -1))
        if self.kind == "inkling":
            output = self.out_shortconv(output, state, "attention.out")
        return residual + output

    def forward(
        self, hidden: Tensor, positions: Tensor, state: LayerState | None = None
    ) -> Tensor:
        q, k, v = self.project_qkv(hidden, state)
        q, k = self.apply_rope(q, k, positions)
        key_start = int(positions[0])
        if state is not None and state.key is not None:
            key_start = state.key_start
            k = torch.cat((state.key, k), 0)
            v = torch.cat((state.value, v), 0)
        keys = torch.arange(key_start, key_start + len(k), device=hidden.device)
        repeat = self.num_heads // self.num_kv_heads
        grouped_k = k.repeat_interleave(repeat, dim=1).float()
        grouped_v = v.repeat_interleave(repeat, dim=1).float()
        # Explicit FP32 attention is a mathematical reference. Native flash
        # attention reduction/rounding must be qualified separately on GPUs.
        query_piece = max(1, self.max_score_elements // (self.num_heads * len(k)))
        pieces = []
        for offset in range(0, len(q), query_piece):
            query_positions = positions[offset : offset + query_piece]
            allowed = keys[None, :] <= query_positions[:, None]
            if self.window is not None:
                allowed &= keys[None, :] > query_positions[:, None] - self.window
            scores = (
                torch.einsum(
                    "thd,shd->hts", q[offset : offset + query_piece].float(), grouped_k
                )
                * self.head_dim**-0.5
            )
            probabilities = scores.masked_fill(
                ~allowed.unsqueeze(0), -torch.inf
            ).softmax(-1)
            pieces.append(
                torch.einsum("hts,shd->thd", probabilities, grouped_v).to(hidden.dtype)
            )
        attended = torch.cat(pieces)
        if state is not None:
            keep = len(k) if self.window is None else min(len(k), self.window - 1)
            state.key = k[len(k) - keep :].clone()
            state.value = v[len(v) - keep :].clone()
            state.key_start = key_start + len(k) - keep
        return self.finish_attention(attended, hidden, state)


class ReferenceFeedForward(nn.Module):
    def __init__(
        self,
        config: Mapping,
        inkling: bool = False,
        *,
        down_projection_cls: type[nn.Linear] = nn.Linear,
    ):
        super().__init__()
        dim, intermediate = config["model_dim"], config["ffn_hidden_dim"]
        self.norm = NativeRMSNorm(
            dim, native_norm_eps(config), config.get("layernorm_num_groups", 1)
        )
        self.fc1 = nn.Linear(dim, intermediate, bias=False)
        self.fc3 = nn.Linear(dim, intermediate, bias=False)
        self.fc2 = down_projection_cls(intermediate, dim, bias=False)
        self.out_shortconv = (
            ResidualShortConv(
                dim,
                config.get("causal_conv_width", 4),
                config.get("short_conv_init_std", 0.02),
            )
            if inkling
            else None
        )

    def forward(self, hidden: Tensor, state: LayerState | None = None) -> Tensor:
        normalized = self.norm(hidden)
        output = self.fc2(F.silu(self.fc1(normalized)) * self.fc3(normalized))
        if self.out_shortconv is not None:
            output = self.out_shortconv(output, state, "nffn.out")
        return hidden + output


class ReferenceBlock(nn.Module):
    def __init__(self, config: Mapping, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "gekko":
            # The Gekko module imports shared helpers lazily, avoiding cycles.
            from .xllm_gekko_reference import GekkoAttention

            self.gda = GekkoAttention(config)
        else:
            self.attention = ReferenceAttention(config, kind)
        self.nffn = ReferenceFeedForward(config, inkling=kind == "inkling")

    def forward(
        self, hidden: Tensor, positions: Tensor, state: LayerState | None = None
    ) -> Tensor:
        if self.kind == "gekko":
            recurrent_position = (
                0 if state is None or state.extra is None else state.extra.position
            )
            if recurrent_position != int(positions[0]):
                raise ValueError(
                    "Gekko recurrent state does not match request position"
                )
            delta, recurrent = self.gda(hidden, None if state is None else state.extra)
            hidden = hidden + delta
            if state is not None:
                state.extra = recurrent
        else:
            hidden = self.attention(hidden, positions, state)
        return self.nffn(hidden, state)


class ReferenceOutput(nn.Module):
    def __init__(self, config: Mapping):
        super().__init__()
        dim = config["model_dim"]
        output_size = config.get("output_size", -1)
        output_size = config["vocab_size"] if output_size == -1 else output_size
        if output_size != config["vocab_size"]:
            raise ValueError("Only untied vocabulary output is supported")
        self.final_norm = NativeRMSNorm(
            dim, native_norm_eps(config), config.get("layernorm_num_groups", 1)
        )
        self.output = nn.Linear(dim, output_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.output(self.final_norm(hidden)).float()


class XllmReferenceModel(nn.Module):
    """Native-named TP1 model; caches and positions belong to each request."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = native_model_config(config)
        config = self.config
        layers = config["num_layers"]
        kinds = config.get("layer_kinds")
        if kinds is None and str(config.get("arch", "transformer")).startswith(
            "hybrid_"
        ):
            raise ValueError("Hybrid checkpoints must declare every layer kind")
        self.layer_kinds = tuple(kinds or ["full"] * layers)
        if len(self.layer_kinds) != layers or any(
            kind not in {"full", "inkling", "gekko"} for kind in self.layer_kinds
        ):
            raise ValueError("Invalid layer_kinds")
        if config.get("scale_emb", False):
            raise ValueError("Embedding scaling is unsupported")
        self.max_position_embeddings = config.get("max_position_embeddings") or 131072
        self.embed = nn.Embedding(config["vocab_size"], config["model_dim"])
        self.layers = nn.ModuleList(
            [
                ReferenceBlock(resolve_family(config, kind), kind)
                for kind in self.layer_kinds
            ]
        )
        self.output = ReferenceOutput(config)

    def new_cache(self) -> RequestState:
        return RequestState(0, [LayerState() for _ in self.layers])

    def forward_request(
        self,
        tokens: Tensor,
        cache: RequestState | None = None,
        selected_positions: Sequence[int] | None = None,
    ) -> Tensor:
        if (
            tokens.ndim != 1
            or len(tokens) == 0
            or tokens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Request tokens must be a nonempty vector of integer IDs")
        start = 0 if cache is None else cache.position
        if start + len(tokens) > self.max_position_embeddings:
            raise ValueError("Request exceeds saved max_position_embeddings")
        if cache is not None and len(cache.layers) != len(self.layers):
            raise ValueError("Request state does not match model layer count")
        selected = None
        if selected_positions is not None:
            selected = torch.as_tensor(
                selected_positions, device=tokens.device, dtype=torch.long
            )
            if selected.ndim != 1 or ((selected < 0) | (selected >= len(tokens))).any():
                raise ValueError(
                    "Selected logits positions must belong to this token piece"
                )
        positions = torch.arange(start, start + len(tokens), device=tokens.device)
        hidden = self.embed(tokens)
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden, positions, None if cache is None else cache.layers[index]
            )
        if cache is not None:
            cache.position += len(tokens)
        if selected is not None:
            hidden = hidden.index_select(0, selected)
        return self.output(hidden)

    def forward(
        self,
        tokens: Tensor,
        cache: RequestState | Sequence[RequestState] | None = None,
        selected_positions: Sequence[int] | None = None,
    ) -> Tensor:
        if tokens.ndim == 1:
            if cache is not None and not isinstance(cache, RequestState):
                raise ValueError("One token vector requires one request state")
            return self.forward_request(tokens, cache, selected_positions)
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError("Tokens must have shape [tokens] or [batch, tokens]")
        if cache is None:
            caches = [None] * len(tokens)
        elif isinstance(cache, RequestState):
            if len(tokens) != 1:
                raise ValueError("Each batch row needs an independent request state")
            caches = [cache]
        else:
            caches = list(cache)
            if len(caches) != len(tokens) or len({id(c) for c in caches}) != len(
                caches
            ):
                raise ValueError("Each batch row needs an independent request state")
        return torch.stack(
            [
                self.forward_request(row, state, selected_positions)
                for row, state in zip(tokens, caches)
            ]
        )


@torch.no_grad()
def native_cache_free_logits(
    model: nn.Module, tokens: Tensor, selected_positions: Sequence[int] | None = None
) -> Tensor:
    """Read-only dense/Inkling logit oracle using an already-loaded native model.

    Native HybridInkling.forward requires loss targets. Calling the original
    embed, rotary, blocks, final norm and projection directly exposes logits
    without editing training code or substituting any operator. The caller
    owns native setup/checkpoint restoration and must qualify this helper
    against the original full-forward loss before using numerical fixtures.
    """
    if model.training or getattr(model, "context_parallel_size", 1) != 1:
        raise ValueError("Native oracle requires an evaluation model with CP=1")
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError("Native tokens must have shape [batch, positive sequence]")
    if getattr(model, "arch", None) not in ("transformer", "hybrid_inkling"):
        raise ValueError("This native helper supports dense and Inkling only")
    hidden = model.embed(tokens)
    length = tokens.shape[1]
    if model.arch == "hybrid_inkling":
        full_freqs = (
            None
            if model.full_rope is None
            else model.full_rope.get_freqs_cis(0, length, hidden.device)
        )
        local_freqs = (
            None
            if model.inkling_rope is None
            else model.inkling_rope.get_freqs_cis(0, length, hidden.device)
        )
        for kind, layer in zip(model.layer_kinds, model.layers):
            freqs = full_freqs if kind == "full" else local_freqs
            args = (hidden, freqs, None, None, True, None)
            hidden, _, _ = layer(*args) if kind == "full" else layer(*args, None)
    else:
        rope = getattr(model, "rotary_emb", None)
        if rope is None:
            rope = getattr(model, "rope", None)
        freqs = None if rope is None else rope.get_freqs_cis(0, length, hidden.device)
        for layer in model.layers:
            hidden, _, _ = layer(hidden, freqs, None, None, True, None)
    if selected_positions is not None:
        hidden = hidden[:, list(selected_positions)]
    return model.output.output(model.output.final_norm(hidden)).float()
