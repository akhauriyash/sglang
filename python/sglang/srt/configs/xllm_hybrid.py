# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Native-format dense and hybrid xLLM checkpoint configuration.

The exported family dictionaries are authoritative. In particular, head width
and the two rotary configurations cannot be inferred from the model width.
"""

from dataclasses import dataclass

import torch
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import BaseLinearStateParams, Mamba2StateDType
from transformers import PretrainedConfig


@dataclass(kw_only=True, frozen=True)
class XllmGekkoStateShape:
    conv: list[tuple[int, int]]
    temporal: tuple[int, int, int]
    disable_conv_window_dedup: bool = True


@dataclass(kw_only=True, frozen=True)
class XllmGekkoCacheParams(BaseLinearStateParams):
    shape: XllmGekkoStateShape


class XllmHybridConfig(PretrainedConfig):
    model_type = "xllm_hybrid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # HF creates an empty config when computing serialization differences.
        if not kwargs:
            return
        if getattr(self, "xllm_export_schema_version", None) != 1:
            raise ValueError("xLLM requires export schema version 1")
        if not getattr(self, "native_norm_offset", False) or not getattr(
            self, "rotary_interleaved", False
        ):
            raise ValueError("xLLM requires native offset norms and interleaved RoPE")
        if len(self.layer_kinds) != self.num_hidden_layers:
            raise ValueError("layer_kinds must specify every xLLM layer")
        if any(kind not in {"full", "inkling", "gekko"} for kind in self.layer_kinds):
            raise ValueError("Unknown xLLM attention family")
        if {"inkling", "gekko"}.issubset(self.layer_kinds):
            raise ValueError("Combining Inkling and Gekko in one model is unsupported")
        for kind in set(self.layer_kinds):
            family = getattr(self, f"{kind}_attention")
            for native, serving in (
                ("num_heads", "num_attention_heads"),
                ("num_kv_heads", "num_key_value_heads"),
                ("head_dim", "head_dim"),
            ):
                if family[native] != getattr(self, serving):
                    raise ValueError(
                        "xLLM serving currently requires uniform head shapes"
                    )
        self.is_hybrid_swa = "inkling" in self.layer_kinds
        self.hybrid_layer_pattern = [int(k == "inkling") for k in self.layer_kinds]
        if self.is_hybrid_swa:
            self.sliding_window = self.inkling_attention["local_attention_window"]
            if type(self.sliding_window) is not int or self.sliding_window < 2:
                raise ValueError(
                    "xLLM serving requires an Inkling window of at least 2"
                )

    @property
    def full_attention_layer_ids(self):
        """Layers dispatched to softmax attention, including local softmax."""
        return [i for i, kind in enumerate(self.layer_kinds) if kind != "gekko"]

    @property
    def linear_layer_ids(self):
        return [i for i, kind in enumerate(self.layer_kinds) if kind != "full"]

    @property
    def mamba_chunk_size(self):
        return 64

    @property
    def mamba2_cache_params(self):
        if "gekko" in self.layer_kinds:
            family = self.gekko_attention
            width = family["causal_conv_width"] - 1
            nk, dk, dv = (
                family["num_kv_heads"],
                family["head_dim"],
                family["v_head_dim"],
            )
            qdim = family["num_heads"] * dk
            kdim, vdim = nk * dk, nk * dv
            retained = 2 * family["chunk_size"]
            groups = family["timenorm_num_groups"]
            # Native TDN replays the partial 256-token scan block. Keep both
            # moment arrays, its starting EMA statistics, and starting powers.
            # This matches xllm_tdn.TDN_STATE_SIZE without importing GPU kernels
            # while reading a model configuration.
            tdn_scan = (
                2 * 32 * 256 + 2 * 32 + 2
                if self.hidden_size == 1536 and groups == 32
                else 0
            )
            return XllmGekkoCacheParams(
                layers=self.linear_layer_ids,
                dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
                shape=XllmGekkoStateShape(
                    conv=[
                        (width, qdim),
                        (width, kdim),
                        (width, vdim),
                        (retained, kdim),
                        (retained, vdim),
                        (retained, kdim),
                        (retained, kdim),
                    ],
                    temporal=(1, 1, nk * dk * dv + nk * dk + 2 * groups + tdn_scan),
                ),
            )
        # Reuse the existing pooled short-conv backend and its stream ordering.
        # Only converted layers allocate state. The first two streams are unused
        # by this model; retaining the common layout avoids a second backend.
        from sglang.srt.configs.inkling import (
            InklingConvCacheParams,
            InklingConvStateShape,
            InklingStateDType,
        )

        width = self.inkling_attention["causal_conv_width"] - 1
        kv_dim = self.num_key_value_heads * self.head_dim
        return InklingConvCacheParams(
            layers=self.linear_layer_ids,
            shape=InklingConvStateShape(
                conv=[(width, kv_dim)] * 4 + [(width, self.hidden_size)] * 2,
                temporal=(0, 0, 0),
            ),
            dtype=InklingStateDType(),
        )


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=XllmHybridConfig,
        config_predicate=lambda config: "inkling" in config.layer_kinds,
        arch_names=["XllmInklingForCausalLM"],
        backend_class_name=(
            "sglang.srt.layers.attention.linear.inkling_sconv_backend."
            "InklingShortConvAttnBackend"
        ),
        hybrid_backend_class_name=(
            "sglang.srt.layers.attention.linear.xllm_inkling_backend."
            "XllmInklingHybridAttnBackend"
        ),
        uses_mamba_radix_cache=True,
        support_mamba_cache=False,
    )
)

register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=XllmHybridConfig,
        config_predicate=lambda config: "gekko" in config.layer_kinds,
        arch_names=["XllmGekkoForCausalLM"],
        backend_class_name=(
            "sglang.srt.layers.attention.linear.xllm_gekko_backend.XllmGekkoAttnBackend"
        ),
        hybrid_backend_class_name=(
            "sglang.srt.layers.attention.linear.xllm_gekko_backend."
            "XllmGekkoHybridAttnBackend"
        ),
        uses_mamba_radix_cache=True,
        support_mamba_cache=False,
    )
)
