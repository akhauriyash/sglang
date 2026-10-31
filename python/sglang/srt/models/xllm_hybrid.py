# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""TP1 serving adapter for native xLLM dense, Inkling and Gekko checkpoints.

Paged full/local attention and request-owned convolution state are provided by
SGLang. Native channel order, offset norms and rotary tables are preserved.
Gekko shares its pooled state between ordinary graph decode and the independent
functional recurrence used for prefill and architectural boundary updates.
"""

from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.inkling_common.sconv import SconvType, ShortConvolution
from sglang.srt.models.xllm_gekko_reference import (
    GekkoAttention,
    supports_pooled_gekko_decode,
)
from sglang.srt.models.xllm_gemm import GekkoDownProjection
from sglang.srt.models.xllm_reference import (
    ReferenceAttention,
    ReferenceFeedForward,
    ReferenceOutput,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_model,
    get_parallel,
    get_spec,
)
from torch import nn


class XllmResidualShortConv(ShortConvolution):
    """Round convolution to activation dtype before the native residual add."""

    def __init__(self, channels, width, stream, layer_id):
        super().__init__(channels, width, stream, use_residual=False, layer_id=layer_id)

    def forward(self, hidden_states, forward_batch, name=None):
        return hidden_states + super().forward(hidden_states, None, forward_batch)


class XllmServingAttention(ReferenceAttention):
    def __init__(self, config, kind, layer_id):
        super().__init__(config, kind)
        self.rope.cache_inverse_frequencies = True
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
            layer_id,
            sliding_window_size=self.window - 1 if self.window is not None else -1,
        )
        if kind == "inkling":
            width = config["causal_conv_width"]
            kv_dim = self.num_kv_heads * self.head_dim
            self.k_shortconv = XllmResidualShortConv(
                kv_dim, width, SconvType.K_LOCAL, layer_id
            )
            self.v_shortconv = XllmResidualShortConv(
                kv_dim, width, SconvType.V_LOCAL, layer_id
            )
            self.out_shortconv = XllmResidualShortConv(
                config["model_dim"], width, SconvType.ATTN, layer_id
            )

    def forward(self, hidden_states, positions, forward_batch):
        q, k, v = self.project_qkv(hidden_states, forward_batch)
        q, k = self.apply_rope(q, k, positions)
        attended = self.attn(q, k, v, forward_batch)
        return self.finish_attention(attended, hidden_states, forward_batch)


class XllmServingBlock(nn.Module):
    def __init__(self, config, kind, layer_id, *, down_projection_cls=nn.Linear):
        super().__init__()
        self.attention = XllmServingAttention(config, kind, layer_id)
        self.nffn = ReferenceFeedForward(
            config,
            inkling=kind == "inkling",
            down_projection_cls=down_projection_cls,
        )
        if kind == "inkling":
            self.nffn.out_shortconv = XllmResidualShortConv(
                config["model_dim"],
                config["causal_conv_width"],
                SconvType.MLP,
                layer_id,
            )

    def forward(self, hidden_states, positions, forward_batch):
        hidden_states = self.attention(hidden_states, positions, forward_batch)
        return self.nffn(hidden_states, forward_batch)


class XllmServingGekkoAttention(GekkoAttention):
    """Use fused native-arithmetic SCA for each architectural chunk piece."""

    def _attend(self, q, k, v, positions, key_start):
        from sglang.srt.models.xllm_sca import native_sca_attention

        # GekkoAttention retains the previous chunk plus the current prefix.
        # Queries are the suffix of that key stream, so bottom-right causal
        # alignment implements the architectural mask for prefill and decode.
        return native_sca_attention(q, k, v, self.head_dim**-0.5)


class XllmServingGekkoBlock(nn.Module):
    def __init__(self, config, layer_id, *, down_projection_cls=nn.Linear):
        super().__init__()
        self.layer_id = layer_id
        self.gda = XllmServingGekkoAttention(config)
        self.gda.rotary.cache_inverse_frequencies = True
        self.nffn = ReferenceFeedForward(
            config, down_projection_cls=down_projection_cls
        )

    def forward(self, hidden_states, positions, forward_batch):
        delta = get_attn_backend().forward_gekko(
            self.layer_id, self.gda, hidden_states, forward_batch
        )
        return self.nffn(hidden_states + delta)


class XllmDenseForCausalLM(nn.Module):
    allowed_layer_kinds = frozenset({"full"})
    down_projection_cls = nn.Linear

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if set(config.layer_kinds) - self.allowed_layer_kinds:
            raise ValueError("Export architecture does not match its layer kinds")
        parallel = get_parallel()
        if parallel.tp_size != 1 or parallel.pp_size != 1:
            raise ValueError("Native xLLM serving currently requires TP1 and PP1")
        if quant_config is not None:
            raise ValueError("Native xLLM quantization is not qualified")
        # ModelLoader constructs modules under the requested model dtype.
        # The reused convolution pool explicitly stores BF16 histories.
        if torch.get_default_dtype() != torch.bfloat16:
            raise ValueError("Native xLLM serving currently requires BF16")
        if get_model().kv_cache_dtype not in {"auto", "bfloat16"}:
            raise ValueError("Native xLLM serving currently requires a BF16 KV cache")
        graph = get_exec().graph.cuda_graph_config
        if (
            "gekko" in config.layer_kinds
            and graph[Phase.DECODE].backend != Backend.DISABLED
            and not supports_pooled_gekko_decode(config.gekko_attention)
        ):
            raise ValueError(
                "Gekko decode graphs require model_dim1536, 32Q/8KV heads of "
                "width64, chunk_size4096 and 32 TDN groups; use eager execution "
                "for other Gekko layouts"
            )
        if any(graph[phase].backend != Backend.DISABLED for phase in Phase.ALL) and (
            graph[Phase.PREFILL].backend != Backend.DISABLED
            or graph[Phase.DECODE].backend != Backend.FULL
        ):
            raise ValueError(
                "Native xLLM permits only full decode CUDA graphs "
                "with disabled prefill graphs"
            )
        if not get_memory().disable_radix_cache:
            raise ValueError("Native xLLM currently requires --disable-radix-cache")
        if get_spec().speculative_algorithm is not None:
            raise ValueError("Native xLLM speculative decoding is not qualified")
        if get_lora().enable_lora:
            raise ValueError("Native xLLM LoRA is not supported")
        if get_disagg().disaggregation_mode != "null":
            raise ValueError("Native xLLM disaggregated serving is not qualified")
        self.config = config
        native = config.native_model_config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            (
                XllmServingGekkoBlock(
                    config.gekko_attention,
                    index,
                    down_projection_cls=self.down_projection_cls,
                )
                if kind == "gekko"
                else XllmServingBlock(
                    getattr(config, f"{kind}_attention"),
                    kind,
                    index,
                    down_projection_cls=self.down_projection_cls,
                )
            )
            for index, kind in enumerate(config.layer_kinds)
        )
        self.output = ReferenceOutput(native)
        self.logits_processor = LogitsProcessor(config)
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or mode.is_extend() or mode.is_idle()):
            raise ValueError(f"Unsupported xLLM forward mode: {mode}")
        if mode.is_target_verify():
            raise ValueError("Native xLLM speculative verification is unsupported")
        hidden_states = self.embed(input_ids) if input_embeds is None else input_embeds
        if input_ids.numel():
            for layer in self.layers:
                hidden_states = layer(hidden_states, positions, forward_batch)
        hidden_states = self.output.final_norm(hidden_states)
        return self.logits_processor(
            input_ids, hidden_states, self.output.output, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        destinations = dict(self.named_parameters())
        destinations.update(dict(self.named_buffers()))
        loaded = set()
        for name, value in weights:
            if name in loaded:
                raise ValueError(f"Duplicate native xLLM tensor: {name}")
            if name not in destinations:
                raise ValueError(f"Unexpected native xLLM tensor: {name}")
            target = destinations[name]
            if name.endswith("shortconv.weight"):
                # Format-only adapter for the upstream convolution kernel.
                value = value.unsqueeze(1)
            if target.shape != value.shape:
                raise ValueError(
                    f"Native xLLM tensor {name}: expected {tuple(target.shape)}, "
                    f"got {tuple(value.shape)}"
                )
            with torch.no_grad():
                target.copy_(value)
            loaded.add(name)
        missing = destinations.keys() - loaded
        if missing:
            raise ValueError(f"Missing native xLLM tensors: {sorted(missing)}")
        return loaded

    def get_attention_sliding_window_size(self):
        if "inkling" in self.config.layer_kinds:
            return self.config.inkling_attention["local_attention_window"] - 1
        return None

    def get_num_kv_cache_layers(self):
        return len(self.config.full_attention_layer_ids)


class XllmInklingForCausalLM(XllmDenseForCausalLM):
    allowed_layer_kinds = frozenset({"full", "inkling"})


class XllmGekkoForCausalLM(XllmDenseForCausalLM):
    allowed_layer_kinds = frozenset({"full", "gekko"})
    down_projection_cls = GekkoDownProjection


EntryClass = [XllmDenseForCausalLM, XllmInklingForCausalLM, XllmGekkoForCausalLM]
