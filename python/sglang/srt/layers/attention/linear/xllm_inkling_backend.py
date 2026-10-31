# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Short-conv sidecar dispatch for xLLM's full/local softmax hybrid."""

from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
    InklingShortConvHybridAttnBackend,
)


class XllmInklingHybridAttnBackend(InklingShortConvHybridAttnBackend):
    def __init__(self, full_attn_backend, linear_attn_backend, full_attn_layers):
        # With hybrid SWA, ModelConfig.full_attention_layer_ids means global
        # attention only. Both global and local layers use the softmax backend;
        # the recurrent backend solely owns their convolution state.
        config = linear_attn_backend._model_runner.model_config.hf_text_config
        super().__init__(
            full_attn_backend,
            linear_attn_backend,
            list(range(config.num_hidden_layers)),
        )
