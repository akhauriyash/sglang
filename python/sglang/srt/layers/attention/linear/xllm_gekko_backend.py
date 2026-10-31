# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Pooled state for the xLLM Gekko recurrence and fixed-shape decode.

All persistent tensors belong to SGLang's request allocator. The reference
operator remains responsible for architectural chunk timing; scheduler prefill
pieces never trigger a memory update on their own. This backend requires TP1,
BF16 and disabled prefix reuse/speculation. Ordinary decode can be captured;
prefill and 256-token state commits use the functional recurrence.
"""

from functools import cached_property

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    ShortConvHybridAttnBackend,
)
from sglang.srt.layers.attention.linear.short_conv_backend import ShortConvAttnBackend
from sglang.srt.models.xllm_gekko_reference import (
    GekkoState,
    supports_pooled_gekko_decode,
)


class XllmGekkoAttnBackend(ShortConvAttnBackend):
    needs_cpu_seq_lens = True
    needs_extend_host_mirrors = False
    cache_indices_dtype = torch.int32

    @cached_property
    def supports_pooled_decode(self):
        config = self._model_runner.model_config.hf_text_config
        return supports_pooled_gekko_decode(config.gekko_attention)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        buffer = getattr(self, "_decode_positions_buf", None)
        if buffer is not None:
            assert len(buffer) >= max_bs, "Size Gekko metadata before graph capture"
            return
        self._decode_positions_buf = torch.empty(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._decode_lengths_buf = torch.empty_like(self._decode_positions_buf)

    def can_run_decode_cuda_graph(self, forward_batch):
        if (
            not self.supports_pooled_decode
            or not forward_batch.forward_mode.is_decode()
        ):
            return False
        lengths = forward_batch.seq_lens_cpu
        # The incoming token finishes the block when seq_len is a multiple of
        # 256. Chunk commits (4096) are a subset of these eager steps. Read the
        # existing CPU mirror so eligibility never synchronizes the GPU.
        return lengths is not None and bool(
            ((lengths > 1) & (lengths.remainder(256) != 0)).all()
        )

    def _prepare_decode(self, forward_batch):
        from sglang.srt.models.xllm_gekko_decode import prepare_decode_metadata

        size = len(self._cache_indices)
        buffer = getattr(self, "_decode_positions_buf", None)
        if buffer is not None and size <= len(buffer):
            self._decode_positions = buffer[:size]
            self._decode_lengths = self._decode_lengths_buf[:size]
        else:
            self._decode_positions = torch.empty_like(self._cache_indices)
            self._decode_lengths = torch.empty_like(self._cache_indices)
        prepare_decode_metadata(
            self._cache_indices,
            forward_batch.seq_lens,
            self._decode_positions,
            self._decode_lengths,
        )
        self._use_pooled_decode = True

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        if in_capture:
            # Warmup and capture must not advance live or allocator-reserved
            # state. Every pooled kernel masks negative slots before access.
            self._cache_indices.fill_(-1)
        self._prepare_decode(forward_batch)

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        self._use_pooled_decode = False
        if (
            self._cache_indices is not None
            and self._cache_indices.is_cuda
            and self.can_run_decode_cuda_graph(forward_batch)
        ):
            self._prepare_decode(forward_batch)
            return
        slots = self._cache_indices.tolist()
        if len(set(slots)) != len(slots):
            raise ValueError("Every Gekko batch row must own a distinct request slot")
        if forward_batch.forward_mode.is_decode():
            prefixes = (forward_batch.seq_lens_cpu - 1).tolist()
            lengths = [1] * len(slots)
        elif forward_batch.forward_mode.is_extend():
            prefixes = forward_batch.extend_prefix_lens_cpu
            lengths = forward_batch.extend_seq_lens_cpu
        elif forward_batch.forward_mode.is_idle():
            prefixes, lengths = [], []
        else:
            raise ValueError("Gekko state supports ordinary prefill and decode only")
        self.pieces = []
        offset = 0
        for slot, prefix, length in zip(slots, prefixes, lengths, strict=True):
            self.pieces.append((int(slot), int(prefix), slice(offset, offset + length)))
            offset += length

    @staticmethod
    def _read_state(layer, cache, slot, position, hidden):
        if position == 0:
            return layer.initial_state(hidden)
        nk, dk, dv = layer.num_kv_heads, layer.head_dim, layer.v_head_dim
        retained = (
            position
            if position < layer.chunk_size
            else layer.chunk_size + position % layer.chunk_size
        )
        conv = [stream[slot] for stream in cache.conv]
        temporal = cache.temporal[slot].flatten()
        matrix_end = nk * dk * dv
        normalizer_end = matrix_end + nk * dk
        groups = layer.timenorm.num_groups
        packed_start = normalizer_end + 2 * groups
        # Count follows absolute request progress. Keeping the saved int64 prior
        # separately avoids packing an integer counter into FP32 state storage.
        count = layer.timenorm.prior_count + position
        return GekkoState(
            position=position,
            tdn_count=count,
            tdn_mean=temporal[normalizer_end : normalizer_end + groups],
            tdn_var=temporal[normalizer_end + groups : normalizer_end + 2 * groups],
            q_history=conv[0],
            k_history=conv[1],
            v_history=conv[2],
            sca_k=conv[3][:retained].view(retained, nk, dk),
            values=conv[4][:retained].view(retained, nk, dv),
            feature_k=conv[5][:retained].view(retained, nk, dk),
            raw_k=conv[6][:retained].view(retained, nk, dk),
            memory=temporal[:matrix_end].view(nk, dk, dv).to(hidden.dtype),
            log_normalizer=temporal[matrix_end:normalizer_end].view(nk, dk),
            tdn_packed=(
                temporal[packed_start:] if layer.uses_native_tdn(hidden) else None
            ),
        )

    @staticmethod
    def _write_state(cache, slot, state):
        for stream, value in zip(
            cache.conv,
            (
                state.q_history,
                state.k_history,
                state.v_history,
                state.sca_k.flatten(1),
                state.values.flatten(1),
                state.feature_k.flatten(1),
                state.raw_k.flatten(1),
            ),
            strict=True,
        ):
            stream[slot, : len(value)].copy_(value)
        temporal = cache.temporal[slot].flatten()
        offset = 0
        for value in (
            state.memory,
            state.log_normalizer,
            state.tdn_mean,
            state.tdn_var,
        ):
            temporal[offset : offset + value.numel()].copy_(value.flatten())
            offset += value.numel()
        if state.tdn_packed is not None:
            if temporal.numel() - offset != state.tdn_packed.numel():
                raise ValueError("Gekko pool does not match the packed TDN state size")
            temporal[offset:].copy_(state.tdn_packed)

    def forward_gekko(self, layer_id, layer, hidden, forward_batch):
        cache = self.conv_state_metadata(layer_id, forward_batch).layer_cache
        if self._use_pooled_decode:
            from sglang.srt.models.xllm_gekko_decode import pooled_gekko_decode

            return pooled_gekko_decode(
                layer,
                hidden,
                cache,
                self._cache_indices,
                self._decode_positions,
                self._decode_lengths,
            )
        if sum(piece.stop - piece.start for _, _, piece in self.pieces) != len(hidden):
            raise ValueError("Gekko request metadata does not cover the input tokens")
        result = torch.empty_like(hidden)
        for slot, position, piece in self.pieces:
            if slot < 0:
                raise ValueError("Padded graph slots are unsupported for eager Gekko")
            state = self._read_state(layer, cache, slot, position, hidden[piece])
            output, state = layer(hidden[piece], state)
            result[piece] = output
            self._write_state(cache, slot, state)
        return result


class XllmGekkoHybridAttnBackend(ShortConvHybridAttnBackend):
    def __init__(self, full_attn_backend, short_conv_backend, full_attn_layers):
        super().__init__(full_attn_backend, short_conv_backend, full_attn_layers)
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        if (
            isinstance(full_attn_backend, FlashAttentionBackend)
            and full_attn_backend.fa_impl_ver == 4
        ):
            # Match native Gekko full-prefix accumulation. Automatic split-KV
            # changes BF16 decode rounding with request length and batch size.
            # Keep this model-local; global deterministic mode also changes GEMMs.
            full_attn_backend.num_splits = 1
            full_attn_backend.decode_num_splits = 1

    def forward_gekko(self, layer_id, layer, hidden, forward_batch):
        return self.short_conv_backend.forward_gekko(
            layer_id, layer, hidden, forward_batch
        )
