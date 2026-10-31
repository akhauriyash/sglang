"""Dependency-free, strict configuration contract for native xLLM exports.

This deliberately preserves adjacent-pair RoPE and zero-centered norm weights.
It does not convert these models to Llama, nor infer head_dim from hidden_size.
"""

import re
from copy import deepcopy

ARCHITECTURES = {
    "transformer": "XllmDenseForCausalLM",
    "hybrid_inkling": "XllmInklingForCausalLM",
    "hybrid_gekko": "XllmGekkoForCausalLM",
}


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be an explicit positive integer, got {value!r}")
    return value


def resolve_config(saved_config):
    """Resolve serialized family overrides without importing training code."""
    model = deepcopy(saved_config["model"])
    arch = model["arch"]
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unsupported native architecture {arch!r}")
    for key in ("model_dim", "ffn_hidden_dim", "num_layers", "vocab_size", "head_dim"):
        _positive_int(model.get(key), key)
    for key, expected in {
        "apply_rmsnorm": True,
        "norm_affine": True,
        "swiglu": True,
        "scale_emb": False,
        "num_experts": 0,
        "num_values": 0,
        "residual_func": "base",
        "layernorm_num_groups": 1,
        "rescale_nffn": False,
        "two_hop_residual": False,
    }.items():
        if model.get(key, expected) != expected:
            raise ValueError(f"Unsupported {key}={model[key]!r}")
    if model.get("output_size", -1) not in (-1, model["vocab_size"]):
        raise ValueError("Output vocabulary must match input vocabulary")
    converted = {"hybrid_gekko": "gekko", "hybrid_inkling": "inkling"}.get(arch)
    kinds = model.get("layer_kinds") if converted else ["full"] * model["num_layers"]
    if not kinds or len(kinds) != model["num_layers"]:
        raise ValueError("layer_kinds must specify every layer")
    if any(kind not in {"full", converted} for kind in kinds):
        raise ValueError("layer_kinds contains an unsupported family")
    families = {}
    shared = {
        k: v
        for k, v in model.items()
        if k
        not in ("full_attention", "gekko_attention", "inkling_attention", "layer_kinds")
    }
    for kind in sorted(set(kinds)):
        family = deepcopy(shared)
        if converted:
            overrides = model.get(f"{kind}_attention")
            if not isinstance(overrides, dict):
                raise ValueError(f"Missing serialized {kind}_attention overrides")
            family.update(deepcopy(overrides))
            if family.get("layerwise_ckpt") is None:
                family["layerwise_ckpt"] = model.get("layerwise_ckpt", False)
        family["arch"] = "gekko" if kind == "gekko" else "transformer"
        family["rmsnorm_eps"] = model.get("rmsnorm_eps", model.get("norm_eps", 1e-6))
        family["v_head_dim"] = family.get("v_head_dim") or family["head_dim"]
        family["num_kv_heads"] = family.get("num_kv_heads") or family["num_heads"]
        if family.get("rope_head_dim") is None:
            family["rope_head_dim"] = family["head_dim"]
        for key in ("num_heads", "num_kv_heads", "head_dim", "v_head_dim"):
            _positive_int(family.get(key), f"{kind}.{key}")
        if family["num_heads"] % family["num_kv_heads"]:
            raise ValueError(f"{kind}: query heads must be divisible by KV heads")
        if family.get("attn_act_func", "softmax") != "softmax":
            raise ValueError("Only softmax attention is supported")
        if family.get("qknorm", False) or family.get("apply_attn_gate", False):
            raise ValueError(
                "Optional source Q/K normalization and attention gates are not supported"
            )
        rotary_dim = family["rope_head_dim"]
        if (
            type(rotary_dim) is not int
            or rotary_dim % 2
            or not 0 <= rotary_dim <= family["head_dim"]
        ):
            raise ValueError(f"Invalid {kind} rotary dimension")
        if kind == "inkling":
            _positive_int(
                family.get("local_attention_window"), "local_attention_window"
            )
            if family.get("causal_conv_weight_normalization") is not False:
                raise ValueError("Inkling requires raw convolution weights")
        if kind in {"inkling", "gekko"}:
            _positive_int(family.get("causal_conv_width"), "causal_conv_width")
        families[kind] = family
    first = families[kinds[0]]
    config = {
        "model_type": "xllm_hybrid",
        "architectures": [ARCHITECTURES[arch]],
        "xllm_export_schema_version": 1,
        "native_model_config": model,
        "hidden_size": model["model_dim"],
        "intermediate_size": model["ffn_hidden_dim"],
        "num_hidden_layers": model["num_layers"],
        "vocab_size": model["vocab_size"],
        "num_attention_heads": first["num_heads"],
        "num_key_value_heads": first["num_kv_heads"],
        "head_dim": first["head_dim"],
        "v_head_dim": first["v_head_dim"],
        "rms_norm_eps": first["rmsnorm_eps"],
        "hidden_act": "silu",
        "max_position_embeddings": model["max_position_embeddings"],
        "rope_theta": first["rope_base"],
        "layer_kinds": kinds,
        "native_norm_offset": True,
        "rotary_interleaved": True,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": None,
        "use_cache": True,
        "torch_dtype": {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(
            saved_config.get("dtype", "bf16"), saved_config.get("dtype", "bfloat16")
        ),
    }
    config.update({f"{kind}_attention": family for kind, family in families.items()})
    return config


def canonical_name(name):
    """The only weight-name change: legacy dense pre-attention RMSNorm."""
    return re.sub(
        r"^layers\.(\d+)\.norm\.weight$", r"layers.\1.attention.norm.weight", name
    )


def expected_tensors(config):
    """Expected canonical tensor shapes and parameter/buffer roles for TP1."""
    width, hidden, vocab = (
        config["hidden_size"],
        config["intermediate_size"],
        config["vocab_size"],
    )
    result = {}

    def add(name, shape, role="parameter"):
        result[name] = {"shape": list(shape), "role": role}

    add("embed.weight", (vocab, width))
    add("output.final_norm.weight", (width,))
    add("output.output.weight", (vocab, width))
    for index, kind in enumerate(config["layer_kinds"]):
        family = config[f"{kind}_attention"]
        prefix = f"layers.{index}"
        q = family["num_heads"] * family["head_dim"]
        k = family["num_kv_heads"] * family["head_dim"]
        v = family["num_kv_heads"] * family["v_head_dim"]
        out = family["num_heads"] * family["v_head_dim"]
        attn = f"{prefix}.{'gda' if kind == 'gekko' else 'attention'}"
        for name, shape in {
            "wq.weight": (q, width),
            "wk.weight": (k, width),
            "wv.weight": (v, width),
            "wo.weight": (width, out),
        }.items():
            add(f"{attn}.{name}", shape)
        if kind == "gekko":
            groups = family.get("timenorm_num_groups") or width
            if width % groups:
                raise ValueError("TDN groups must divide model width")
            for name in ("weight", "bias"):
                add(f"{attn}.timenorm.{name}", (width,))
            add(f"{attn}.timenorm.prior_count", (), "buffer")
            for name in ("prior_mean", "prior_var"):
                add(f"{attn}.timenorm.{name}", (groups,), "buffer")
            add(f"{attn}.wv.bias", (v,))
            add(f"{attn}.wr.weight", (out, width))
            if family.get("apply_bias_term", False):
                add(f"{attn}.wr.bias", (out,))
            add(f"{attn}.query_norm.weight", (q,))
            add(f"{attn}.key_norm.weight", (k,))
            for name, channels in (("q", q), ("k", k), ("v", v)):
                add(
                    f"{attn}.{name}_conv.weight",
                    (channels, family["causal_conv_width"]),
                )
        else:
            add(f"{attn}.norm.weight", (width,))
        add(f"{prefix}.nffn.norm.weight", (width,))
        add(f"{prefix}.nffn.fc1.weight", (hidden, width))
        add(f"{prefix}.nffn.fc2.weight", (width, hidden))
        add(f"{prefix}.nffn.fc3.weight", (hidden, width))
        if kind == "inkling":
            for name, channels in (
                ("attention.k_shortconv", k),
                ("attention.v_shortconv", v),
                ("attention.out_shortconv", width),
                ("nffn.out_shortconv", width),
            ):
                add(f"{prefix}.{name}.weight", (channels, family["causal_conv_width"]))
    return result


def validate_inventory(config, inventory):
    expected = expected_tensors(config)
    canonical = {}
    for source, item in inventory.items():
        destination = canonical_name(source)
        if destination in canonical:
            raise ValueError(f"Duplicate mapping to {destination}")
        canonical[destination] = item
    missing, unexpected = (
        expected.keys() - canonical.keys(),
        canonical.keys() - expected.keys(),
    )
    if missing or unexpected:
        raise ValueError(
            f"Tensor inventory mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, spec in expected.items():
        if canonical[name]["shape"] != spec["shape"]:
            raise ValueError(
                f"Shape mismatch for {name}: {canonical[name]['shape']} != {spec['shape']}"
            )
        dtype = canonical[name]["dtype"]
        if (name.endswith(".prior_count") and dtype != "int64") or (
            not name.endswith(".prior_count")
            and dtype not in {"float32", "bfloat16", "float16"}
        ):
            raise ValueError(f"Unsupported tensor dtype for {name}: {dtype}")
    return expected
