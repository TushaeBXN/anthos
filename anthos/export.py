"""
anthos/export.py — Export and Quantization for Anthos Deployment

Anthos uses a custom architecture that doesn't map 1:1 to any model
llama.cpp or HuggingFace natively supports. This module handles:

  1. Weight export to safetensors (for Unsloth, vLLM, HF ecosystem)
  2. Architecture metadata generation (for GGUF conversion)
  3. Quantization utilities (bfloat16, int8, int4 via bitsandbytes)
  4. Unsloth-compatible config generation for LoRA fine-tuning

Full GGUF export requires:
  - This module (config + weights)
  - llama.cpp convert_hf_to_gguf.py with a custom Anthos converter
    (see docs/gguf_export.md for the full walkthrough)

Quick path to usable model (no custom GGUF needed):
  - Export to safetensors + use with HF Transformers custom model class
  - OR: run via vLLM with --trust-remote-code
  - OR: use CachedGenerator directly in Python

Usage:
    from anthos.export import export_safetensors, export_gguf_config, quantize_model

    # Export weights
    export_safetensors(model, "exports/anthos_1b/model.safetensors")

    # Generate HF config.json (needed for HF ecosystem)
    export_hf_config(model_cfg, "exports/anthos_1b/config.json")

    # Quantize in-place to bfloat16 (recommended first step)
    model = quantize_model(model, dtype="bfloat16")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Safetensors Export
# ─────────────────────────────────────────────────────────────────────────────

def export_safetensors(
    model:     nn.Module,
    path:      str,
    dtype:     torch.dtype = torch.bfloat16,
    metadata:  Optional[dict] = None,
):
    """
    Export model weights to safetensors format.

    Safetensors is the recommended interchange format — it's:
      - Safe (no arbitrary code execution unlike pickle)
      - Fast (memory-mapped loading)
      - Compatible with HF, Unsloth, vLLM, llama.cpp

    Args:
        model:    trained Anthos model
        path:     output path (.safetensors)
        dtype:    export dtype (bfloat16 recommended — half the size of float32)
        metadata: optional dict to embed in safetensors header
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install safetensors")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Cast weights to target dtype
    state_dict = {}
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            state_dict[k] = v.to(dtype)
        else:
            state_dict[k] = v

    meta = {"format": "anthos_safetensors", "version": "0.2.0"}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})

    save_file(state_dict, path, metadata=meta)

    size_mb = Path(path).stat().st_size / 1e6
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Exported {n_params/1e6:.1f}M params → {path} ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Config Export (enables HF ecosystem compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def export_hf_config(model_cfg, output_dir: str):
    """
    Generate HuggingFace-compatible config.json for Anthos.

    This enables:
      - from_pretrained() loading with --trust-remote-code
      - HF Hub model cards
      - Unsloth LoRA fine-tuning (once Anthos is registered as a HF model)
      - vLLM serving

    Args:
        model_cfg:   AnthosConfig instance
        output_dir:  directory to write config.json (and tokenizer_config.json)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map Anthos config fields to HF standard + custom fields
    config = {
        "architectures":      ["AnthosForCausalLM"],
        "model_type":         "anthos",
        "auto_map": {
            "AutoConfig":           "configuration_anthos.AnthosConfig",
            "AutoModelForCausalLM": "modeling_anthos.AnthosForCausalLM",
        },

        # Standard HF fields (mapped from AnthosConfig)
        "vocab_size":         getattr(model_cfg, "vocab_size", 32000),
        "hidden_size":        getattr(model_cfg, "dim", 512),
        "num_attention_heads": getattr(model_cfg, "n_heads", 8),
        "num_key_value_heads": getattr(model_cfg, "n_kv_heads", 4),
        "max_position_embeddings": getattr(model_cfg, "max_seq_len", 1024),
        "torch_dtype":        "bfloat16",

        # Anthos-specific fields
        "n_thought_tokens":   getattr(model_cfg, "n_thought_tokens", 16),
        "max_loop_iters":     getattr(model_cfg, "max_loop_iters", 16),
        "n_experts":          getattr(model_cfg, "n_experts", 16),
        "expert_dim":         getattr(model_cfg, "expert_dim", 256),
        "attn_type":          getattr(model_cfg, "attn_type", "gqa"),

        # Inference defaults
        "use_cache":          True,
        "tie_word_embeddings": False,
        "transformers_version": "4.40.0",
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Minimal tokenizer config
    tok_config = {
        "model_max_length": config["max_position_embeddings"],
        "tokenizer_class":  "PreTrainedTokenizerFast",
        "bos_token":        "<|begin_of_text|>",
        "eos_token":        "<|end_of_text|>",
        "pad_token":        "<|pad|>",
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tok_config, f, indent=2)

    # generation_config.json
    gen_config = {
        "max_new_tokens":  512,
        "temperature":     0.7,
        "top_p":           0.9,
        "top_k":           40,
        "do_sample":       True,
        "repetition_penalty": 1.1,
        # Anthos-specific generation params
        "n_loops":         12,
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    print(f"✓ HF config written → {output_dir}/")
    print(f"  Load with: AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")


# ─────────────────────────────────────────────────────────────────────────────
# GGUF Metadata (for llama.cpp conversion)
# ─────────────────────────────────────────────────────────────────────────────

def export_gguf_metadata(model_cfg, output_path: str):
    """
    Generate the GGUF metadata JSON needed for llama.cpp's convert_hf_to_gguf.py.

    Full GGUF conversion requires a custom converter script in llama.cpp.
    This function generates the metadata that converter needs.

    See docs/gguf_export.md for the full llama.cpp integration walkthrough.

    Target: GGUF Q4_K_M quantization
      - ~4-5x smaller than float16
      - <1% accuracy loss on well-trained models
      - Compatible with Ollama, LM Studio, Unsloth Studio chat mode
    """
    meta = {
        "general.architecture":      "anthos",
        "general.name":              "Anthos",
        "general.author":            "Tushae Thomas",
        "general.version":           "0.2.0",
        "general.description":       "Thought-Token Bifurcated Recurrent Transformer",
        "general.license":           "MIT",

        # Standard GGUF fields (llama.cpp compatibility)
        "anthos.context_length":     getattr(model_cfg, "max_seq_len", 1024),
        "anthos.embedding_length":   getattr(model_cfg, "dim", 512),
        "anthos.attention.head_count": getattr(model_cfg, "n_heads", 8),
        "anthos.attention.head_count_kv": getattr(model_cfg, "n_kv_heads", 4),
        "anthos.vocab_size":         getattr(model_cfg, "vocab_size", 32000),

        # Anthos-specific GGUF extensions
        "anthos.thought_token_count":  getattr(model_cfg, "n_thought_tokens", 16),
        "anthos.max_loop_iterations":  getattr(model_cfg, "max_loop_iters", 16),
        "anthos.expert_count":         getattr(model_cfg, "n_experts", 16),
        "anthos.expert_feed_forward_length": getattr(model_cfg, "expert_dim", 256),
        "anthos.attention.layer_norm_rms_epsilon": 1e-5,

        # Recommended quantization
        "recommended_quantization":  "Q4_K_M",
        "quantization_notes": (
            "Q4_K_M recommended for deployment. "
            "Q8_0 for maximum accuracy. "
            "Q2_K minimum viable for experimental use."
        ),
    }

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ GGUF metadata → {output_path}")
    print(f"  Next: python llama.cpp/convert_hf_to_gguf.py <export_dir> --outtype q4_k_m")


# ─────────────────────────────────────────────────────────────────────────────
# In-place Quantization
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model(
    model:  nn.Module,
    dtype:  str = "bfloat16",
    device: str = "cpu",
) -> nn.Module:
    """
    Quantize model weights in-place.

    Supported dtypes:
      "bfloat16"  — recommended first step, half the size, minimal accuracy loss
      "float16"   — similar to bfloat16, better on CUDA
      "int8"      — requires bitsandbytes, ~4x smaller than float32
      "int4"      — requires bitsandbytes, ~8x smaller, noticeable accuracy loss

    On M1 Max: use bfloat16. Metal supports bfloat16 natively.
    For GGUF export: quantize via llama.cpp after safetensors export (better quality).
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }

    if dtype in dtype_map:
        model = model.to(dtype=dtype_map[dtype], device=device)
        n_params = sum(p.numel() for p in model.parameters())
        bits     = {"bfloat16": 16, "float16": 16, "float32": 32}[dtype]
        size_mb  = n_params * bits / 8 / 1e6
        print(f"✓ Quantized to {dtype}: ~{size_mb:.1f} MB")
        return model

    elif dtype in ("int8", "int4"):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "pip install bitsandbytes\n"
                "Note: bitsandbytes requires CUDA. For M1 Mac, use bfloat16 instead."
            )

        bits = 8 if dtype == "int8" else 4
        print(f"  Applying {bits}-bit quantization via bitsandbytes...")
        # bitsandbytes quantization is applied layer by layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if bits == 8:
                    new_module = bnb.nn.Linear8bitLt(
                        module.in_features, module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                    )
                else:
                    new_module = bnb.nn.Linear4bit(
                        module.in_features, module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.bfloat16,
                    )
                # Set weights (quantized on first forward pass)
                new_module.weight = module.weight
                if module.bias is not None:
                    new_module.bias = module.bias
                # Replace in model
                parent = model
                parts  = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], new_module)

        print(f"✓ {bits}-bit quantization applied")
        return model

    else:
        raise ValueError(f"Unknown dtype: {dtype}. Choose from: bfloat16, float16, int8, int4")


# ─────────────────────────────────────────────────────────────────────────────
# Full Export Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def export_for_deployment(
    model,
    model_cfg,
    output_dir:  str,
    dtype:       str = "bfloat16",
    gguf_meta:   bool = True,
):
    """
    One-shot export: safetensors + HF config + GGUF metadata.

    After running this:
      1. Model is loadable via HF Transformers (trust_remote_code=True)
      2. GGUF metadata is ready for llama.cpp conversion
      3. Safetensors file can be loaded by Unsloth Studio (chat mode)
      4. Can be pushed to HuggingFace Hub directly

    Args:
        model:      trained Anthos model
        model_cfg:  AnthosConfig
        output_dir: directory for all export files
        dtype:      export weight dtype
        gguf_meta:  whether to generate GGUF metadata
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"\n── Anthos Export Pipeline ──────────────────────────────────────")

    # 1. Weights
    export_safetensors(
        model,
        str(out / "model.safetensors"),
        dtype=torch_dtype,
        metadata={"dim": str(getattr(model_cfg, "dim", 512))},
    )

    # 2. HF config
    export_hf_config(model_cfg, str(out))

    # 3. GGUF metadata
    if gguf_meta:
        export_gguf_metadata(model_cfg, str(out / "gguf_metadata.json"))

    print(f"\n✓ Export complete → {out}/")
    print(f"  Files: model.safetensors, config.json, tokenizer_config.json,")
    print(f"         generation_config.json, gguf_metadata.json")
    print(f"\n  Next steps:")
    print(f"  1. Push to HF Hub: huggingface-cli upload TushaeThomas/anthos-1b {out}/")
    print(f"  2. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py {out}/ --outtype q4_k_m")
    print(f"  3. Run in Unsloth Studio: load the .gguf file in Chat mode")
