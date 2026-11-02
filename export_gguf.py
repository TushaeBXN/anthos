"""
export_gguf.py — Export Anthos checkpoint to GGUF format for llama.cpp

GGUF (GPT-Generated Unified Format) is llama.cpp's binary format for
CPU-efficient inference. This script converts an Anthos checkpoint into
a GGUF file that can be loaded by llama.cpp, Ollama, and LM Studio.

Architecture mapping:
  Anthos is a custom architecture (recurrent + MoE + thought tokens).
  llama.cpp does not natively support it, so this exporter maps Anthos
  to the closest supported architecture: LLaMA-style transformer.

  Mapping (approximate, no recurrent unrolling):
    Anthos prelude blocks  → LLaMA decoder layers 0…prelude_layers-1
    Anthos coda blocks     → LLaMA decoder layers prelude_layers…end
    AnthosRecurrentBlock   → one LLaMA decoder layer (single pass, no loops)
    MoE FFN                → dense FFN using the top-1 expert's weights
    Thought tokens         → dropped (inference only)

  This produces a lossy export: the recurrent depth and MoE routing are
  not preserved. For full fidelity, use the Python inference path instead.

Usage:
    python export_gguf.py --checkpoint checkpoints/sft/step_000500.pt \
                          --tier sft \
                          --output anthos-q4_k_m.gguf \
                          --quant q4_k_m

Requirements:
    pip install gguf     # pip install gguf (official Python bindings)
    pip install sentencepiece  # for tokenizer vocab export
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from anthos.main    import Anthos
from anthos.configs import get_training_config


# ─────────────────────────────────────────────────────────────────────────────
# GGUF quant type table
# ─────────────────────────────────────────────────────────────────────────────

QUANT_TYPE = {
    "f32":    0,
    "f16":    1,
    "q4_0":   2,
    "q4_1":   3,
    "q5_0":   6,
    "q5_1":   7,
    "q8_0":   8,
    "q8_1":   9,
    "q2_k":   10,
    "q3_k_s": 11,
    "q3_k_m": 12,
    "q3_k_l": 13,
    "q4_k_s": 14,
    "q4_k_m": 15,
    "q5_k_s": 16,
    "q5_k_m": 17,
    "q6_k":   18,
    "q8_k":   19,
    "bf16":   30,
}


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_linear_weight(state_dict: dict, key: str) -> torch.Tensor | None:
    """Return a weight tensor from the state dict, or None if not found."""
    for candidate in [key, key + ".weight"]:
        if candidate in state_dict:
            return state_dict[candidate].float()
    return None


def _get_top1_expert_weights(state_dict: dict, layer_prefix: str) -> dict | None:
    """
    For MoE layers, extract the top-1 expert's gate/up/down weights by looking
    for the expert whose L2 norm is highest (a proxy for the most-used expert).
    Returns {"gate": tensor, "up": tensor, "down": tensor} or None.
    """
    # Keys: layer_prefix.ffn.routed_experts.{N}.gate/up/down.weight
    expert_keys: dict[int, dict] = {}
    for k, v in state_dict.items():
        if layer_prefix + ".ffn.routed_experts." not in k:
            continue
        parts = k.split(".")
        try:
            exp_idx = int(parts[parts.index("routed_experts") + 1])
        except (ValueError, IndexError):
            continue
        if exp_idx not in expert_keys:
            expert_keys[exp_idx] = {}
        if ".gate." in k:
            expert_keys[exp_idx]["gate"] = v.float()
        elif ".up." in k:
            expert_keys[exp_idx]["up"] = v.float()
        elif ".down." in k:
            expert_keys[exp_idx]["down"] = v.float()

    if not expert_keys:
        return None

    # Pick expert with highest total weight norm (most-used heuristic)
    best_idx = max(
        expert_keys.keys(),
        key=lambda i: sum(v.norm() for v in expert_keys[i].values()),
    )
    return expert_keys[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# GGUF writer (minimal implementation using the gguf Python package)
# ─────────────────────────────────────────────────────────────────────────────

def export_to_gguf(
    model:       Anthos,
    model_cfg,
    output_path: Path,
    quant:       str  = "f16",
    tokenizer_path: str = "gpt2",
) -> None:
    """
    Export an Anthos model to GGUF format.

    This requires the `gguf` Python package (pip install gguf).
    """
    try:
        from gguf import GGUFWriter, GGMLQuantizationType
    except ImportError:
        raise ImportError(
            "gguf package not installed.\n"
            "Run: pip install gguf"
        )

    quant_upper = quant.upper().replace("_", "_")

    print(f"[gguf] Exporting to {output_path} (quant={quant})...", flush=True)
    writer = GGUFWriter(str(output_path), arch="llama")

    # ── Architecture metadata ────────────────────────────────────────────────
    dim        = model_cfg.dim
    n_heads    = model_cfg.n_heads
    n_kv_heads = model_cfg.n_kv_heads
    n_layers   = model_cfg.prelude_layers + 1 + model_cfg.coda_layers   # approx

    writer.add_name("Anthos")
    writer.add_description(
        f"Anthos Thought-Token Recurrent Transformer — "
        f"exported from step checkpoint (recurrent depth=1, MoE collapsed to top-1)"
    )
    writer.add_context_length(model_cfg.max_seq_len)
    writer.add_embedding_length(dim)
    writer.add_block_count(n_layers)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_freq_base(model_cfg.rope_theta)
    writer.add_vocab_size(model_cfg.vocab_size)
    writer.add_file_type(QUANT_TYPE.get(quant, 1))
    # FFN intermediate dim: prelude/coda dense layers use 4/3 * dim ≈ 1024.
    # MoE recurrent layer uses expert_dim=384; we zero-pad it to ffn_dim at export.
    ffn_dim = getattr(model_cfg, "ffn_dim", None) or (dim * 4 // 3)
    writer.add_feed_forward_length(ffn_dim)
    # Store ffn_dim on model_cfg so _add_transformer_layer can access it
    model_cfg._export_ffn_dim = ffn_dim

    # ── Tokenizer ────────────────────────────────────────────────────────────
    try:
        import tiktoken

        # GPT-2 bytes-to-unicode: every byte gets a unique printable char,
        # so token strings are always distinct (no UTF-8 replacement collisions).
        def _b2u():
            bs = (list(range(ord("!"), ord("~") + 1))
                  + list(range(ord("¡"), ord("¬") + 1))
                  + list(range(ord("®"), ord("ÿ") + 1)))
            cs = bs[:]
            n = 0
            for b in range(256):
                if b not in bs:
                    bs.append(b)
                    cs.append(256 + n)
                    n += 1
            return {b: chr(c) for b, c in zip(bs, cs)}

        byte_enc = _b2u()

        def _bytes_to_str(raw: bytes) -> str:
            return "".join(byte_enc[b] for b in raw)

        enc = tiktoken.get_encoding("cl100k_base")
        vocab_size = model_cfg.vocab_size  # 100352 for Anthos

        # Build id→bytes map
        id_to_bytes: dict[int, bytes] = {v: k for k, v in enc._mergeable_ranks.items()}
        for tok_str, tok_id in enc._special_tokens.items():
            id_to_bytes[tok_id] = tok_str.encode("utf-8")

        token_strs = []
        token_types_list = []
        for i in range(vocab_size):
            raw = id_to_bytes.get(i, f"\x00[UNK{i}]".encode())
            token_strs.append(_bytes_to_str(raw))
            # 1 = NORMAL, 3 = CONTROL (special tokens)
            token_types_list.append(3 if i in enc._special_tokens.values() else 1)

        # Reconstruct BPE merge rules from rank ordering
        ranks = enc._mergeable_ranks
        merges = []
        for token_bytes, rank in sorted(ranks.items(), key=lambda x: x[1]):
            if rank >= vocab_size or len(token_bytes) <= 1:
                continue
            best_pair = None
            best_max = float("inf")
            for i in range(1, len(token_bytes)):
                left, right = token_bytes[:i], token_bytes[i:]
                if left in ranks and right in ranks:
                    mx = max(ranks[left], ranks[right])
                    if mx < best_max:
                        best_max = mx
                        best_pair = (ranks[left], ranks[right])
            if best_pair:
                l, r = best_pair
                if l < vocab_size and r < vocab_size:
                    merges.append(f"{token_strs[l]} {token_strs[r]}")

        scores = [0.0] * vocab_size
        eot_id = enc._special_tokens.get("<|endoftext|>", 100257)

        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(token_strs)
        writer.add_token_scores(scores)
        writer.add_token_types(token_types_list)
        writer.add_token_merges(merges)
        writer.add_bos_token_id(eot_id)
        writer.add_eos_token_id(eot_id)
        print(f"  [gguf] Tokenizer: {vocab_size} tokens, {len(merges)} merges (cl100k_base)")
    except Exception as e:
        print(f"  [gguf] Tokenizer export skipped: {e}")

    # ── Weight extraction ─────────────────────────────────────────────────────
    state_dict = model.state_dict()

    def _tensor(t: torch.Tensor):
        import numpy as np
        t = t.detach().float()
        # 1-D tensors (norms, biases) must stay f32; only quantize weight matrices
        if t.ndim >= 2 and quant in ("f16", "q4_0", "q4_1", "q4_k_m", "q8_0"):
            return t.half().numpy().astype(np.float16)
        return t.numpy().astype(np.float32)

    # Token embedding
    embed = _extract_linear_weight(state_dict, "embed.weight")
    if embed is not None:
        writer.add_tensor("token_embd.weight", _tensor(embed))
        print("  [gguf] ✓ token embeddings")

    # Output norm + head
    norm = _extract_linear_weight(state_dict, "norm.weight")
    if norm is not None:
        writer.add_tensor("output_norm.weight", _tensor(norm))
    head_w = _extract_linear_weight(state_dict, "head.weight")
    if head_w is not None:
        writer.add_tensor("output.weight", _tensor(head_w))
        print("  [gguf] ✓ output head")

    ffn_dim = getattr(model_cfg, "_export_ffn_dim", dim * 4 // 3)

    # Prelude blocks → LLaMA layers 0…prelude_layers-1
    for i in range(model_cfg.prelude_layers):
        pfx = f"prelude.{i}"
        _add_transformer_layer(writer, state_dict, pfx, i, _tensor, ffn_dim=ffn_dim)

    # Recurrent block → one LLaMA layer
    recur_idx = model_cfg.prelude_layers
    pfx = "recurrent.block"
    _add_transformer_layer(writer, state_dict, pfx, recur_idx, _tensor,
                           is_moe=True, ffn_dim=ffn_dim)

    # Coda blocks
    for i in range(model_cfg.coda_layers):
        pfx = f"coda.{i}"
        layer_idx = model_cfg.prelude_layers + 1 + i
        _add_transformer_layer(writer, state_dict, pfx, layer_idx, _tensor, ffn_dim=ffn_dim)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"[gguf] Done → {output_path} ({size_mb:.0f} MB)")


def _add_transformer_layer(
    writer,
    state_dict: dict,
    pfx:        str,
    layer_idx:  int,
    _tensor,
    is_moe:     bool = False,
    ffn_dim:    int  = 0,
) -> None:
    """Add one transformer block's tensors to the GGUF writer."""
    li = layer_idx

    # Attention norms (Anthos uses attn_norm / ffn_norm)
    for norm_key, gguf_name in [
        (f"{pfx}.attn_norm.weight", f"blk.{li}.attn_norm.weight"),
        (f"{pfx}.ffn_norm.weight",  f"blk.{li}.ffn_norm.weight"),
    ]:
        if norm_key in state_dict:
            writer.add_tensor(gguf_name, _tensor(state_dict[norm_key].float()))

    # Attention QKV projections (Anthos uses wq/wk/wv/wo)
    for src, dst in [
        (f"{pfx}.attn.wq.weight",  f"blk.{li}.attn_q.weight"),
        (f"{pfx}.attn.wk.weight",  f"blk.{li}.attn_k.weight"),
        (f"{pfx}.attn.wv.weight",  f"blk.{li}.attn_v.weight"),
        (f"{pfx}.attn.wo.weight",  f"blk.{li}.attn_output.weight"),
    ]:
        if src in state_dict:
            writer.add_tensor(dst, _tensor(state_dict[src].float()))

    # FFN weights
    if is_moe:
        # Extract top-1 routed expert FFN weights; zero-pad to ffn_dim if smaller
        expert = _get_top1_expert_weights(state_dict, pfx)
        if expert and ffn_dim > 0:
            import numpy as np
            for key, dst in [
                ("gate", f"blk.{li}.ffn_gate.weight"),
                ("up",   f"blk.{li}.ffn_up.weight"),
            ]:
                if key in expert:
                    w = expert[key]           # shape [expert_dim, dim]
                    ed, d = w.shape
                    if ed < ffn_dim:
                        pad = torch.zeros(ffn_dim - ed, d, dtype=w.dtype)
                        w = torch.cat([w, pad], dim=0)
                    writer.add_tensor(dst, _tensor(w))
            if "down" in expert:
                w = expert["down"]            # shape [dim, expert_dim]
                d, ed = w.shape
                if ed < ffn_dim:
                    pad = torch.zeros(d, ffn_dim - ed, dtype=w.dtype)
                    w = torch.cat([w, pad], dim=1)
                writer.add_tensor(f"blk.{li}.ffn_down.weight", _tensor(w))
    else:
        # Anthos dense FFN uses gate/up/down (not gate_proj/up_proj/down_proj)
        for src, dst in [
            (f"{pfx}.ffn.gate.weight", f"blk.{li}.ffn_gate.weight"),
            (f"{pfx}.ffn.up.weight",   f"blk.{li}.ffn_up.weight"),
            (f"{pfx}.ffn.down.weight", f"blk.{li}.ffn_down.weight"),
        ]:
            if src in state_dict:
                writer.add_tensor(dst, _tensor(state_dict[src].float()))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export Anthos to GGUF")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--tier",       default="proof",
                        help="Config tier used to build the model")
    parser.add_argument("--output",     default=None,
                        help="Output .gguf path (default: <checkpoint>.gguf)")
    parser.add_argument("--quant",      default="f16",
                        choices=list(QUANT_TYPE.keys()),
                        help="Quantization type (default: f16)")
    parser.add_argument("--tokenizer",  default=None,
                        help="Tokenizer path (default: auto)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path  = Path(args.output) if args.output else ckpt_path.with_suffix(".gguf")

    print(f"\n{'─'*60}")
    print(f"  Anthos → GGUF Export")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {out_path}")
    print(f"  Quant      : {args.quant}")
    print(f"{'─'*60}\n")

    model_cfg, _ = get_training_config(args.tier)
    # apply checkpoint-specific overrides when the tier defaults don't match
    if args.tier == "proof":
        ckpt_tmp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt_tmp["model"]
        if "embed.weight" in sd:
            model_cfg.vocab_size  = sd["embed.weight"].shape[0]
            model_cfg.dim         = sd["embed.weight"].shape[1]
        if "recurrent.loop_embeds" in sd:
            model_cfg.max_loop_iters = sd["recurrent.loop_embeds"].shape[0]
        if "recurrent.block.ffn.routed_experts.0.gate.weight" in sd:
            model_cfg.expert_dim  = sd["recurrent.block.ffn.routed_experts.0.gate.weight"].shape[0]
        if "freqs_cis" in sd:
            head_dim = sd["freqs_cis"].shape[1] * 2
            model_cfg.n_heads = model_cfg.dim // head_dim
        if "prelude.0.attn.wk.weight" in sd:
            kv_dim = sd["prelude.0.attn.wk.weight"].shape[0]
            head_dim = model_cfg.dim // model_cfg.n_heads
            model_cfg.n_kv_heads = kv_dim // head_dim
        del ckpt_tmp
    model        = Anthos(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    tok_path = args.tokenizer
    if not tok_path:
        if Path("data/anthos_tokenizer").exists():
            tok_path = "data/anthos_tokenizer"
        else:
            tok_path = "gpt2"

    export_to_gguf(model, model_cfg, out_path, quant=args.quant, tokenizer_path=tok_path)


if __name__ == "__main__":
    main()
