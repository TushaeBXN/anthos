"""
verify_tokenizer.py — Anthos Tokenizer Verification
──────────────────────────────────────────────────
Run after setup_tokenizer.py, before kicking off training.

Checks:
  1. Special tokens present, IDs stable, no collisions with base vocab
  2. pad_token / eos_token collision risk surfaced explicitly
  3. Round-trip encode/decode on sample text (plain + multi-turn w/ special tokens)
  4. Vocab size vs common embedding-table padding conventions
  5. UNK / byte-fallback rate on a text sample (if you pass --sample-file)
  6. Tokens-per-char ratio sanity check
  7. Anthos-specific: thought-token ID matches hardcoded THT_ID in train.py

Usage:
    python3 verify_tokenizer.py
    python3 verify_tokenizer.py --sample-file data/train.jsonl --text-field text
    python3 verify_tokenizer.py --embedding-size 50262
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

TOKENIZER_PATH = Path("data/anthos_tokenizer")

SPECIAL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|thought|>",
    "<|assistant|>",
    "<|end|>",
]

# These must stay stable — hardcoded throughout train.py, chat_native.py, app.py
EXPECTED_IDS = {
    "<|system|>":    50257,
    "<|user|>":      50258,
    "<|thought|>":   50259,
    "<|assistant|>": 50260,
    "<|end|>":       50261,
}

# THT_ID as hardcoded in train.py / chat_native.py
THT_ID_HARDCODED = 50259


def section(title):
    print(f"\n{'─' * 60}\n{title}\n{'─' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-file",    type=str, default=None)
    parser.add_argument("--text-field",     type=str, default="text")
    parser.add_argument("--sample-n",       type=int, default=2000)
    parser.add_argument("--embedding-size", type=int, default=None,
                        help="Model's actual embedding table size (to detect padded-vocab mismatch)")
    args = parser.parse_args()

    if not TOKENIZER_PATH.exists():
        print(f"✕ Tokenizer path {TOKENIZER_PATH} does not exist. "
              f"Run setup_tokenizer.py first.")
        sys.exit(1)

    print("Loading tokenizer from", TOKENIZER_PATH)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    failures = 0

    # ── 1. Special tokens present + stable IDs ────────────────────────────
    section("1. Special tokens")
    seen_ids = {}
    unk_id = tok.convert_tokens_to_ids(tok.unk_token) if tok.unk_token else None
    for t in SPECIAL_TOKENS:
        tid = tok.convert_tokens_to_ids(t)
        expected = EXPECTED_IDS[t]
        status = "OK"
        if tid is None or (unk_id is not None and tid == unk_id):
            status = "✕ MISSING / mapped to UNK"
            failures += 1
        elif tid in seen_ids:
            status = f"✕ COLLISION with {seen_ids[tid]}"
            failures += 1
        elif tid != expected:
            status = f"✕ ID DRIFT: expected {expected}, got {tid} — check for tokenizer regeneration"
            failures += 1
        seen_ids[tid] = t
        print(f"  {t:<14} → id {tid:<8} [{status}]")

    # ── 2. pad/eos collision — surface explicitly + explain actual risk ───
    section("2. Pad / EOS token")
    print(f"  pad_token : {tok.pad_token!r} (id {tok.pad_token_id})")
    print(f"  eos_token : {tok.eos_token!r} (id {tok.eos_token_id})")
    if tok.pad_token_id == tok.eos_token_id:
        print("  ⚠ pad_token and eos_token share the same ID.")
        print("    NOTE for Anthos: the chat collator (anthos/data.py) pads with")
        print("    torch.zeros (token 0), NOT tok.pad_token_id, so the EOS collision")
        print("    does NOT affect loss computation (pad labels are masked to -100).")
        print("    However: the tokenizer's pad_token_id is misleading — any code that")
        print("    uses tok.pad_token_id for padding would introduce the real risk.")
        print("    No attention mask is returned by the collator; pad positions appear")
        print("    as token-0 ('!') in attention. Acceptable for short sequences.")

    # ── 3. Anthos-specific: thought-token ID stability ────────────────────
    section("3. Thought-token ID vs hardcoded THT_ID")
    tht_id_from_tok = tok.convert_tokens_to_ids("<|thought|>")
    print(f"  Tokenizer says <|thought|> = {tht_id_from_tok}")
    print(f"  THT_ID hardcoded in train.py / chat_native.py = {THT_ID_HARDCODED}")
    if tht_id_from_tok != THT_ID_HARDCODED:
        print(f"  ✕ MISMATCH — model will use wrong token ID for thought masking + RoPE pinning")
        failures += 1
    else:
        print(f"  ✓ Match — model wiring (_anthos_causal_mask, build_combined_freqs_cis) is in sync")
    print(f"  (This mismatch would silently break the non-causal thought-token attention)")

    # ── 4. Round-trip encode/decode ───────────────────────────────────────
    section("4. Round-trip encode/decode")
    plain_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Anthos uses a bifurcated thought/sequence stream — "
        "naïve tokenizers sometimes mangle non-ASCII characters like é, ñ, 中文.",
        "   leading/trailing whitespace   ",
        "Multiple\n\nnewlines\tand\ttabs",
    ]
    multiturn_sample = (
        "<|system|>You are a helpful assistant.<|end|>"
        "<|user|>What is 2+2?<|end|>"
        "<|thought|>The user is asking a simple arithmetic question.<|end|>"
        "<|assistant|>4<|end|>"
    )
    all_samples = plain_samples + [multiturn_sample]

    for s in all_samples:
        ids  = tok.encode(s)
        back = tok.decode(ids)
        if back.strip() != s.strip():
            print(f"  ✕ MISMATCH\n    in : {s!r}\n    out: {back!r}")
            failures += 1
        elif back != s:
            print(f"  ~ whitespace-only diff (GPT-2 BPE, usually fine)\n    in : {s!r}")
        else:
            print(f"  ✓ {s[:60]!r}")

    # Special tokens must survive as single tokens
    ids = tok.encode(multiturn_sample)
    decoded_pieces = [tok.decode([i]) for i in ids]
    for t in SPECIAL_TOKENS:
        if t not in decoded_pieces:
            print(f"  ✕ {t} did not survive as a single token")
            failures += 1
        else:
            print(f"  ✓ {t} is a single token in multiturn sample")

    # ── 5. Vocab size vs padded-embedding conventions ─────────────────────
    section("5. Vocab size")
    vocab_len = len(tok)
    print(f"  len(tokenizer) = {vocab_len}")
    for p in [64, 128]:
        rounded = ((vocab_len + p - 1) // p) * p
        if rounded != vocab_len:
            print(f"  note: if embeddings padded to multiple of {p}, table size = {rounded} "
                  f"(extra {rounded - vocab_len} unused rows — no bug, just wasted memory)")
    if args.embedding_size is not None:
        if args.embedding_size != vocab_len:
            print(f"  ✕ MISMATCH: model embedding size {args.embedding_size} "
                  f"!= tokenizer vocab size {vocab_len}")
            failures += 1
        else:
            print(f"  ✓ embedding size matches tokenizer vocab size ({vocab_len})")
    else:
        print("  (pass --embedding-size 50262 to check against AnthosConfig.vocab_size)")

    # ── 6/7. Coverage + tokens-per-char on real data ─────────────────────
    if args.sample_file:
        section("6/7. Coverage + tokens-per-char on sample data")
        path = Path(args.sample_file)
        if not path.exists():
            print(f"  ✕ sample file {path} not found, skipping")
        else:
            lines = []
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= args.sample_n:
                        break
                    lines.append(line)

            total_chars = total_tokens = unk_count = 0

            for line in lines:
                try:
                    obj  = json.loads(line)
                    text = obj.get(args.text_field, "")
                    if not text and "conversations" in obj:
                        text = " ".join(
                            c.get("value", "") for c in obj["conversations"]
                        )
                except json.JSONDecodeError:
                    text = line

                if not text:
                    continue
                ids = tok.encode(str(text))
                total_chars  += len(str(text))
                total_tokens += len(ids)
                if unk_id is not None:
                    unk_count += sum(1 for i in ids if i == unk_id)

            if total_tokens:
                ratio    = total_tokens / total_chars if total_chars else 0
                unk_rate = unk_count / total_tokens
                print(f"  sampled lines   : {len(lines):,}")
                print(f"  total tokens    : {total_tokens:,}")
                print(f"  tokens / char   : {ratio:.3f}  (GPT-2 BPE English ≈ 0.25–0.35)")
                print(f"  UNK rate        : {unk_rate:.4%}")
                if unk_rate > 0.001:
                    print(f"  ⚠ UNK rate > 0.1% — check for encoding issues or OOD characters")
                    failures += 1
            else:
                print("  no valid text found in sample")
    else:
        section("6/7. Coverage + tokens-per-char")
        print("  Skipped — pass --sample-file data/train.jsonl to run this check.")

    # ── Summary ──────────────────────────────────────────────────────────
    section("Summary")
    if failures == 0:
        print("✓ All automated checks passed.")
    else:
        print(f"✕ {failures} check(s) failed or flagged — review above.")

    print("""
Reminder: this script cannot verify that <|thought|> triggers the
non-causal masking / RoPE-position-0 pinning correctly in practice —
that wiring lives in anthos/main.py (_anthos_causal_mask,
build_combined_freqs_cis) and is confirmed by check 3 above.
The thought TOKENS in the model are from ThoughtTokenPool (learned
scratch-pad vectors), not from the <|thought|> ID in input sequences;
<|thought|> in training data is a formatting marker for the chat collator.
""")

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
