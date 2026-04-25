"""
setup_tokenizer.py — Anthos Custom Tokenizer Setup
────────────────────────────────────────────────────
Adds special tokens to GPT-2 tokenizer and saves to data/anthos_tokenizer/.

Special tokens:
  <|system|>     — system prompt boundary
  <|user|>       — user turn
  <|thought|>    — model's internal reasoning (maps to Anthos thought stream)
  <|assistant|>  — model's spoken response
  <|end|>        — turn end marker

Run once before SFT training:
    python3 setup_tokenizer.py

New vocab size: 50257 + 5 = 50262
"""

from pathlib import Path
from transformers import AutoTokenizer

SAVE_PATH = Path("data/anthos_tokenizer")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

print("Loading GPT-2 tokenizer...")
tok = AutoTokenizer.from_pretrained("gpt2")

# ── Add Anthos special tokens ─────────────────────────────────────────────────
special_tokens = {
    "additional_special_tokens": [
        "<|system|>",
        "<|user|>",
        "<|thought|>",
        "<|assistant|>",
        "<|end|>",
    ]
}

num_added = tok.add_special_tokens(special_tokens)
tok.pad_token = tok.eos_token

# Save
tok.save_pretrained(SAVE_PATH)

print(f"""
✓ Anthos tokenizer saved → {SAVE_PATH}
  Base vocab      : 50,257
  Special tokens  : {num_added} added
  New vocab size  : {len(tok)}

Token IDs:
  <|system|>    → {tok.convert_tokens_to_ids('<|system|>')}
  <|user|>      → {tok.convert_tokens_to_ids('<|user|>')}
  <|thought|>   → {tok.convert_tokens_to_ids('<|thought|>')}
  <|assistant|> → {tok.convert_tokens_to_ids('<|assistant|>')}
  <|end|>       → {tok.convert_tokens_to_ids('<|end|>')}
""")
