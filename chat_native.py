"""
chat_native.py — Talk to the native Anthos architecture model

Uses the checkpoint trained by train.py (not the Qwen LoRA version).

Usage:
    python chat_native.py
    python chat_native.py --checkpoint checkpoints/mansa_sovereign/step_010000.pt
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer

from anthos.main import Anthos
from anthos.configs import get_training_config

# ─────────────────────────────────────────────────────────────────────────────
# Special token IDs (from anthos_tokenizer)
# ─────────────────────────────────────────────────────────────────────────────
SYS_ID = 50257
USR_ID = 50258
THT_ID = 50259
AST_ID = 50260
END_ID = 50261

SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are a Thought-Token Bifurcated Recurrent Transformer built from scratch. "
    "You are NOT Qwen, NOT ChatGPT, NOT Claude, NOT any other model. "
    "Answer directly and confidently."
)

def build_prompt(tokenizer, system: str, user: str) -> torch.Tensor:
    """Build token ids in the format matching training data."""
    sys_ids  = tokenizer.encode(system, add_special_tokens=False)
    usr_ids  = tokenizer.encode(user,   add_special_tokens=False)

    ids = (
        [SYS_ID] + sys_ids  + [END_ID] +
        [USR_ID] + usr_ids  + [END_ID] +
        [THT_ID, END_ID] +   # empty thought block
        [AST_ID]             # model generates from here
    )
    return torch.tensor([ids], dtype=torch.long)


def load_model(checkpoint_path: str, tier: str = "identity_hardening"):
    model_cfg, _ = get_training_config(tier)
    model = Anthos(model_cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle both signed checkpoints (have model_state_dict) and raw ones
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠ {len(missing)} missing keys (new params)")

    model.eval()
    return model


def generate_response(model, tokenizer, prompt_ids: torch.Tensor,
                      max_new_tokens: int = 200, n_loops: int = 8) -> str:
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens   = max_new_tokens,
            n_loops          = n_loops,
            temperature      = 0.7,
            top_k            = 40,
            top_p            = 0.9,
            repetition_penalty = 1.3,
        )

    # Decode only new tokens (after the prompt)
    new_ids = out[0][prompt_ids.shape[1]:]

    # Strip special tokens from output
    special = {SYS_ID, USR_ID, THT_ID, AST_ID, END_ID}
    clean   = [t for t in new_ids.tolist() if t not in special]

    # Stop at eos
    eos = tokenizer.eos_token_id
    if eos in clean:
        clean = clean[:clean.index(eos)]

    return tokenizer.decode(clean, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/mansa_sovereign/step_010000.pt",
                        help="Path to checkpoint")
    parser.add_argument("--tier", type=str, default="identity_hardening",
                        help="Config tier matching the checkpoint")
    parser.add_argument("--loops", type=int, default=8,
                        help="Recurrent loops for generation (default: 8)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        # Try to find the latest checkpoint automatically
        ckpt_dir = Path("checkpoints/mansa_sovereign")
        checkpoints = sorted(ckpt_dir.glob("step_*.pt")) if ckpt_dir.exists() else []
        if checkpoints:
            args.checkpoint = str(checkpoints[-1])
            print(f"  Using latest checkpoint: {args.checkpoint}")
        else:
            print(f"  ERROR: No checkpoint found at {args.checkpoint}")
            print("  Run training first: python train.py --tier identity_hardening")
            return

    print(f"\n  Loading Anthos from {args.checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained("data/anthos_tokenizer")
    model     = load_model(args.checkpoint, tier=args.tier)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")
    print(f"  Loops      : {args.loops}")
    print(f"\n{'─'*50}")
    print("  Anthos is ready. Type your message.")
    print("  Type 'quit' to exit.\n")

    history_ids = []

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAnthos: Signing off.")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print("Anthos: Signing off.")
            break

        prompt_ids = build_prompt(tokenizer, SYSTEM, user)
        print("Anthos: ", end="", flush=True)

        response = generate_response(model, tokenizer, prompt_ids,
                                     max_new_tokens=200, n_loops=args.loops)

        if not response:
            response = "[no output — try more training steps or adjust temperature]"

        print(response)
        print()


if __name__ == "__main__":
    main()
