"""
generate_regular.py — Inference for the Regular pretrained checkpoint.

    # Single prompt (completion mode)
    python generate_regular.py --prompt "Once upon a time"

    # Interactive REPL
    python generate_regular.py

    # Specific checkpoint
    python generate_regular.py --checkpoint checkpoints/regular-ts/step_001000.pt
"""
import argparse
import sys
import torch
import tiktoken
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from regular import Regular, RegularConfig


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = RegularConfig(**cfg_dict) if cfg_dict else RegularConfig()
    model = Regular(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded step {step} | {model.num_params():,} params | loss floor ~1.5 on TinyStories")
    return model, cfg


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new: int, temperature: float,
             top_k: int, device: torch.device) -> list[int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        ctx = ids[:, -model.cfg.max_seq_len:]
        logits = model(ctx)                         # (1, T, vocab)
        logits = logits[0, -1, :] / temperature     # last position
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == 100257:                       # <|endoftext|>
            break
    return ids[0].tolist()


def latest_checkpoint(run: str) -> str | None:
    ckpts = sorted(Path("checkpoints").glob(f"{run}/step_*.pt"))
    return str(ckpts[-1]) if ckpts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--run", default="regular-ts")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--instruct", action="store_true",
                    help="Wrap prompt in SFT instruction format (### Instruction / ### Response)")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available()
         else "mps" if torch.backends.mps.is_available()
         else "cpu") if a.device == "auto" else a.device
    )

    ckpt_path = a.checkpoint or latest_checkpoint(a.run)
    if not ckpt_path:
        sys.exit(f"No checkpoint found in checkpoints/{a.run}/  — training still running?")

    model, cfg = load_model(ckpt_path, device)
    tok = tiktoken.get_encoding("cl100k_base")
    eos = 100257

    def run_prompt(text: str):
        if a.instruct:
            text = f"### Instruction:\n{text}\n\n### Response:\n"
        ids = tok.encode_ordinary(text)
        out = generate(model, ids, a.max_new, a.temperature, a.top_k, device)
        result = tok.decode([t for t in out if t < eos])
        if a.instruct:
            # Print only the response part
            marker = "### Response:\n"
            idx = result.find(marker)
            result = result[idx + len(marker):] if idx != -1 else result
        print(result)
        print()

    if a.prompt:
        run_prompt(a.prompt)
        return

    # Interactive REPL
    print("Regular model — completion mode. Ctrl-C to quit.\n")
    while True:
        try:
            prompt = input(">>> ")
        except (KeyboardInterrupt, EOFError):
            break
        if prompt.strip():
            run_prompt(prompt)


if __name__ == "__main__":
    main()
