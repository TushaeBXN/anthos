#!/usr/bin/env python3
"""
anthos_refine_code.py — Multi-pass code generation using chat_anthos as backbone.

This is the real version of "iterative refinement" — three generate() calls in sequence.
It works with your existing Qwen LoRA or any Ollama model. No fake APIs needed.

Usage:
    python3 anthos_refine_code.py --prompt "Write a Python LRU cache"
    python3 anthos_refine_code.py --prompt "Write a Go HTTP server" --passes 2
"""

import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH  = "checkpoints/anthos-qwen-lora/final"

SYSTEM = (
    "You are Anthos, a coding AI created by Brian Tushae Thomas. "
    "Write clean, complete, well-documented code. Include all imports. "
    "Handle edge cases. Show example usage. No filler. No flattery."
)


def load_model():
    print("Loading Anthos...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, messages: list, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_code(text: str) -> str:
    """Pull the first code block out of a response."""
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def generate_with_refinement(model, tokenizer, prompt: str, passes: int = 3) -> str:
    """
    Pass 1: Generate initial code.
    Pass 2: Add error handling and edge cases.
    Pass 3: Add documentation and example usage.
    """
    sys_msg = {"role": "system", "content": SYSTEM}

    # Pass 1 — initial implementation
    print("\n[Pass 1] Generating initial code...")
    p1 = _generate(model, tokenizer, [
        sys_msg,
        {"role": "user", "content": prompt},
    ])
    code = extract_code(p1)
    print(f"  {len(code.splitlines())} lines generated.")

    if passes < 2:
        return p1

    # Pass 2 — add error handling
    print("[Pass 2] Adding error handling and edge cases...")
    p2 = _generate(model, tokenizer, [
        sys_msg,
        {"role": "user", "content": (
            f"Improve this code by adding proper error handling, input validation, "
            f"and handling of edge cases. Return only the improved code.\n\n```\n{code}\n```"
        )},
    ])
    code = extract_code(p2)
    print(f"  {len(code.splitlines())} lines after refinement.")

    if passes < 3:
        return p2

    # Pass 3 — documentation and example
    print("[Pass 3] Adding documentation and example usage...")
    p3 = _generate(model, tokenizer, [
        sys_msg,
        {"role": "user", "content": (
            f"Add a docstring explaining parameters and return values, "
            f"and append a working example usage at the bottom. "
            f"Return only the final code.\n\n```\n{code}\n```"
        )},
    ])
    return p3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--passes", type=int, default=3, choices=[1, 2, 3],
                        help="1=initial only, 2=+error handling, 3=+docs")
    args = parser.parse_args()

    model, tokenizer = load_model()
    result = generate_with_refinement(model, tokenizer, args.prompt, args.passes)

    print("\n" + "─" * 60)
    print("  Final output")
    print("─" * 60)
    print(result)


if __name__ == "__main__":
    main()
