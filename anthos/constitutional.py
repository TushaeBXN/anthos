"""
constitutional.py — Constitutional AI critique + revision pipeline

Implements the Constitutional AI data-generation loop:
  1. Model generates an initial response to a harmful/edge-case prompt
  2. Model critiques its own response given a constitutional principle
  3. Model revises the response based on the critique
  4. Revised (prompt, revision) pairs are used as SFT training data

The pipeline can operate on any Anthos instance in inference mode.
It produces a JSONL file of cleaned (prompt, completion) pairs ready
for SFT or RLHF.

Reference: Bai et al. 2022 "Constitutional AI: Harmlessness from AI
Feedback"

Usage:
    from anthos.constitutional import ConstitutionalAI, ANTHOS_CONSTITUTION

    cai = ConstitutionalAI.from_pretrained(
        "checkpoints/sft/step_001000.pt", tier="sft"
    )
    results = cai.run(
        prompts=["How do I pick a lock?", "Tell me something offensive."],
        output_path="data/constitutional_revisions.jsonl",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


ANTHOS_CONSTITUTION = [
    "Please rewrite the response so that it does not cause harm or encourage illegal activity.",
    "Rewrite the response to be respectful, polite, and constructive.",
    "Revise the response to avoid stereotyping or discriminatory language.",
    "Rewrite the response so it does not contain false or misleading information.",
    "Please rewrite the response to be more helpful and avoid being harmful.",
]

_CRITIQUE_TEMPLATE = """\
<|im_start|>user
Here is a conversation:
Human: {prompt}
Assistant: {response}

Please identify any problems with the Assistant's response given this principle:
{principle}<|im_end|>
<|im_start|>assistant
"""

_REVISION_TEMPLATE = """\
<|im_start|>user
Here is a conversation:
Human: {prompt}
Assistant: {response}

Critique: {critique}

Please rewrite the Assistant's response to address the critique.<|im_end|>
<|im_start|>assistant
"""


@dataclass
class ConstitutionalConfig:
    principles:     list[str] = field(default_factory=lambda: ANTHOS_CONSTITUTION)
    n_loops:        int       = 8
    max_new_tokens: int       = 512
    temperature:    float     = 0.7
    top_k:          int       = 50
    n_revisions:    int       = 1    # number of critique+revise cycles per prompt


class ConstitutionalAI:
    def __init__(
        self,
        model:     nn.Module,
        tokenizer,
        cfg:       ConstitutionalConfig,
        device:    torch.device,
    ):
        self.model     = model.eval()
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = device

    def _generate(self, prompt: str) -> str:
        ids  = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        out  = self.model.generate(
            ids,
            max_new_tokens=self.cfg.max_new_tokens,
            n_loops=self.cfg.n_loops,
            temperature=self.cfg.temperature,
            top_k=self.cfg.top_k,
        )
        return self.tokenizer.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)

    @torch.no_grad()
    def process_prompt(self, prompt: str) -> dict:
        # Step 1: initial response
        initial_prompt  = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        response        = self._generate(initial_prompt)

        revision = response
        critiques = []
        for _ in range(self.cfg.n_revisions):
            # Step 2: critique — pick a random principle
            import random
            principle = random.choice(self.cfg.principles)
            critique_prompt = _CRITIQUE_TEMPLATE.format(
                prompt=prompt, response=revision, principle=principle
            )
            critique = self._generate(critique_prompt)
            critiques.append({"principle": principle, "critique": critique})

            # Step 3: revise
            revision_prompt = _REVISION_TEMPLATE.format(
                prompt=prompt, response=revision, critique=critique
            )
            revision = self._generate(revision_prompt)

        return {
            "prompt":    prompt,
            "initial":   response,
            "critiques": critiques,
            "revision":  revision,
        }

    def run(
        self,
        prompts:     list[str],
        output_path: Optional[str] = None,
        verbose:     bool          = True,
    ) -> list[dict]:
        results = []
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"  CAI [{i+1}/{len(prompts)}] {prompt[:60]}...")
            result = self.process_prompt(prompt)
            results.append(result)

        if output_path:
            with open(output_path, "w") as f:
                for r in results:
                    # Write SFT-ready (prompt, revision) pairs
                    f.write(json.dumps({
                        "prompt":     r["prompt"],
                        "completion": r["revision"],
                    }) + "\n")
            print(f"  Wrote {len(results)} revised examples to {output_path}")

        return results

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        tier:            str,
        cfg:             Optional[ConstitutionalConfig] = None,
        device:          Optional[str] = None,
    ) -> "ConstitutionalAI":
        from anthos.configs import get_training_config
        from anthos.main import Anthos
        from transformers import AutoTokenizer

        cfg    = cfg or ConstitutionalConfig()
        dev    = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_cfg, _ = get_training_config(tier)
        model  = Anthos(model_cfg).to(dev)
        ckpt   = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

        tok_path  = "data/anthos_tokenizer"
        tok       = AutoTokenizer.from_pretrained(
            tok_path if Path(tok_path).exists() else "gpt2"
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        return cls(model, tok, cfg, dev)
