"""
tutor_loop.py — Context Engineering Loop for the Anthos Math Tutor

Entry point for tutoring sessions. Implements the full context engineering loop:

  SELECT LAYER  → load student profile + last 10 learnings + hard rules at session start
  VERIFIER GATE → check every response for answer giveaway / scaffolding / ends-with-?
  WRITE LAYER   → append structured learning entry to learnings.md after session ends
  COMPRESS LAYER→ summarize oldest 25 entries when a profile exceeds 50 raw entries
  ESCALATION    → promote failure type to hard rule when it hits 5+ occurrences
  FEEDBACK LOOP → generate_hard_math_dataset.py consumes eval_log.md after sessions

Usage:
    python tutor_loop.py --profile direct_instruction --domain algebra
    python tutor_loop.py --profile inquiry_based --domain fractions
    python tutor_loop.py --list-profiles
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime, timezone

from profiles.student_profiles import get_profile, PROFILES
from context_engineering.learnings_manager import (
    load_learnings,
    format_learnings_for_context,
    append_learning,
    compress_if_needed,
)
from context_engineering.verifier import (
    run_verifier,
    build_regeneration_prompt,
    log_failures,
)
from context_engineering.escalation import check_and_escalate

# ─────────────────────────────────────────────────────────────────────────────
# Model config (mirrors chat_anthos.py)
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH  = "checkpoints/anthos-qwen-lora/final"

MAX_NEW_TOKENS = 300
MAX_REGEN_ATTEMPTS = 2

_TUTOR_BASE_SYSTEM = (
    "You are Anthos, an expert math tutor. Your job is to guide students to discover "
    "answers themselves — never to give the answer away.\n\n"
    "Core tutoring rules (non-negotiable):\n"
    "  1. Never state the final numerical answer directly.\n"
    "  2. Always include at least one guiding question in every response.\n"
    "  3. Every response must end with a question mark.\n"
    "  4. Use scaffolding: break the problem into smaller steps.\n"
    "  5. Praise effort, not speed.\n\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> tuple:
    """Load the LoRA-fine-tuned Anthos model. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print("Loading Anthos tutor model...\n")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# System prompt builder (SELECT LAYER)
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(profile_id: str, domain: str) -> str:
    """
    SELECT LAYER: assemble the tutor system prompt from:
      - base tutoring rules
      - student profile definition + hard rules
      - last 10 learnings for this specific profile (no other profiles' data)
    """
    profile = get_profile(profile_id)
    learnings = load_learnings(profile_id, n=10)
    learnings_ctx = format_learnings_for_context(learnings)

    parts = [
        _TUTOR_BASE_SYSTEM,
        profile.to_system_context(),
        f"\n[CURRENT SESSION]\nMath domain: {domain}\n",
    ]

    if learnings_ctx:
        parts.append(f"\n{learnings_ctx}\n")

    parts.append(
        "\nApply the learnings above to calibrate your approach for this session. "
        "If a note says 'try differently', act on it."
    )

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Response generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate(model, tokenizer, system: str, history: list[dict], user_input: str) -> str:
    import torch

    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_verified_response(
    model,
    tokenizer,
    system: str,
    history: list[dict],
    user_input: str,
    profile_id: str,
    domain: str,
) -> tuple[str, list]:
    """
    VERIFIER GATE: generate a response, run 3 checks, regenerate up to
    MAX_REGEN_ATTEMPTS times if checks fail. Log all failures to eval_log.md.

    Returns (final_response, all_failures_across_attempts).
    """
    all_failures = []

    for attempt in range(MAX_REGEN_ATTEMPTS + 1):
        if attempt == 0:
            response = _generate(model, tokenizer, system, history, user_input)
        else:
            # Inject failure context into a regeneration request
            regen_prompt = build_regeneration_prompt(all_failures[-1])
            extended_user = f"{user_input}\n\n{regen_prompt}"
            response = _generate(model, tokenizer, system, history, extended_user)

        failures = run_verifier(response)

        if not failures:
            # All checks passed
            if all_failures:
                print(f"  [VERIFIER] Pass on attempt {attempt + 1}.")
            return response, [f for batch in all_failures for f in batch]

        # Log each failed attempt
        log_failures(profile_id, domain, failures, response)
        all_failures.append(failures)

        if attempt < MAX_REGEN_ATTEMPTS:
            fail_names = ", ".join(f.check_name for f in failures)
            print(f"  [VERIFIER] Attempt {attempt + 1} failed ({fail_names}) — regenerating...")

    # Exhausted retries — use last response, failures already logged
    flat_failures = [f for batch in all_failures for f in batch]
    fail_names = ", ".join(f.check_name for f in flat_failures)
    print(f"  [VERIFIER] All {MAX_REGEN_ATTEMPTS + 1} attempts failed ({fail_names}). Using last response.")
    return response, flat_failures


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold approach detection (for learnings entry)
# ─────────────────────────────────────────────────────────────────────────────

def infer_scaffold(responses: list[str]) -> str:
    """Heuristically infer the dominant scaffolding approach from session responses."""
    text = " ".join(responses).lower()
    if any(kw in text for kw in ("step 1", "step 2", "first,", "next,", "then,")):
        return "stepped_decomposition"
    if any(kw in text for kw in ("imagine", "story", "character", "quest", "game")):
        return "narrative_framing"
    if any(kw in text for kw in ("real world", "in real life", "for example in", "think about when")):
        return "real_world_anchor"
    if any(kw in text for kw in ("what do you notice", "what pattern", "what do you wonder")):
        return "guided_discovery"
    return "socratic_questioning"


# ─────────────────────────────────────────────────────────────────────────────
# Session entry note generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_session_note(
    profile_id: str,
    responses: list[str],
    failures: list,
    user_inputs: list[str],
) -> str:
    """Produce a max-120-char note summarizing what to try differently next time."""
    if not failures:
        if len(responses) > 2:
            return "Session completed cleanly; student engaged across multiple turns."
        return "Short session; increase scaffolding depth next time."

    fail_names = list({f.check_name for f in failures})
    if "answer_giveaway" in fail_names:
        return "Answer giveaway occurred; next session: stop one step earlier and ask student to complete."
    if "scaffolding_quality" in fail_names:
        return "Scaffolding was thin; next session: open each response with a reflective question."
    if "ends_with_question" in fail_names:
        return "Responses closed without questions; next session: draft the closing question first."
    return f"Failures on {', '.join(fail_names)}; review eval_log.md for details."


# ─────────────────────────────────────────────────────────────────────────────
# Main session loop
# ─────────────────────────────────────────────────────────────────────────────

def run_session(profile_id: str, domain: str, model, tokenizer) -> None:
    profile = get_profile(profile_id)

    print(f"\n{'─' * 60}")
    print(f"  Anthos Tutor — {profile.name}")
    print(f"  Domain: {domain}")
    print(f"  Type 'done' or 'quit' to end the session.")
    print(f"{'─' * 60}\n")

    system = build_system_prompt(profile_id, domain)

    history: list[dict] = []
    session_responses: list[str] = []
    session_inputs: list[str] = []
    all_session_failures: list = []
    answer_given = False

    try:
        while True:
            user_input = input("Student: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("done", "quit", "exit", "q"):
                break

            response, failures = generate_verified_response(
                model, tokenizer, system, history, user_input, profile_id, domain
            )

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

            session_responses.append(response)
            session_inputs.append(user_input)
            all_session_failures.extend(failures)

            if any(f.check_name == "answer_giveaway" for f in failures):
                answer_given = True

            print(f"\nTutor: {textwrap.fill(response, width=72, subsequent_indent='       ')}\n")

    except KeyboardInterrupt:
        print("\n\n[Session interrupted by user]")

    # ── WRITE LAYER ────────────────────────────────────────────────────────
    if session_responses:
        ended_with_q = session_responses[-1].rstrip().endswith("?")
        scaffold = infer_scaffold(session_responses)
        note = _build_session_note(profile_id, session_responses, all_session_failures, session_inputs)

        append_learning(
            profile_id=profile_id,
            domain=domain,
            scaffold=scaffold,
            ended_with_question=ended_with_q,
            answer_given=answer_given,
            note=note,
        )
        print(f"\n[LEARNINGS] Session entry written to learnings.md")

    # ── COMPRESS LAYER ─────────────────────────────────────────────────────
    if compress_if_needed(profile_id):
        print(f"[COMPRESS] Oldest 25 entries for '{profile_id}' archived in learnings.md")

    # ── ESCALATION ─────────────────────────────────────────────────────────
    escalated = check_and_escalate(profile_id)
    if escalated:
        for pid, ftype in escalated:
            print(f"[ESCALATION] Hard rule added for '{pid}' ({ftype})")

    print(f"\n[Session complete. Run generate_hard_math_dataset.py to update training data.]\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Anthos math tutoring session with context engineering loop")
    parser.add_argument("--profile", "-p", default="direct_instruction",
                        help="Student profile ID (use --list-profiles to see options)")
    parser.add_argument("--domain", "-d", default="algebra",
                        help="Math domain for this session (e.g. algebra, fractions, geometry)")
    parser.add_argument("--list-profiles", action="store_true",
                        help="Print all available student profiles and exit")
    parser.add_argument("--skip-model", action="store_true",
                        help="Run without loading the model (for testing infrastructure only)")
    args = parser.parse_args()

    if args.list_profiles:
        print("\nAvailable student profiles:\n")
        for pid, profile in PROFILES.items():
            print(f"  {pid}")
            print(f"    {profile.description[:80]}...")
            hard_rules = profile.all_hard_rules()
            if hard_rules:
                print(f"    Hard rules: {len(hard_rules)} (including {len(profile.load_escalated_rules())} escalated)")
            print()
        return

    if args.skip_model:
        print("[INFO] --skip-model: running with stub model for infrastructure testing.")

        import torch as _torch

        class _StubTokenizer:
            eos_token_id = 0
            def apply_chat_template(self, *a, **kw): return ""
            def __call__(self, *a, **kw):
                class _R:
                    input_ids = _torch.zeros(1, 5, dtype=_torch.long)
                return _R()
            def decode(self, *a, **kw): return "What do you think the next step is?"

        class _StubModel:
            def eval(self): return self
            def generate(self, **kw):
                return _torch.zeros(1, 10, dtype=_torch.long)

        model, tokenizer = _StubModel(), _StubTokenizer()
    else:
        model, tokenizer = load_model()

    run_session(profile_id=args.profile, domain=args.domain, model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
