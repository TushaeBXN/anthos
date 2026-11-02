#!/usr/bin/env python3
"""
ingest_crownelius.py — Download and convert Crownelius datasets into Anthos SFT format

Fetches all usable Crownelius datasets, converts to Anthos ShareGPT format,
applies a 900-token filter, deduplicates against sft_master, and appends.

Datasets ingested:
  1.  Crownelius/Qwen-CoT-Library               — 26,898  | Apache-2.0-source
  2.  Crownelius/GLM-5.2-CoT-Library             — 41,333  | CC-BY-NC-4.0-source
  3.  Crownelius/GPT-5.6-Sol-Luna-Terra-Traces   — 15,353  | CC-BY-4.0
  4.  Crownelius/Crow-8B-Training-Data           — 188k    | no explicit license
  5.  Crownelius/Crow-8B-Training-Data-Clean     — 91k     | no explicit license
  6.  Crownelius/Opus-4.6-Reasoning-2100x-formatted — ~2.1k | formatted traces
  7.  Crownelius/Kimi-K3-CoT-Library             — varies  | CoT reasoning
  8.  Crownelius/UltraCHAT-4200x-Qwen3          — varies  | conversational
  9.  Crownelius/Creative-Writing-Reasoning-KimiK2.5-600x — 600 | creative
  10. Crownelius/Complete-FABLE.5-traces-2M      — 2M      | FABLE.5 traces (schema b)

Skipped:
  Human-Archtypes-25k        → character profiles with adult content, wrong format
  High-Coder-Reasoning-Multi-Turn → criteria/transform format, not conversations
  Hyper-UltraData-Grok-V1   → unusual format (stream/content), no license
  Qwen3.5-Creative-Reasoning → preference/reward format, not SFT
  qwen3.7-max-pi-traces-bucket → bucket URL, not a dataset

Usage:
    python ingest_crownelius.py
    python ingest_crownelius.py --dry-run
    python ingest_crownelius.py --out data/crownelius_sft.jsonl
    python ingest_crownelius.py --dataset Crownelius/Crow-8B-Training-Data-Clean

After running:
    python build_training_manifest.py
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry
# Each entry carries:
#   id       — HuggingFace dataset ID
#   handler  — which parse function to call
#   license  — human note
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = [
    {
        "id":      "Crownelius/Qwen-CoT-Library",
        "handler": "messages_row_json",
        "license": "other (Qwen3 Apache-2.0 source outputs)",
    },
    {
        "id":      "Crownelius/GLM-5.2-CoT-Library",
        "handler": "messages_row_json",
        "license": "other (GLM-4 CC BY-NC 4.0 source)",
    },
    {
        "id":      "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces",
        "handler": "messages_row_json",
        "license": "CC-BY-4.0",
    },
    {
        "id":      "Crownelius/Crow-8B-Training-Data",
        "handler": "messages_direct",
        "license": "no explicit license — included per user instruction",
    },
    {
        "id":      "Crownelius/Crow-8B-Training-Data-Clean",
        "handler": "messages_direct",
        "license": "no explicit license — included per user instruction",
    },
    {
        "id":      "Crownelius/Opus-4.6-Reasoning-2100x-formatted",
        "handler": "messages_direct",
        "license": "formatted Opus traces",
    },
    {
        "id":      "Crownelius/Kimi-K3-CoT-Library",
        "handler": "kimi_k3",
        "license": "Kimi-K3 CoT traces",
    },
    {
        "id":      "Crownelius/UltraCHAT-4200x-Qwen3",
        "handler": "ultrachat",
        "license": "UltraChat Qwen3 variant",
    },
    {
        "id":      "Crownelius/Creative-Writing-Reasoning-KimiK2.5-600x",
        "handler": "creative_writing",
        "license": "KimiK2.5 creative writing traces",
    },
    {
        "id":      "Crownelius/Complete-FABLE.5-traces-2M",
        "handler": "fable5",
        "license": "FABLE.5 Claude Code execution traces",
    },
]

MAX_TOKENS = 900    # chars / 4 estimate  (matches build_training_manifest.py)
MIN_ANSWER = 20


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(text) // 4


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="replace")).hexdigest()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def make_example(human: str, gpt: str, source: str = "") -> dict | None:
    """Build a ShareGPT example, applying length filters.  Returns None to skip."""
    human = human.strip()
    gpt   = gpt.strip()
    if not human or len(gpt) < MIN_ANSWER:
        return None
    if count_tokens(human + gpt) > MAX_TOKENS:
        return None
    obj = {"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt",   "value": gpt},
    ]}
    if source:
        obj["source"] = source
    return obj


def make_example_with_system(system: str, human: str, gpt: str, source: str = "") -> dict | None:
    """Build a ShareGPT example with an optional system turn."""
    human = human.strip()
    gpt   = gpt.strip()
    system = (system or "").strip()
    if not human or len(gpt) < MIN_ANSWER:
        return None
    total_chars = len(system) + len(human) + len(gpt)
    if count_tokens(total_chars * "x") > MAX_TOKENS:  # rough upper bound
        return None
    if count_tokens(system + human + gpt) > MAX_TOKENS:
        return None
    convs = []
    if system:
        convs.append({"from": "system", "value": system})
    convs.append({"from": "human", "value": human})
    convs.append({"from": "gpt",   "value": gpt})
    obj = {"conversations": convs}
    if source:
        obj["source"] = source
    return obj


def openai_messages_to_sharegpt(messages: list, source: str = "") -> dict | None:
    """Convert OpenAI messages list → Anthos ShareGPT.  Returns None to skip."""
    if not messages:
        return None

    conversations = []
    system_val = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role    = msg.get("role", "")
        content = msg.get("content")
        # Handle list-type content (multi-part OpenAI format)
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        content = str(content or "").strip()
        if not content:
            # Skip tool_calls and empty turns
            continue

        if role == "system":
            system_val = content
        elif role in ("user", "human"):
            conversations.append({"from": "human", "value": content})
        elif role in ("assistant", "gpt"):
            # Strip leading <think>...</think> block if present
            content = _strip_think_tags(content)
            if content:
                conversations.append({"from": "gpt", "value": content})

    human_turns = [c for c in conversations if c["from"] == "human"]
    gpt_turns   = [c for c in conversations if c["from"] == "gpt"]
    if not human_turns or not gpt_turns:
        return None

    answer = gpt_turns[0]["value"]
    if len(answer) < MIN_ANSWER:
        return None

    full = []
    if system_val:
        full.append({"from": "system", "value": system_val})
    full.extend(conversations)

    if count_tokens("".join(c["value"] for c in full)) > MAX_TOKENS:
        return None

    obj = {"conversations": full}
    if source:
        obj["source"] = source
    return obj


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> prefix and return the remainder."""
    m = re.match(r"^\s*<think>.*?</think>\s*", text, re.DOTALL)
    if m:
        return text[m.end():].strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset parse functions
#   Each takes a raw dict row, returns a ShareGPT dict or None.
# ─────────────────────────────────────────────────────────────────────────────

def parse_messages_row_json(raw: dict, source: str) -> dict | None:
    """rows that store data in row_json → messages or conversations field."""
    rj = raw.get("row_json")
    if rj:
        try:
            obj = json.loads(rj) if isinstance(rj, str) else rj
        except json.JSONDecodeError:
            return None
    else:
        obj = raw

    messages = obj.get("messages", [])
    if not messages:
        messages = obj.get("conversations", [])
        if messages:
            messages = [
                {"role": m.get("from", m.get("role", "")),
                 "content": m.get("value", m.get("content", ""))}
                for m in messages
            ]
    return openai_messages_to_sharegpt(messages, source=source)


def parse_messages_direct(raw: dict, source: str) -> dict | None:
    """rows where messages/conversations is directly on the row dict."""
    messages = raw.get("messages", [])
    if not messages:
        messages = raw.get("conversations", [])
        if messages:
            messages = [
                {"role": m.get("from", m.get("role", "")),
                 "content": m.get("value", m.get("content", ""))}
                for m in messages
            ]
    return openai_messages_to_sharegpt(messages, source=source)


def parse_kimi_k3(raw: dict, source: str) -> dict | None:
    """Kimi-K3-CoT-Library: row_json with messages + assistant_step."""
    rj = raw.get("row_json")
    if rj:
        try:
            obj = json.loads(rj) if isinstance(rj, str) else rj
        except json.JSONDecodeError:
            return None
    else:
        obj = raw

    messages = obj.get("messages", [])
    if not messages:
        return None
    return openai_messages_to_sharegpt(messages, source=source)


def parse_ultrachat(raw: dict, source: str) -> dict | None:
    """UltraCHAT-4200x-Qwen3: data field is alternating [user, assistant, ...] strings."""
    data = raw.get("data", [])
    if not isinstance(data, list) or len(data) < 2:
        return None
    # Build messages by alternating user/assistant
    messages = []
    for i, text in enumerate(data):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": str(text or "").strip()})
    return openai_messages_to_sharegpt(messages, source=source)


def parse_creative_writing(raw: dict, source: str) -> dict | None:
    """Creative-Writing-Reasoning-KimiK2.5-600x: instruction + thought + response."""
    instruction = str(raw.get("instruction") or "").strip()
    response    = str(raw.get("response")    or "").strip()
    if not instruction or not response:
        return None
    return make_example(instruction, response, source=source)


def _parse_fable5_context(context_str: str) -> list[dict]:
    """
    Parse a FABLE.5 context string into a list of {role, text} turns.

    Context format:
        USER: <question>
        ASSISTANT: <answer>
        USER: <local-command-caveat>...
        ...

    Returns turns in order, skipping <local-command-caveat> USER entries.
    """
    # Split on turn boundaries
    parts = re.split(r'\n(?=(?:USER|ASSISTANT): )', context_str.strip())
    turns = []
    for part in parts:
        part = part.strip()
        if part.startswith("USER: "):
            text = part[len("USER: "):].strip()
            # Skip synthetic caveat messages injected by the eval harness
            if text.startswith("<local-command-caveat>"):
                continue
            if text:
                turns.append({"role": "user", "text": text})
        elif part.startswith("ASSISTANT: "):
            text = part[len("ASSISTANT: "):].strip()
            if text:
                turns.append({"role": "assistant", "text": text})
    return turns


def parse_fable5(raw: dict, source: str) -> dict | None:
    """
    Complete-FABLE.5-traces-2M — two schemas:
      (a) event log: {type, userType, sessionId, uuid, ...} — skip (need full session)
      (b) distilled Q&A: {completion, cot, context, output, output_type, model, ...}

    For schema (b):
      - output_type == "text": take last user turn from context + output['text'] as answer
      - output_type == "tool_use": take last user turn from context + cot as answer
        (cot is the raw reasoning, which teaches thinking without depending on tool calls)
    """
    rj = raw.get("row_json")
    if rj:
        try:
            obj = json.loads(rj) if isinstance(rj, str) else rj
        except json.JSONDecodeError:
            return None
    else:
        obj = raw

    # Detect schema (a) — event log, skip it
    if "type" in obj and "sessionId" in obj and "uuid" in obj:
        return None

    # Schema (b)
    output_type = obj.get("output_type", "")
    context_str = str(obj.get("context") or "")
    cot         = str(obj.get("cot")     or "").strip()
    output      = obj.get("output")
    completion  = str(obj.get("completion") or "").strip()

    if not context_str:
        return None

    # Extract last human turn from context
    turns = _parse_fable5_context(context_str)
    user_turns = [t for t in turns if t["role"] == "user"]
    if not user_turns:
        return None
    human = user_turns[-1]["text"]

    # Determine gpt response
    if output_type == "text":
        # output is a dict-like string: {'text': '...'}
        if isinstance(output, dict):
            gpt = str(output.get("text", "") or "").strip()
        elif isinstance(output, str):
            m = re.search(r"'text'\s*:\s*'(.*?)'", output, re.DOTALL)
            if m:
                gpt = m.group(1).strip()
            else:
                # Try JSON-style double quotes
                m = re.search(r'"text"\s*:\s*"(.*?)"', output, re.DOTALL)
                gpt = m.group(1).strip() if m else ""
        else:
            gpt = ""
        if not gpt and completion:
            # Fall back to completion (strip <think> block)
            gpt = _strip_think_tags(completion)
    elif output_type == "tool_use":
        # Use the reasoning (cot) as the assistant response
        gpt = cot
        if not gpt and completion:
            gpt = _strip_think_tags(completion)
    else:
        # Unknown output_type — try cot then completion
        gpt = cot or _strip_think_tags(completion)

    return make_example(human, gpt, source=source)


# Map handler name → function
HANDLER_MAP = {
    "messages_row_json": parse_messages_row_json,
    "messages_direct":   parse_messages_direct,
    "kimi_k3":           parse_kimi_k3,
    "ultrachat":         parse_ultrachat,
    "creative_writing":  parse_creative_writing,
    "fable5":            parse_fable5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Load existing dedup fingerprints
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_fingerprints(master_path: str) -> tuple[set, set]:
    """Returns (exact_hashes, near_hashes) from master file."""
    exact_seen = set()
    near_seen  = set()
    p = Path(master_path)
    if not p.exists():
        return exact_seen, near_seen

    print(f"  Loading dedup fingerprints from {master_path}...")
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i % 500_000 == 0 and i > 0:
                print(f"    ... {i:,} lines scanned")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            convs = obj.get("conversations", [])
            human = [c for c in convs if c.get("from") in ("human", "user")]
            gpt   = [c for c in convs if c.get("from") in ("gpt", "assistant")]
            if not human or not gpt:
                continue
            q = str(human[0].get("value", "")).strip()
            a = str(gpt[0].get("value",  "")).strip()
            exact_seen.add(md5(q + a))
            near_seen.add(normalize(q)[:120])
    print(f"  Loaded {len(exact_seen):,} fingerprints.")
    return exact_seen, near_seen


# ─────────────────────────────────────────────────────────────────────────────
# Stream one dataset from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────

def stream_dataset(dataset_id: str):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        return

    ds = load_dataset(dataset_id, split="train", streaming=True)
    for row in ds:
        yield row


# ─────────────────────────────────────────────────────────────────────────────
# Dedup helper
# ─────────────────────────────────────────────────────────────────────────────

def is_duplicate(obj: dict, exact_seen: set, near_seen: set) -> bool:
    """Returns True if example is a duplicate; updates sets if new."""
    convs = obj.get("conversations", [])
    human = [c for c in convs if c.get("from") in ("human", "user")]
    gpt   = [c for c in convs if c.get("from") in ("gpt", "assistant")]
    if not human or not gpt:
        return True
    q = str(human[0].get("value", "")).strip()
    a = str(gpt[0].get("value",  "")).strip()
    h = md5(q + a)
    if h in exact_seen:
        return True
    exact_seen.add(h)
    nk = normalize(q)[:120]
    if nk in near_seen:
        return True
    near_seen.add(nk)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master",   default="data/sft_master.jsonl")
    parser.add_argument("--out",      default=None)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--dataset",  default=None,
                        help="Process only this dataset ID (e.g. Crownelius/Crow-8B-Training-Data-Clean)")
    args = parser.parse_args()

    master_path = args.master
    out_path    = args.out or master_path
    dry_run     = args.dry_run

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["id"] == args.dataset]
        if not datasets:
            print(f"ERROR: dataset '{args.dataset}' not in registry")
            sys.exit(1)

    print(f"\nCrownelius dataset ingestion")
    print(f"  Master  : {master_path}")
    print(f"  Output  : {out_path}  {'(dry run)' if dry_run else ''}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Filter  : max {MAX_TOKENS} tokens, min answer {MIN_ANSWER} chars")
    print()

    exact_seen, near_seen = load_existing_fingerprints(master_path)
    print()

    total_written = 0
    all_new = []

    for info in datasets:
        ds_id   = info["id"]
        handler = HANDLER_MAP[info["handler"]]
        print(f"── {ds_id}")
        print(f"   License : {info['license']}")

        n_read = n_converted = n_duped = n_skipped = n_written = 0

        for raw in stream_dataset(ds_id):
            n_read += 1
            if n_read % 10_000 == 0:
                print(f"   ... {n_read:,} read | kept={n_written:,}", flush=True)

            obj = handler(raw, source=ds_id)
            if obj is None:
                n_skipped += 1
                continue
            n_converted += 1

            if is_duplicate(obj, exact_seen, near_seen):
                n_duped += 1
                continue

            n_written += 1
            all_new.append(json.dumps(obj, ensure_ascii=False))

        print(f"   Read={n_read:,}  converted={n_converted:,}  "
              f"skipped={n_skipped:,}  duped={n_duped:,}  NEW={n_written:,}")
        total_written += n_written
        print()

    print(f"{'═'*60}")
    print(f"  Total new examples : {total_written:,}")

    if dry_run:
        print("  DRY RUN — nothing written.")
        return

    if total_written == 0:
        print("  Nothing to write.")
        return

    mode = "a" if out_path == master_path else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for line in all_new:
            f.write(line + "\n")

    print(f"  Written {total_written:,} examples → {out_path}")
    if out_path == master_path:
        print(f"  sft_master.jsonl updated (append-only).")
    print()
    print(f"  Next: python build_training_manifest.py")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
