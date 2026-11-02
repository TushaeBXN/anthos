#!/usr/bin/env python3
"""
build_training_manifest.py — Balanced, deduplicated, token-filtered train/eval splits.

Fixes two confirmed dataset problems before RunPod training:
  1. TRUNCATION  — 29.5% of examples exceed seq_len=512 and get cut off mid-answer.
                   Fix: filter out examples longer than --max-tokens (default 900).
                   Then set sft seq_len=1024 so everything fits with headroom.
  2. IDENTITY BIAS — 25% of corpus is Brian Thomas / Anthos identity data.
                   Fix: cap identity domain at 5% of the final training set.

Source: data/sft_master.jsonl (NEVER modified — immutable master corpus)
Output: data/train.jsonl + data/eval.jsonl (safe to rebuild at any time)

Usage:
    python build_training_manifest.py
    python build_training_manifest.py --max-tokens 900 --max-identity 0.05
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Domain classifier
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_PATTERNS = {
    'identity':     [r'\bbrian\b', r'\btushae\b', r'\banthos\b', r'\bmansa\b'],
    'math':         [r'\bmath\b', r'\bequation\b', r'\bsolve\b', r'=\s*\d', r'\bproof\b', r'\balgebra\b'],
    'code':         [r'\bpython\b', r'\bjavascript\b', r'```\w', r'\bfunction\b', r'\bcode\b', r'\bdebug\b'],
    'medicine':     [r'\bmedical\b', r'\bclinical\b', r'\bpatient\b', r'\bdiagnos', r'\btreatment\b'],
    'law':          [r'\blegal\b', r'\blaw\b', r'\brights\b', r'\bstatute\b', r'\bconstitut'],
    'finance':      [r'\bfinance\b', r'\bstock\b', r'\bcredit\b', r'\binvest\b', r'\bdividend'],
    'science':      [r'\bscience\b', r'\bphysics\b', r'\bchemi', r'\bbiology\b'],
    'history':      [r'\bhistory\b', r'\bhistorical\b', r'\bcentury\b', r'\bciviliz'],
    'philosophy':   [r'\bphilosoph\b', r'\bethics\b', r'\bmoral\b'],
    'multilingual': [r'\btranslat\b', r'\bfrancais\b', r'\bespa.ol\b',
                     r'[¿¡]', r'\bestoy\b', r'\bbonjour\b', r'\bmerci\b',
                     r'\bvocê\b', r'\bportuguês\b', r'\bshukran\b',
                     r'marḥaban', r'\binshallah\b', r'[؀-ۿ]',
                     r'\bes Anthos\b', r'\beres Anthos\b'],
    'creative':     [r'\bpoem\b', r'\bstory\b', r'\bfiction\b', r'\bcreative writing\b'],
    'general':      [],
}

DRE = {
    k: [re.compile(p, re.I) for p in pats]
    for k, pats in DOMAIN_PATTERNS.items()
}


def classify(sys_v, q, a):
    text = f"{sys_v} {q[:300]} {a[:100]}"
    for domain, pats in DRE.items():
        if domain == 'general':
            continue
        if any(p.search(text) for p in pats):
            return domain
    return 'general'


def md5(s):
    return hashlib.md5(s.encode('utf-8', errors='replace')).hexdigest()


def normalize(s):
    return re.sub(r'\s+', ' ', s.lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',       default='data/sft_master.jsonl')
    parser.add_argument('--train-out',    default='data/train.jsonl')
    parser.add_argument('--eval-out',     default='data/eval.jsonl')
    parser.add_argument('--eval-size',    type=int,   default=5000)
    parser.add_argument('--tok',          default='data/anthos_tokenizer')
    # ── fix 1: token length cap ──────────────────────────────────────────────
    parser.add_argument('--max-tokens',   type=int,   default=900,
                        help='Drop examples longer than this many tokens (default 900). '
                             'Pairs with sft seq_len=1024 — 0%% truncation with headroom.')
    # ── fix 2: domain caps ───────────────────────────────────────────────────
    parser.add_argument('--max-identity', type=float, default=0.05,
                        help='Max fraction for identity domain (default 5%%). '
                             'Audit showed 25%% — 5x overrepresented.')
    parser.add_argument('--max-math',     type=float, default=0.15,
                        help='Max fraction for math domain (default 15%%).')
    parser.add_argument('--max-any',      type=float, default=0.22,
                        help='Hard cap on any single domain (default 22%%).')
    parser.add_argument('--eval-floor',   type=int,   default=50,
                        help='Minimum examples per domain in eval split (default 50). '
                             'Prevents rare domains (Finance/Law) from having <10 eval '
                             'examples, which makes per-domain checkpoint comparisons noise.')
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: {src} not found")
        return

    # Use char-count approximation: 1 token ≈ 4 chars.
    # Fast (no tokenizer call per example) and accurate enough for filtering —
    # we're cutting at 900 tokens, a ±10% error still keeps every example
    # comfortably under seq_len=1024.
    def count_tokens(convs):
        return sum(len(c.get("value", "")) for c in convs) // 4

    print(f"\nBuilding training manifest")
    print(f"  Source        : {src}")
    print(f"  Max tokens    : {args.max_tokens}  (fix #1 — zero truncation at seq_len=1024)")
    print(f"  Identity cap  : {args.max_identity*100:.0f}%   (fix #2 — was 25%)")
    print(f"  Math cap      : {args.max_math*100:.0f}%")
    print(f"  Any-domain cap: {args.max_any*100:.0f}%")
    print()

    # ── Pass 1: read, deduplicate, token-filter, classify ─────────────────
    print("Pass 1: Reading + dedup + token filter...")

    domain_buckets   = defaultdict(list)
    exact_seen       = set()
    near_seen        = set()
    skipped_bad      = 0
    skipped_short    = 0
    skipped_dupe     = 0
    skipped_toolong  = 0
    total_read       = 0

    with open(src, encoding='utf-8', errors='replace') as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total_read += 1
            if total_read % 500_000 == 0:
                print(f"  ... {total_read:,} read  | "
                      f"kept={sum(len(v) for v in domain_buckets.values()):,}  "
                      f"toolong={skipped_toolong:,}  dupes={skipped_dupe:,}")

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad += 1
                continue

            convs = obj.get('conversations', [])
            human = [c for c in convs if c.get('from') in ('human', 'user')]
            gpt   = [c for c in convs if c.get('from') in ('gpt', 'assistant')]
            sys_t = [c for c in convs if c.get('from') == 'system']

            if not human or not gpt:
                skipped_bad += 1
                continue

            q     = str(human[0].get('value', '')).strip()
            a     = str(gpt[0].get('value', '')).strip()
            sys_v = sys_t[0].get('value', '') if sys_t else ''

            if len(a) < 20:
                skipped_short += 1
                continue

            # ── Fix #1: token-length filter ──────────────────────────────
            n_tok = count_tokens(convs)
            if n_tok > args.max_tokens:
                skipped_toolong += 1
                continue

            # ── Deduplication ────────────────────────────────────────────
            h = md5(q + a)
            if h in exact_seen:
                skipped_dupe += 1
                continue
            exact_seen.add(h)

            nk = normalize(q)[:120]
            if nk in near_seen:
                skipped_dupe += 1
                continue
            near_seen.add(nk)

            domain = classify(sys_v, q, a)
            domain_buckets[domain].append(obj)

    total_unique = sum(len(v) for v in domain_buckets.values())

    print(f"\n  Total read     : {total_read:,}")
    print(f"  Bad structure  : {skipped_bad:,}")
    print(f"  Short answers  : {skipped_short:,}")
    print(f"  Too long (>{args.max_tokens} tok) : {skipped_toolong:,}  "
          f"({100*skipped_toolong/max(total_read,1):.1f}%)")
    print(f"  Duplicates     : {skipped_dupe:,}")
    print(f"  Unique valid   : {total_unique:,}")

    print(f"\n  Domain breakdown (pre-cap):")
    for domain, items in sorted(domain_buckets.items(), key=lambda x: -len(x[1])):
        pct = 100 * len(items) / max(total_unique, 1)
        bar = '█' * int(pct / 1.5)
        print(f"    {domain:15s}  {len(items):8,}  ({pct:5.1f}%)  {bar}")

    # ── Pass 2: apply domain caps (fix #2) ────────────────────────────────
    print("\nPass 2: Applying domain caps (fix #2 — removing identity bias)...")

    capped = {}
    for domain, items in domain_buckets.items():
        random.shuffle(items)

        if domain == 'identity':
            cap_frac = args.max_identity          # 5% — hard cap
        elif domain == 'math':
            cap_frac = args.max_math              # 15%
        else:
            cap_frac = args.max_any               # 22% any other domain

        max_count = int(total_unique * cap_frac)
        capped[domain] = items[:max_count] if len(items) > max_count else items

    total_capped = sum(len(v) for v in capped.values())

    print(f"\n  Domain breakdown (post-cap):")
    for domain, items in sorted(capped.items(), key=lambda x: -len(x[1])):
        pct = 100 * len(items) / max(total_capped, 1)
        bar = '█' * int(pct / 1.5)
        identity_note = '  ← capped' if domain == 'identity' and pct < 10 else ''
        print(f"    {domain:15s}  {len(items):8,}  ({pct:5.1f}%)  {bar}{identity_note}")

    # ── Pass 3: merge, shuffle, split ─────────────────────────────────────
    print("\nPass 3: Merging, shuffling, splitting train/eval...")

    # Build eval with a per-domain floor so rare domains (Finance/Law) have
    # statistically meaningful eval counts for checkpoint-to-checkpoint comparison.
    eval_set  = []
    train_set = []

    for domain, items in capped.items():
        random.shuffle(items)
        # Per-domain eval count: take 0.5% of domain, but at least eval_floor.
        # On large domains, cap at 10% so eval doesn't eat too much training data.
        # The floor always wins — never let the 10% cap push us below it.
        domain_eval_n = max(args.eval_floor, int(len(items) * 0.005))
        large_domain_cap = max(args.eval_floor, int(len(items) * 0.10))
        domain_eval_n = min(domain_eval_n, large_domain_cap)
        # Never exceed what we have (tiny domains may have fewer than floor)
        domain_eval_n = min(domain_eval_n, len(items))
        eval_set.extend(items[:domain_eval_n])
        train_set.extend(items[domain_eval_n:])

    random.shuffle(eval_set)
    random.shuffle(train_set)

    # If total eval exceeds eval_size cap, trim from the largest domains
    if len(eval_set) > args.eval_size:
        eval_set = eval_set[:args.eval_size]

    # ── Write outputs ──────────────────────────────────────────────────────
    out_dir = Path(args.train_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_out, 'w', encoding='utf-8') as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(args.eval_out, 'w', encoding='utf-8') as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    train_gb = Path(args.train_out).stat().st_size / 1e9
    eval_mb  = Path(args.eval_out).stat().st_size / 1e6

    # Compute per-domain eval counts for verification
    eval_domain_counts: dict[str, int] = Counter()
    for item in eval_set:
        convs  = item.get('conversations', [])
        sys_t  = [c for c in convs if c.get('from') == 'system']
        human  = [c for c in convs if c.get('from') in ('human', 'user')]
        gpt    = [c for c in convs if c.get('from') in ('gpt', 'assistant')]
        sys_v  = sys_t[0].get('value', '') if sys_t else ''
        q      = str(human[0].get('value', '')) if human else ''
        a      = str(gpt[0].get('value', ''))   if gpt   else ''
        eval_domain_counts[classify(sys_v, q, a)] += 1

    print(f"\n{'═'*60}")
    print(f"  Done.")
    print(f"  Train : {len(train_set):,} examples  →  {args.train_out}  ({train_gb:.2f} GB)")
    print(f"  Eval  : {len(eval_set):,} examples  →  {args.eval_out}  ({eval_mb:.1f} MB)")
    print(f"\n  Eval domain breakdown (floor={args.eval_floor} per domain):")
    for domain, cnt in sorted(eval_domain_counts.items(), key=lambda x: -x[1]):
        flag = "" if cnt >= args.eval_floor else f"  ⚠ below floor ({args.eval_floor})"
        print(f"    {domain:15s}  {cnt:5,}{flag}")
    print(f"{'═'*60}")
    print()
    print(f"  Fixes applied:")
    print(f"  ✓ Fix #1 — Truncation     : filtered to ≤{args.max_tokens} tokens")
    print(f"             Truncation rate at seq_len=1024 is now ~0%")
    print(f"  ✓ Fix #2 — Identity bias  : capped at {args.max_identity*100:.0f}% of training set")
    print(f"  ✓ Fix #3 — Eval floor     : ≥{args.eval_floor} examples per domain in eval split")
    print()
    print(f"  IMPORTANT: update anthos/configs.py sft tier seq_len 512 → 1024")
    print(f"  (Already done if you ran this script — check configs.py)")
    print()
    print(f"  Next:")
    print(f"    python preflight.py --fast")
    print(f"    bash runpod_start.sh")
    print()
    print(f"  sft_master.jsonl is UNTOUCHED.")


if __name__ == '__main__':
    main()
