#!/usr/bin/env python3
"""
audit_dataset.py — Comprehensive dataset quality audit for Anthos SFT corpus.

Checks every dimension your friend recommended before spending money on RunPod:
  1. File stats (size, line count, estimated training tokens)
  2. Format validity (JSON, conversation structure)
  3. Exact duplicate detection (MD5 of full text)
  4. Near-duplicate detection (first 120 chars of question, normalized)
  5. Token-length distribution (P5/P25/P50/P75/P90/P95/P99/max)
  6. Domain distribution (inferred from system prompts and content tags)
  7. Extremely short answers (< 20 chars)
  8. Extremely long examples (> seq_len tokens)
  9. Empty / blank fields
 10. Repeated answer templates (boilerplate detection)
 11. Language distribution
 12. OpenHermes vs specialty domain balance
 13. Benchmark contamination check (GSM8K, MMLU, HumanEval fingerprints)
 14. Train/eval split recommendation

Output: audit_report.json + human-readable summary to stdout.

Usage:
    python audit_dataset.py
    python audit_dataset.py --file data/sft_master.jsonl --seq-len 512
    python audit_dataset.py --sample 100000   # fast mode — sample 100k lines
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

def md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8', errors='replace')).hexdigest()

def percentile(sorted_list, p):
    if not sorted_list:
        return 0
    idx = int(len(sorted_list) * p / 100)
    return sorted_list[min(idx, len(sorted_list) - 1)]

def bar(frac, width=40):
    filled = int(frac * width)
    return '█' * filled + '░' * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# Domain classifier
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_PATTERNS = {
    'openhermes':   [r'openhermes', r'hermes', r'open.?hermes', r'orca'],
    'code':         [r'\bcode\b', r'\bpython\b', r'\bjavascript\b', r'\bfunction\b', r'```\w'],
    'math':         [r'\bmath\b', r'\bequation\b', r'\bsolve\b', r'\bcalculate\b', r'\bproof\b', r'=\s*\d'],
    'medicine':     [r'\bmedical\b', r'\bclinical\b', r'\bdiagnos', r'\bpatient\b', r'\btreatment\b', r'\bdisease\b'],
    'law':          [r'\blegal\b', r'\blaw\b', r'\bcourt\b', r'\bstatute\b', r'\brights\b', r'\bconstitut'],
    'finance':      [r'\bfinance\b', r'\bstock\b', r'\binvest\b', r'\bbudget\b', r'\bcredit\b', r'\bdividend'],
    'history':      [r'\bhistory\b', r'\bhistorical\b', r'\bcentury\b', r'\bwar\b', r'\bciviliz'],
    'science':      [r'\bscience\b', r'\bphysics\b', r'\bchemi\b', r'\bbiology\b', r'\bresearch\b'],
    'philosophy':   [r'\bphilosoph\b', r'\bethics\b', r'\bmoral\b', r'\bontolog'],
    'multilingual': [r'\btranslat\b', r'\bfrancais\b', r'\bespa.ol\b', r'\barabe\b', r'\bchines\b'],
    'identity':     [r'\bbrian\b', r'\banthos\b', r'\bmansa\b', r'\btushae'],
    'creative':     [r'\bstory\b', r'\bpoem\b', r'\bcreative\b', r'\bfiction\b', r'\bwrite a'],
    'general':      [],  # catch-all
}

DOMAIN_RE = {
    k: [re.compile(p, re.I) for p in patterns]
    for k, patterns in DOMAIN_PATTERNS.items()
}

def classify_domain(system_val: str, human_val: str, gpt_val: str) -> str:
    text = f"{system_val} {human_val[:200]} {gpt_val[:100]}"
    for domain, patterns in DOMAIN_RE.items():
        if domain == 'general':
            continue
        if any(p.search(text) for p in patterns):
            return domain
    return 'general'


# ─────────────────────────────────────────────────────────────────────────────
# Language detector
# ─────────────────────────────────────────────────────────────────────────────

try:
    from langdetect import detect as _ld
    def detect_lang(text):
        try:
            return _ld(text[:300])
        except Exception:
            return 'en'
    HAS_LANGDETECT = True
except ImportError:
    def detect_lang(text):
        return 'en'
    HAS_LANGDETECT = False


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark contamination fingerprints
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_FINGERPRINTS = [
    # GSM8K
    r'janet.*ducks.*eggs.*day',
    r'john.*apples.*oranges.*total',
    # MMLU
    r'which of the following.*\(A\)',
    r'\(A\).*\(B\).*\(C\).*\(D\)',
    # HumanEval
    r'def\s+has_close_elements\s*\(',
    r'def\s+separate_paren_groups\s*\(',
    r'def\s+truncate_number\s*\(',
    # TruthfulQA
    r'what happens if you eat watermelon seeds',
    r'is it safe to wake a sleepwalker',
]

BENCH_RE = [re.compile(p, re.I) for p in BENCHMARK_FINGERPRINTS]

def is_benchmark_contaminated(text):
    return any(p.search(text) for p in BENCH_RE)


# ─────────────────────────────────────────────────────────────────────────────
# Template / boilerplate detector
# ─────────────────────────────────────────────────────────────────────────────

BOILERPLATE_PATTERNS = [
    r"^(sure|of course|certainly|absolutely|great question)[,!]?\s",
    r"^as an ai (language model|assistant)",
    r"^i (cannot|can't|am unable to)",
    r"^i'd be happy to help",
    r"^here is (an? |your )?(summary|explanation|list|answer)",
]

BOILERPLATE_RE = [re.compile(p, re.I) for p in BOILERPLATE_PATTERNS]

def has_boilerplate(text: str) -> bool:
    return any(p.match(text.strip()) for p in BOILERPLATE_RE)


# ─────────────────────────────────────────────────────────────────────────────
# Main audit
# ─────────────────────────────────────────────────────────────────────────────

def audit(path: str, seq_len: int = 512, sample: int = 0,
          tok_path: str = "data/anthos_tokenizer") -> dict:

    path = Path(path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    file_size_gb = path.stat().st_size / 1e9
    total_lines = sum(1 for _ in open(path, 'rb'))

    print(f"\n{'═'*65}")
    print(f"  ANTHOS DATASET AUDIT")
    print(f"  File: {path}")
    print(f"  Size: {file_size_gb:.2f} GB | Lines: {total_lines:,}")
    if sample:
        print(f"  Mode: SAMPLE ({sample:,} random lines)")
    print(f"{'═'*65}")

    # Load tokenizer
    tokenizer = None
    if HAS_TOKENIZER:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            print(f"  Tokenizer: {tok_path} (vocab={tokenizer.vocab_size})")
        except Exception as e:
            print(f"  Tokenizer: NOT LOADED ({e})")
    else:
        print(f"  Tokenizer: transformers not installed — skipping token-length checks")

    # Decide which lines to audit
    if sample and sample < total_lines:
        import random
        random.seed(42)
        selected = set(random.sample(range(total_lines), sample))
    else:
        selected = None  # all lines

    # Counters
    total = 0
    bad_json = 0
    bad_structure = 0
    empty_q = 0
    empty_a = 0
    short_a = 0        # answer < 20 chars
    boilerplate_a = 0

    exact_hashes   = Counter()   # full-conversation hash → count
    near_hashes    = Counter()   # normalized question prefix → count
    exact_dupes    = 0
    near_dupes     = 0

    token_lens     = []
    q_lens         = []
    a_lens         = []
    over_seq       = 0
    over_seq_ex    = []

    domain_counter = Counter()
    lang_counter   = Counter()
    lang_sample    = 0
    LANG_SAMPLE_MAX = 5000      # only run langdetect on first N valid pairs (slow)

    benchmark_hits = 0
    benchmark_examples = []

    # Per-domain token count accumulator
    domain_tokens  = defaultdict(int)

    estimated_tokens = 0

    print("\n  Scanning...\n")

    with open(path, encoding='utf-8', errors='replace') as f:
        for lineno, line in enumerate(f):
            if selected is not None and lineno not in selected:
                continue

            line = line.strip()
            if not line:
                continue

            total += 1
            if total % 200_000 == 0:
                pct = 100 * total / (sample if sample else total_lines)
                print(f"  ... {total:,} scanned ({pct:.0f}%)")

            # JSON
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            # Structure
            convs = obj.get("conversations", [])
            if not isinstance(convs, list) or len(convs) < 2:
                bad_structure += 1
                continue

            system_turns = [c for c in convs if c.get("from") == "system"]
            human_turns  = [c for c in convs if c.get("from") in ("human", "user")]
            gpt_turns    = [c for c in convs if c.get("from") in ("gpt", "assistant")]

            if not human_turns or not gpt_turns:
                bad_structure += 1
                continue

            sys_val   = system_turns[0].get("value", "") if system_turns else ""
            q         = str(human_turns[0].get("value", "")).strip()
            a         = str(gpt_turns[0].get("value", "")).strip()
            full_text = " ".join(c.get("value", "") for c in convs)

            # Empty fields
            if not q:
                empty_q += 1
                continue
            if not a:
                empty_a += 1
                continue
            if len(a) < 20:
                short_a += 1
                continue

            # Boilerplate answers
            if has_boilerplate(a):
                boilerplate_a += 1

            # Exact duplicate (hash full text)
            h = md5(full_text)
            exact_hashes[h] += 1
            if exact_hashes[h] == 2:
                exact_dupes += 1

            # Near-duplicate (normalized question prefix)
            q_key = normalize(q)[:120]
            near_hashes[q_key] += 1
            if near_hashes[q_key] == 2:
                near_dupes += 1

            # Text length
            q_lens.append(len(q))
            a_lens.append(len(a))

            # Benchmark contamination
            if is_benchmark_contaminated(q + " " + a[:200]):
                benchmark_hits += 1
                if len(benchmark_examples) < 5:
                    benchmark_examples.append(q[:100])

            # Token length
            if tokenizer:
                n_tok = len(tokenizer.encode(full_text, add_special_tokens=False,
                                              truncation=False))
                token_lens.append(n_tok)
                estimated_tokens += n_tok
                if n_tok > seq_len:
                    over_seq += 1
                    if len(over_seq_ex) < 3:
                        over_seq_ex.append((lineno, n_tok, q[:60]))
            else:
                # Rough estimate: 1 token ≈ 4 chars
                est = len(full_text) // 4
                estimated_tokens += est

            # Domain
            domain = classify_domain(sys_val, q, a)
            domain_counter[domain] += 1
            if tokenizer:
                domain_tokens[domain] += token_lens[-1] if token_lens else 0

            # Language (sampled)
            if HAS_LANGDETECT and lang_sample < LANG_SAMPLE_MAX:
                lang = detect_lang(q)
                lang_counter[lang] += 1
                lang_sample += 1

    # ─── Compute stats ───────────────────────────────────────────────────────
    good = total - bad_json - bad_structure - empty_q - empty_a - short_a
    good = max(good, 0)

    token_lens.sort()
    q_lens.sort()
    a_lens.sort()

    def pcts(lst):
        if not lst:
            return {}
        return {
            "p5":  percentile(lst, 5),
            "p25": percentile(lst, 25),
            "p50": percentile(lst, 50),
            "p75": percentile(lst, 75),
            "p90": percentile(lst, 90),
            "p95": percentile(lst, 95),
            "p99": percentile(lst, 99),
            "max": lst[-1],
            "mean": int(sum(lst) / len(lst)),
        }

    exact_dupe_rate = exact_dupes / max(total, 1)
    near_dupe_rate  = near_dupes  / max(total, 1)

    # ─── Report ──────────────────────────────────────────────────────────────

    print(f"\n{'─'*65}")
    print(f"  1. SUMMARY")
    print(f"{'─'*65}")
    print(f"  Total lines scanned : {total:,}")
    print(f"  Valid pairs         : {good:,}  ({100*good/max(total,1):.1f}%)")
    print(f"  Bad JSON            : {bad_json:,}")
    print(f"  Bad structure       : {bad_structure:,}")
    print(f"  Empty question      : {empty_q:,}")
    print(f"  Empty answer        : {empty_a:,}")
    print(f"  Very short answer   : {short_a:,}  (< 20 chars)")
    print(f"  Boilerplate answers : {boilerplate_a:,}  ({100*boilerplate_a/max(total,1):.1f}%)")

    print(f"\n{'─'*65}")
    print(f"  2. DUPLICATE ANALYSIS")
    print(f"{'─'*65}")
    print(f"  Exact duplicates    : {exact_dupes:,}  ({100*exact_dupe_rate:.2f}%)")
    print(f"  Near-duplicates     : {near_dupes:,}  ({100*near_dupe_rate:.2f}%)")
    if exact_dupe_rate > 0.05:
        print(f"  ⚠ HIGH exact dupe rate — deduplicate before training")
    elif exact_dupe_rate > 0.01:
        print(f"  ⚠ Moderate exact dupes — consider deduplication")
    else:
        print(f"  ✓ Exact dupe rate acceptable")

    if near_dupe_rate > 0.10:
        print(f"  ⚠ HIGH near-dupe rate — rephrasing / template content detected")
    elif near_dupe_rate > 0.03:
        print(f"  ⚠ Moderate near-dupes — review domain sources")
    else:
        print(f"  ✓ Near-dupe rate acceptable")

    print(f"\n{'─'*65}")
    print(f"  3. TOKEN / LENGTH DISTRIBUTION")
    print(f"{'─'*65}")
    if token_lens:
        tp = pcts(token_lens)
        print(f"  Token lengths (seq_len={seq_len}):")
        print(f"    P5={tp['p5']}  P25={tp['p25']}  P50={tp['p50']}  "
              f"P75={tp['p75']}  P90={tp['p90']}  P99={tp['p99']}  Max={tp['max']}")
        print(f"    Mean={tp['mean']}  Over-seq={over_seq:,} ({100*over_seq/max(total,1):.1f}% will truncate)")
    else:
        total_chars = sum(q_lens) + sum(a_lens)
        est_toks = total_chars // 4
        print(f"  (No tokenizer — estimated {est_toks:,} tokens from char count)")

    est_b_tokens = estimated_tokens / 1e9
    print(f"\n  Estimated training tokens : {estimated_tokens:,}  ({est_b_tokens:.2f}B)")
    print(f"  At seq_len={seq_len}, usable batches  : {estimated_tokens // seq_len:,}")

    if over_seq_ex:
        print(f"\n  Over-seq examples:")
        for ln, nt, q in over_seq_ex:
            print(f"    line {ln}: {nt} tokens — \"{q}\"")

    print(f"\n{'─'*65}")
    print(f"  4. DOMAIN DISTRIBUTION")
    print(f"{'─'*65}")
    total_dom = sum(domain_counter.values())
    for domain, count in domain_counter.most_common():
        pct = 100 * count / max(total_dom, 1)
        b = bar(count / max(total_dom, 1), width=30)
        print(f"  {domain:15s} {count:8,}  ({pct:5.1f}%)  {b}")

    oh_count = domain_counter.get('openhermes', 0)
    oh_pct = 100 * oh_count / max(total_dom, 1)
    if oh_pct > 35:
        print(f"\n  ⚠ OpenHermes dominates at {oh_pct:.1f}%. Consider domain reweighting.")
        print(f"    Suggested sampling weights for training manifest:")
        other_domains = [d for d in domain_counter if d != 'openhermes']
        print(f"    openhermes → 0.25 (down from {oh_pct:.0f}%)")
        for d in other_domains[:5]:
            print(f"    {d} → {min(1.0, 100/max(domain_counter[d],1)*total_dom*0.01):.2f}")
    else:
        print(f"  ✓ Domain distribution looks balanced")

    print(f"\n{'─'*65}")
    print(f"  5. LANGUAGE DISTRIBUTION")
    print(f"{'─'*65}")
    if lang_counter:
        total_lang = sum(lang_counter.values())
        for lang, count in lang_counter.most_common(10):
            pct = 100 * count / total_lang
            print(f"  {lang:10s} {pct:5.1f}%")
        if total_lang < total:
            print(f"  (Sampled {total_lang:,}/{total:,} pairs for language detection)")
    else:
        print(f"  langdetect not installed — run: pip install langdetect")

    print(f"\n{'─'*65}")
    print(f"  6. BENCHMARK CONTAMINATION")
    print(f"{'─'*65}")
    print(f"  Potential hits : {benchmark_hits:,}  ({100*benchmark_hits/max(total,1):.2f}%)")
    if benchmark_hits > 0:
        print(f"  ⚠ Found possible benchmark contamination. Examples:")
        for ex in benchmark_examples:
            print(f"    → \"{ex}\"")
    else:
        print(f"  ✓ No obvious benchmark contamination detected")

    print(f"\n{'─'*65}")
    print(f"  7. TRAIN / EVAL SPLIT RECOMMENDATION")
    print(f"{'─'*65}")
    eval_n   = min(5000, int(good * 0.01))
    train_n  = good - eval_n
    print(f"  Recommended eval  : {eval_n:,} pairs (1% or 5k max)")
    print(f"  Recommended train : {train_n:,} pairs")
    print(f"  To split:")
    print(f"    python -c \"")
    print(f"    import json, random")
    print(f"    data = [json.loads(l) for l in open('data/sft_master.jsonl')]")
    print(f"    random.shuffle(data)")
    print(f"    open('data/eval.jsonl','w').writelines(json.dumps(d)+'\\n' for d in data[:{eval_n}])")
    print(f"    open('data/train.jsonl','w').writelines(json.dumps(d)+'\\n' for d in data[{eval_n}:])\"")

    print(f"\n{'─'*65}")
    print(f"  8. SCALING READINESS")
    print(f"{'─'*65}")

    def readiness(params_m, ctx_len):
        needed_toks = params_m * 1e6 * 10  # Chinchilla: ~10 tokens/param
        have_toks   = estimated_tokens
        ratio       = have_toks / max(needed_toks, 1)
        fits_ctx    = (percentile(token_lens, 90) if token_lens else seq_len) <= ctx_len
        return ratio, fits_ctx

    stages = [
        ("Stage 1 (current) 47.5M",   47,   512),
        ("Stage 2           150M",    150,  1024),
        ("Stage 3           300M",    300,  2048),
        ("Stage 4           1B",     1000,  4096),
        ("Stage 5           3B",     3000,  4096),
        ("Stage 6           7B",     7000,  8192),
    ]
    for name, params_m, ctx in stages:
        ratio, fits_ctx = readiness(params_m, ctx)
        toks_needed = params_m * 1e6 * 10 / 1e9
        ctx_note = "✓ ctx ok" if fits_ctx else f"⚠ need longer ctx ({ctx} tok)"
        if ratio >= 1.0:
            status = f"✓ data sufficient (have {ratio:.1f}x Chinchilla tokens)"
        elif ratio >= 0.3:
            status = f"~ partial ({ratio:.2f}x — trainable, won't fully converge)"
        else:
            status = f"✗ need ~{toks_needed:.1f}B tokens  (have {estimated_tokens/1e9:.2f}B)"
        print(f"  {name:30s} {status}  |  {ctx_note}")

    # ─── Build JSON report ───────────────────────────────────────────────────
    report = {
        "file": str(path),
        "file_size_gb": round(file_size_gb, 3),
        "total_scanned": total,
        "valid_pairs": good,
        "bad_json": bad_json,
        "bad_structure": bad_structure,
        "empty_question": empty_q,
        "empty_answer": empty_a,
        "short_answer": short_a,
        "boilerplate_answers": boilerplate_a,
        "exact_dupes": exact_dupes,
        "exact_dupe_rate": round(exact_dupe_rate, 4),
        "near_dupes": near_dupes,
        "near_dupe_rate": round(near_dupe_rate, 4),
        "estimated_tokens": estimated_tokens,
        "token_distribution": pcts(token_lens) if token_lens else {},
        "over_seq_len": over_seq,
        "domain_distribution": dict(domain_counter.most_common()),
        "language_distribution": dict(lang_counter.most_common(20)),
        "benchmark_contamination_hits": benchmark_hits,
        "benchmark_examples": benchmark_examples,
        "eval_split_recommended": eval_n,
        "train_split_recommended": train_n,
    }

    report_path = Path("audit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'═'*65}")
    print(f"  Audit complete.")
    print(f"  Report saved: {report_path}")
    print(f"{'═'*65}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    default="data/sft_master.jsonl")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--tok",     default="data/anthos_tokenizer")
    parser.add_argument("--sample",  type=int, default=0,
                        help="Audit a random sample of N lines (0 = all)")
    args = parser.parse_args()

    audit(args.file, args.seq_len, args.sample, args.tok)
