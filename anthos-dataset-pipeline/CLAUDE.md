# Anthos Training Dataset Pipeline — Claude Code Handoff

## Owner
Brian Thomas (Tushae) — Anthos Intelligence Company
GitHub: TushaeBXN / TushaeThomas
HuggingFace: machomenc

## What This Is
A pipeline that ingests code and conversation data from three sources and
produces ShareGPT-format JSONL training data for fine-tuning Anthos and Matus models.

## What's Already Built (by Claude.ai)
- `ingest/github_ingest.py` — pulls all repos via GitHub API, extracts code files
- `ingest/claude_ai_ingest.py` — parses Claude.ai data export JSON
- `ingest/claude_code_ingest.py` — reads ~/.claude/projects/ session logs
- `normalize/normalize.py` — converts all three formats to ShareGPT JSONL
- `normalize/deduper.py` — removes near-duplicate examples (MinHash)
- `config.py` — all paths, keys, and toggles in one place
- `requirements.txt` — all dependencies

## What Claude Code Needs to Finish

### Priority 1 — Wiring & Testing
1. Run `pip install -r requirements.txt`
2. Set env vars (see config.py — GITHUB_TOKEN, HF_TOKEN, CLAUDE_EXPORT_PATH)
3. Run each ingest script individually and verify output JSONL looks correct
4. Run normalize.py end-to-end and check the merged output

### Priority 2 — Quality Filter
Build `filter/quality_filter.py` that drops examples where:
- Code block is under 10 lines (too trivial)
- No code block present at all
- Response is just "I can't help with that" type refusals
- Exact duplicate of another example (hash match)

### Priority 3 — HuggingFace Push
Build `push_to_hub.py` that:
- Loads the final filtered JSONL
- Pushes to `machomenc/anthos-code-training-v1` on HuggingFace
- Uses `datasets` library, DatasetDict with train/test 95/5 split
- Prints upload URL when done

### Priority 4 — Thought-Token Augmentation (Anthos-specific)
Build `augment/thought_token_wrap.py` that:
- Takes the final JSONL
- For each assistant turn that contains code, calls the Anthropic API
- Asks Claude to prepend a <|thought|>...</|thought|> reasoning block
  before the code explaining the design decision
- Outputs `anthos-code-training-v1-thought.jsonl`
- This teaches Anthos's bifurcated thought/sequence stream architecture

### Priority 5 — Stats Report
Build `report.py` that prints:
- Total examples per source (GitHub / Claude.ai / Claude Code)
- Token count estimate (rough: chars / 4)
- Domain breakdown (ML, frontend, backend, GRC, infra)
- Duplicate removal count

## Output Files
```
output/
  raw/
    github_raw.jsonl
    claude_ai_raw.jsonl
    claude_code_raw.jsonl
  merged/
    anthos-code-training-v1.jsonl          # normalized + deduped
    anthos-code-training-v1-thought.jsonl  # with thought tokens
```

## Important Notes
- Never hardcode API keys — all in config.py via os.environ
- GitHub ingest: skip repos that are forks (is_fork: true)
- Claude Code logs: ~/.claude/projects/ — each subfolder is one project session
- The thought-token augmentation is optional — only run if HF_TOKEN and
  ANTHROPIC_API_KEY are both set
- Match existing ShareGPT format exactly:
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
