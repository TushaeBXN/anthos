# Anthos Action Plan
**Owner:** Brian Tushae Thomas | **Last updated:** 2026-08-21

This is the living punch list — everything decided and ready to execute, in priority order.
The Training Roadmap (TRAINING_ROADMAP.md) covers the long arc. This file covers what's next.

---

## Status Legend
- `[ ]` — not started
- `[~]` — in progress
- `[x]` — done

---

## 1. Code Training — MacBook (no GPU needed)

### 1a. Generate code teacher data from HuggingFace (free, multi-language)
```bash
cd ~/Desktop/anthos-repo
source venv/bin/activate
python3 generate_code_teacher_data.py --n 20000 --hf
```
Pulls from `nickrosh/Evol-Instruct-Code-80k-v1` + `iamtarun/python_code_instructions_18k_alpaca`.
Output: `data/code_teacher.jsonl`

- [ ] Run this

### 1b. Generate code eval data (teaches Anthos to spot bugs and self-correct)
```bash
python3 generate_code_eval_data.py --n 2000 --model qwen2.5-coder:7b
```
Requires Ollama running with `qwen2.5-coder:7b` pulled.
Output: `data/code_eval.jsonl`

- [ ] Pull model: `ollama pull qwen2.5-coder:7b`
- [ ] Run this

### 1c. Academic paper knowledge — arXiv (44 categories) + papers-we-love (free, no GPU, no API)
```bash
cd ~/Desktop/anthos-repo
source venv/bin/activate
python3 generate_papers_training_data.py
```
arXiv categories: CS (15), Math (8), Physics (6), Biology (5), Economics (3), Stats (3), EE (3).
+ papers-we-love topic READMEs for 14 CS/systems topics.
Each paper → 2–3 Q&A pairs. Expected: ~50–150K examples.
Output: `data/papers_knowledge.jsonl`
Note: arXiv rate-limits to 1 req/sec — takes ~15 min to run.

- [ ] Run this

### 1c2. PhD-breadth Wikipedia knowledge (free, no GPU, no API)
```bash
python3 generate_wikipedia_training_data.py
```
Pulls Wikipedia Featured Articles (6,700) + Good Articles (37,000) + 90 PhD domain categories:
Physics, Chemistry, Biology, Mathematics, Medicine, Law, Philosophy, History, Linguistics,
Economics, Psychology, Neuroscience, Engineering, Astronomy, Art History, Music Theory,
Environmental Science, Political Science, Sociology, Anthropology, and more.
Each article → 3–4 Q&A pairs. Up to 50K articles = ~150–200K training records.
Output: `data/wiki_knowledge.jsonl`
Note: Wikipedia is generous with rate limits — takes ~2–3 hours for full run.
For a faster test: `python3 generate_wikipedia_training_data.py --max-articles 5000`

- [ ] Run this (can run overnight)

### 1d. Awesome-list ecosystem knowledge (free, no GPU, no API)
```bash
cd ~/Desktop/anthos-repo
source venv/bin/activate
python3 generate_awesome_training_data.py
```
Fetches 10 awesome-* lists (Python, JS, Node, Go, Rust, Security, Hacking, Docker, Shell, TypeScript).
Each library entry → 5 Q&A variations + per-category summary.
Expected output: ~30–60K examples.
Output: `data/awesome_knowledge.jsonl`

- [ ] Run this (fast — only hits GitHub, no inference needed)

### 1e. Pull 338K code examples from HuggingFace (Magicoder + CodeFeedback)
```bash
cd ~/Desktop/anthos-repo/anthos_code_training
python3 download_and_build.py        # downloads ~3 datasets, deduplicates → combined/anthos_code_sft_deduped.jsonl
python3 convert_to_sharegpt.py       # converts to Anthos ShareGPT → ../data/code_magicoder.jsonl
```
Sources: Magicoder-OSS-75K (MIT), CodeFeedback-156K (Apache-2.0), Magicoder-Evol-110K (Apache-2.0).
Output: `data/code_magicoder.jsonl` (~338K rows after dedup)

- [ ] Run download_and_build.py (large download, needs internet)
- [ ] Run convert_to_sharegpt.py

### 1f. Merge and train code tier
```bash
cat data/code_teacher.jsonl data/code_eval.jsonl data/code_magicoder.jsonl \
    data/awesome_knowledge.jsonl data/papers_knowledge.jsonl data/wiki_knowledge.jsonl \
    > data/code_combined.jsonl
python3 train.py --tier code --steps 20000
```
Config: SEQ_LEN=1024, MAX_LR=5e-5, 10k steps base + 5k extra for eval patterns.

- [ ] Bump SEQ_LEN in train.py code tier from 512 → 1024 before running
- [ ] Run training

### 1g. Wire Ollama to use Anthos as the model
```bash
ollama create anthos -f Modelfile
ANTHOS_MODEL=anthos anthos-engineer
```
Test it: ask it to build a simple HTML page, then use the Preview button in the UI.

- [ ] Create ollama model from Modelfile
- [ ] Test with anthos-engineer

---

## 2. anthos-engineer Improvements

### 2a. Add `str_replace` action to agent.py
**Why:** The model currently rewrites entire files on `modify`/`debug` intents.
Targeted edits (find exact block → replace with fix) is how top-performing
coding agents work — it's how Claude scores on SWE-bench.

New action in the plan step JSON:
```json
{"action": "str_replace", "target": "main.py", "find": "...", "replace": "..."}
```
Agent reads the file, replaces the exact block, writes it back.
Much lower chance of introducing regressions vs full rewrites.

- [ ] Add `str_replace` action to `execute_step()` in `anthos_engineer/agent.py`
- [ ] Add it to the prompt so the model knows to use it for modify/debug intents

### 2b. Bump code tier SEQ_LEN 512 → 1024
In `train.py`, code tier block:
```python
SEQ_LEN = 1024   # was 512 — needed for real codebases
```
- [ ] Make this change before running training (step 1c)

---

## 3. Constitutional AI Training Data Generator

**What:** A script `generate_constitution_data.py` that creates self-critique + revision
training pairs grounded in `docs/CONSTITUTION.md`. Teaches Anthos to:
1. Generate a response
2. Check it against its own constitution (honesty, no-war, helpfulness)
3. Revise it if it fails any article

This closes the loop between the written constitution and what the model actually does.

- [ ] Write `generate_constitution_data.py`
- [ ] Add `constitution` tier to `train.py`
- [ ] Generate ~2,000 constitution critique/revision pairs
- [ ] Run `python3 train.py --tier constitution --steps 5000`

---

## 4. Code Behavior Steering Vector

Generates a contrastive steering vector that pushes Anthos toward code-direct behavior
(concise, uses fences, no filler) vs prose-verbose behavior.
Requires the native arch checkpoint (not Qwen LoRA).

```bash
python3 generate_code_steering.py --checkpoint checkpoints/mansa_sovereign/
```
Output: `vectors/code_persona.pt`

Then in `chat_native.py`:
```python
steer.load_persona("vectors/code_persona.pt")
steer.engage(strength=0.6)
```

- [ ] Confirm native arch checkpoint path is current
- [ ] Run `generate_code_steering.py`
- [ ] Wire into `chat_native.py` with `--code` flag

---

## 5. When DGX Spark (GB10) Arrives

The MacBook is for iteration. The Spark is for scale. When it arrives:

### 5a. Phase 2 — Cybersecurity domain (first GPU run)
```bash
# ~160K examples, Fenrir + Trendyol + CyberNative
# See TRAINING_ROADMAP.md Phase 2 for exact commands
python3 train.py --tier instruct --steps 50000   # or dedicated cyber tier
```

### 5b. Qwen2.5-72B LoRA — "Big Anthos"
Same approach as the current 1.5B LoRA, 48x bigger base model.
Much stronger reasoning out of the box. Fine-tune for Anthos identity + code style.
Needs 128GB unified memory — the GB10 has exactly that.

```bash
# Base: Qwen/Qwen2.5-72B-Instruct
# LoRA rank 64, target all attention + MLP projections
# Train on code_combined.jsonl + cybersecurity + identity data
```

### 5c. Phase 3 — Reasoning (Claude CoT + NuminaMath + OpenR1)
~250K examples. See TRAINING_ROADMAP.md Phase 3.

### 5d. Phase 4 — Scale (UltraChat + Orca + Nemotron)
1M+ examples. The full Anthos intelligence expansion.

---

## 6. Anthos Engineer — Pending Features

Beyond str_replace (item 2a above):

### 6a. Multi-file context
Agent should be able to read existing workspace files before writing new ones,
so modify/debug tasks don't ignore what's already there.

- [ ] Auto-read all existing workspace files into `_context` at session start when intent = modify/debug

### 6b. Streaming output tokens
Currently waits for the full model response before yielding the `step_done` event.
Should stream tokens to the UI in real time for long file generations.

- [ ] Switch `/api/session/{id}/stream` write step to token-by-token SSE streaming

---

## Notes

- **Constitution is already complete** — `docs/CONSTITUTION.md` v3.0. No changes needed.
  The CAI training data generator (item 3) is what bridges it from document to behavior.
- **Training Roadmap is the long game** — `docs/TRAINING_ROADMAP.md` covers 6.5M examples.
  This action plan is what's executable *right now* before the big GPU runs.
- **anthos-engineer already has**: preview panel, download, open-in-browser (added 2026-08-21),
  code review loop (`_review_and_fix`), semantic session history, intent routing, compaction.

---

*Built independently. Bachelor's in Entertainment Business, Full Sail University. Just the work.*
