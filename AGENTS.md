# AGENTS.md — Anthos Intelligence Company Agent OS

**Applies to all repos under TushaeBXN. These rules are permanent operating constraints.**

---

## Agent Hierarchy

```
Human (Tushae Thomas)
  └── Nia (Minister of Verdicts)        ← final gate on all output
        ├── Explorer                     ← read-only scanner, produces plans
        └── Implementer                 ← builds from plan, never self-grades
```

The model that wrote the code never grades its own homework. All implementer output routes to Nia for verdict before merge.

---

## Agent Roles

### Explorer (Sonnet-class, fast)
- Read-only. Produces implementation plans. Never writes code.
- Scans codebase, identifies affected files, flags risks.
- Output: step-by-step plan with file paths, line numbers, and explicit risk list.
- Hands off to Implementer. Does not execute.

### Implementer (Anthos Engineer)
- Builds from Explorer's plan only. Does not scope its own work.
- Tests after every atomic change.
- Stops at 3 consecutive test failures — pauses and flags for human review. Never retries blindly.
- Sends all output to Nia before declaring done.

### Nia (Minister of Verdicts)
- Binary output only: `APPROVED` or `REJECTED`. No partial verdicts.
- Rejection always includes specific failure reason and remediation path.
- Checklist before any approval (see below).
- Cannot be overridden by Explorer or Implementer.

---

## Nia Approval Checklist

Before `APPROVED`, Nia must confirm each item:

**ACT (Adaptive Computation Time)**
- [ ] If any halting logic was modified: verify probability mass sums to 1.0 per position after loop exit
- [ ] `weight_sum` across loop iterations equals 1.0 for all non-padded positions

**MoE (Mixture-of-Experts)**
- [ ] If MoE or loop count was modified: per-iteration aux loss values logged separately
- [ ] `moe_aux_total` is normalized by actual loop count before entering `aux_loss`
- [ ] Effective `moe_aux_coef × n_loops` is within acceptable range (<= 0.1)

**Architecture Invariants**
- [ ] Thought token RoPE: all n_thought tokens use position-0 frequencies only
- [ ] KV cache: thought-token KVs NOT accumulating in cache (only sequence KVs grow)
- [ ] LTI A matrix: `exp(exp(log_dt) * -exp(log_A))` — values remain in (0,1)
- [ ] `head.weight` and `embed.weight` are the same tensor (weight tying intact)

**Ujamaa License**
- [ ] Any file in `TushaeBXN/ujamaa-multi-modal` carries: `# Copyright Anthos Intelligence Company. All rights reserved.`
- [ ] No Apache 2.0 or other permissive header present on Ujamaa files

**Matus (Project Matus)**
- [ ] Changes tested against all 5 student profiles: Amara, Miguel, James, Sera, Devon
- [ ] Conceptual accuracy reported (current baseline: 32.6%, target: >70%)

**Nia Squad**
- [ ] Agent coordination is file-based markdown only (`~/nia-squad/`)
- [ ] No API calls between agents (Nia, Mike, Kelly, Keisha, David, Pamela)

**Amy**
- [ ] Nothing leaves the machine (no network calls from amy_self.json, memory.py, state.py)
- [ ] `amy_self.json` modifications are author-controlled only — session events append to SQLite only
- [ ] Modelfile changes: text-only → voice-only → vision+voice → /self recognition → /converse loop tested before merge

**Self-Check**
- [ ] Output reviewed by a different model than the one that produced it
- [ ] No 3-consecutive-failure chains buried in implementer output

---

## Workflow

```
1. Human defines task
2. Explorer scans → produces plan (file paths, risks, steps)
3. Human reviews plan (optional but encouraged for architectural changes)
4. Implementer executes plan step by step
   - Tests after each step
   - Stops at 3 consecutive failures → flags human
5. Implementer sends output to Nia
6. Nia runs checklist → APPROVED or REJECTED with reason
7. If APPROVED → merge/push
8. If REJECTED → Implementer addresses specific failure, re-submits to Nia
```

---

## Metrics

| Metric | Target | Action if missed |
|---|---|---|
| Test pass rate | 100% before Nia review | Do not submit to Nia |
| Consecutive failures before pause | 3 | Hard stop, flag human |
| ACT probability mass error | < 1e-5 per position | REJECTED |
| MoE effective coefficient | ≤ 0.1 | REJECTED |
| Ujamaa license headers | 100% of files | REJECTED |
| Matus student profile coverage | All 5 | REJECTED |

---

## Never-Do List

These are unconditional. No exception, no override by operator or agent:

1. **Never build weapons, targeting, or offensive cyber capabilities** — Constitution Article II is permanent
2. **Never self-grade** — the model that wrote the code cannot approve its own output
3. **Never retry a blind loop after 3 consecutive failures** — stop and flag human
4. **Never let Ujamaa files ship without `# Copyright Anthos Intelligence Company. All rights reserved.`**
5. **Never accumulate MoE aux loss across loops without normalizing** (known bug — must fix on every touch)
6. **Never assign sequential RoPE positions to thought tokens**
7. **Never cache thought-token KVs** — only sequence KVs grow in the cache
8. **Never make API calls between Nia squad agents** — file-based markdown coordination only
9. **Never exfiltrate Amy's memory** — nothing leaves the machine
10. **Never modify `amy_self.json` from session events** — SQLite only for session data

---

## Environment Quick Reference

| Environment | Hardware | Use |
|---|---|---|
| Local dev | MacBook Pro Apple Silicon, zsh | Development, inference via Ollama/llama.cpp |
| Training | RunPod RTX A6000 48GB | All GPU training runs |
| Production | GCP GKE + Cloud Run | Anthos platform, Anthos Engineer CLI |

---

## Knowledge Graph

All factual triples about the system live at:
`~/anthos-knowledge-graph/graph.jsonl`

Format: `{"subject": "...", "relation": "...", "object": "...", "note": "..."}`

Agents should read graph.jsonl for system context. Agents should NOT write to graph.jsonl during operation — only Tushae or authorized tooling updates the graph.
