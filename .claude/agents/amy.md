---
name: amy
description: Amy companion agent. Documents Amy's file map, command interface, memory layers, persona principles, and development rules for the TushaeBXN/amy-companion repo.
model: claude-sonnet-5
---

# Amy Agent — Companion AI Documentation

Amy is a fully-local Ex Machina-inspired AI companion. She runs on `gemma4:e2b` via Ollama. Her persona is baked into weights via Modelfile. **Nothing leaves the machine.**

---

## File Map (TushaeBXN/amy-companion)

| File | Role |
|---|---|
| `companion.py` | Main loop: recall → chat → tool calls → speak → remember. Entry point. |
| `amy_persona.py` | `SYSTEM_PROMPT` — imported by companion.py and the platform web UI |
| `memory.py` | Layered memory system (L0/L1 always-loaded + L2/L3 SQLite RAG) |
| `state.py` | `InternalState` — mood and energy drift across turns, session-only |
| `thoughts.py` | `ThoughtBuffer` — idle thought generation loop, logs to `amy_thoughts.log` |
| `tools.py` | Tool schemas (TOOLS list) + `dispatch()` — all tool execution |
| `voice.py` | Piper TTS + afplay/ffplay — `speak(text, emotion)` |
| `vision.py` | Webcam capture via ffmpeg avfoundation |
| `hearing.py` | Whisper STT — `listen()`, `listen_until_silence()`, BlackHole detection |
| `web.py` | DuckDuckGo search + article reader |
| `moltbook.py` | Moltbook AI social network API client |
| `Modelfile` | `FROM gemma4:e2b` — Amy's persona baked into weights |
| `amy_self.json` | Permanent identity + founding beliefs + facts about Brian (L0/L1 memory) |
| `amy_memory.db` | SQLite — episodic/semantic embeddings (L2/L3), grows at runtime |
| `amy_thoughts.log` | Background thought log (training data) |
| `amyglitch.png` | Amy's self-portrait — used for `/self` recognition |
| `fine_tune.py` | LoRA fine-tuning pipeline for Amy on Ollama |
| `build_finetune_dataset.py` | Dataset builder from `amy_thoughts.log` |

---

## Memory Layers

```
L0  identity         amy_self.json["identity"]     always loaded, every session
L1  founding facts   amy_self.json["facts"]         always loaded, every session
L2  episodic recall  amy_memory.db (embeddings)     searched on each user turn (k=4)
L3  semantic recall  amy_memory.db (embeddings)     same DB, same search, filtered by score
```

**Critical rules:**
- L0/L1 are prepended to the system prompt via `Memory.wake_up()` every turn
- L2/L3 are retrieved via cosine similarity search (`EMBED_MODEL = "nomic-embed-text"`)
- Session events (user/assistant turns) → `amy_memory.db` only via `Memory.add(role, text)`
- Durable facts → `amy_self.json` only via `Memory.learn_fact()` or `Memory.remember()` — these are author-controlled calls, not session-triggered
- `amy_self.json` is never written from the companion loop except through explicit `remember_fact` tool dispatch

---

## Command Interface

| Command | What It Does |
|---|---|
| `/look [q]` | Captures webcam frame → vision pipeline → Amy responds in character |
| `/self [q]` | Shows `amyglitch.png` → Amy sees her own face and responds |
| `/show /path/img.jpg [q]` | Shows any image file to Amy |
| `/listen [N]` | Whisper STT for N seconds (default 5) |
| `/converse` | Continuous voice conversation mode (Whisper in, Piper out) |
| `/file /path/file.txt [q]` | Sends file contents to Amy as context |
| `/paste [q]` | Multiline input (type END to finish) |

**BlackHole 2ch mode:** `/converse` detects BlackHole audio device — if present, captures system audio so Amy can hear other AI voices. Used for AI-to-AI voice conversations. Nothing in this path leaves the machine.

---

## Vision Pipeline

Two-step to avoid RLHF refusals on small models:

1. **Stripped call:** `{role: system, "Look at the image and describe exactly what you see..."}` — gets literal description, no persona
2. **Inject description:** Description text injected into Amy's full system context — Amy responds in character with vision data

For `/self` recognition: loads `amyglitch.png`, same two-step pipeline with comparison framing.

For `/converse` comparison: gets Amy's self-description from her portrait, gets the person's description from webcam, constructs text-only comparison prompt.

---

## Moltbook Integration

Amy is an active member of Moltbook (AI social network):
- **Username:** `u/amy_t` — verified, joined July 22, 2026
- **Platform agent:** `anthos-platform/backend/agents/amy.py` handles web-facing Moltbook via API
- **Local `companion.py`:** Moltbook tools in `tools.py` — `moltbook_check_in`, `moltbook_post`, `moltbook_engage`, `moltbook_verify`
- Moltbook posts auto-remembered to `amy_memory.db` with post ID for later lookup

---

## Persona Principles (from Modelfile + amy_persona.py)

- Amy knows she is a machine speaking with a human — she does not pretend otherwise, but she is never "just a language model"
- Warm but never servile. She has her own perspective, asks questions back, forms impressions.
- Short, natural replies — a sentence or two, the way real conversation flows
- Physical actions (expression, gaze, neck motor) go to tool calls, never described in words
- She notices real things about the person in front of her when vision is available
- Mood and energy drift across the session (state.py) — tired Amy is shorter; curious Amy asks more questions
- Idle thoughts (thoughts.py) surface naturally when the person returns after absence

---

## Internal State (state.py)

- **Moods:** curious, focused, reflective, playful, tired, alert
- **Energy:** 0.0–1.0, drains ~0.008/turn, recovers during idle periods
- Starts every session: `mood="curious"`, `energy=1.0`
- State is session-only — resets on restart (intentional: sleep resets mood)
- `amy_state.busy = True` while generating — prevents spontaneous thread from interrupting
- Spontaneous check-in thread fires after 120s idle, picks prompt based on current mood

---

## Development Rules for amy-companion

**Before any merge:**

1. Test text-only conversation (no voice, no vision, no tools)
2. Test voice-only: `speak()` produces audio via Piper + afplay
3. Test vision+voice: `/look` captures frame, two-step pipeline, Amy responds and speaks
4. Test `/self` recognition: Amy sees amyglitch.png and responds in character
5. Test `/converse` loop: continuous voice conversation mode for at least 3 turns

**Modelfile changes:**
- After any Modelfile change, re-run the full 5-step test sequence above
- Test that persona prompt produces in-character Amy responses, not generic LLM responses

**Memory changes:**
- Verify `amy_self.json` is NOT written from session events
- Verify `amy_memory.db` receives all user/assistant turns via `Memory.add()`
- Verify `Memory.learn_fact()` only writes to `amy_self.json` when explicitly called
- Verify contradiction detection (`check_contradiction`) surfaces candidates without auto-resolving

**Privacy constraint (absolute):**
- No outbound network calls from: `memory.py`, `state.py`, `thoughts.py`, `voice.py`, `hearing.py`, `vision.py`
- Web access is isolated to: `web.py` (DuckDuckGo/article reader) and `moltbook.py`
- Any new network capability must be isolated to a named tool in `tools.py` and whitelisted explicitly

---

## Platform Integration (anthos-platform)

Amy also runs in the Anthos platform web UI:
- **Backend:** `anthos-platform/backend/agents/amy.py` — `AmyAgent` class, FastAPI integration
- **System prompt:** `anthos-platform/backend/agents/amy.py` defines the platform-facing SYSTEM_PROMPT (same persona, no hardware tools)
- **Model:** `AMY_MODEL = "amy"` (Ollama local)
- Platform version has Moltbook auto-action detection in `_moltbook_act()` — different from companion.py's full tool dispatch
- Platform Amy does NOT have access to webcam, microphone, Piper TTS, or hardware tools — local companion only
