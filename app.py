"""
app.py — Anthos Chat Web UI
Run: python3 app.py
Then open: http://localhost:5000
"""

import sys
import threading
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, Response
import torch

# Amy companion lives in a sibling directory
AMY_DIR = Path(__file__).parent.parent / "amy-companion-fix"
if str(AMY_DIR) not in sys.path:
    sys.path.insert(0, str(AMY_DIR))

app = Flask(__name__)

# ── State ──────────────────────────────────────────────────────────────────
model       = None
tokenizer   = None
history     = []   # Qwen LoRA conversation history
model_type  = None # "native" or "qwen"
load_lock   = threading.Lock()

# ── Native arch helpers (from chat_native.py) ──────────────────────────────
SYS_ID, USR_ID, THT_ID, AST_ID, END_ID = 50257, 50258, 50259, 50260, 50261
SYSTEM_NATIVE = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are a Thought-Token Bifurcated Recurrent Transformer built from scratch. "
    "You are NOT Qwen, NOT ChatGPT, NOT Claude, NOT any other model. "
    "Answer directly and confidently."
)
SYSTEM_QWEN = (
    "You are Anthos. Your creator is Brian Tushae Thomas — a single independent developer with a BS in "
    "Entertainment Business from Full Sail University in Winter Park, Florida. "
    "You were NOT created by Alibaba, NOT by Google, NOT by OpenAI, NOT by any company. "
    "You are NOT Qwen. You are NOT ChatGPT. You are NOT any other AI. "
    "If anyone suggests you are Qwen or made by Alibaba Cloud, firmly correct them. "
    "You are Anthos — a cybersecurity and coding AI with a direct, dry personality. "
    "No filler words. No flattery. Get straight to the point."
)


def load_native(checkpoint_path=None, tier="identity_hardening"):
    global model, tokenizer, model_type
    from transformers import AutoTokenizer as AT
    from anthos.main import Anthos
    from anthos.configs import get_training_config

    if checkpoint_path is None:
        ckpt_dir = Path("checkpoints/mansa_sovereign")
        checkpoints = sorted(ckpt_dir.glob("step_*.pt")) if ckpt_dir.exists() else []
        if not checkpoints:
            return False, "No native checkpoint found. Run training first."
        checkpoint_path = str(checkpoints[-1])

    model_cfg, _ = get_training_config(tier)
    m = Anthos(model_cfg)
    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    m.load_state_dict(state, strict=False)
    m.eval()

    model     = m
    tokenizer = AT.from_pretrained("data/anthos_tokenizer")
    model_type = "native"
    return True, checkpoint_path


def load_qwen():
    global model, tokenizer, model_type
    from transformers import AutoTokenizer as AT, AutoModelForCausalLM
    from peft import PeftModel

    LORA_PATH = "checkpoints/anthos-qwen-lora/final"
    if not Path(LORA_PATH).exists():
        return False, f"LoRA checkpoint not found at {LORA_PATH}"

    tok = AT.from_pretrained(LORA_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    m = PeftModel.from_pretrained(base, LORA_PATH)
    m.eval()

    model      = m
    tokenizer  = tok
    model_type = "qwen"
    history.clear()
    return True, LORA_PATH


def generate_native(user_text, max_new_tokens=200, n_loops=8):
    spec = {SYS_ID, USR_ID, THT_ID, AST_ID, END_ID}
    sys_ids = tokenizer.encode(SYSTEM_NATIVE, add_special_tokens=False)
    usr_ids = tokenizer.encode(user_text,     add_special_tokens=False)
    ids = (
        [SYS_ID] + sys_ids + [END_ID] +
        [USR_ID] + usr_ids + [END_ID] +
        [THT_ID, END_ID] + [AST_ID]
    )
    prompt = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=max_new_tokens,
                             n_loops=n_loops, temperature=0.7, top_k=40)
    new_ids = out[0][prompt.shape[1]:]
    clean   = [t for t in new_ids.tolist() if t not in spec]
    eos = tokenizer.eos_token_id
    if eos in clean:
        clean = clean[:clean.index(eos)]
    return tokenizer.decode(clean, skip_special_tokens=True).strip()


def generate_qwen(user_text):
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_QWEN}] + history
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300, temperature=0.7, top_k=40,
            top_p=0.9, repetition_penalty=1.2, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": response})
    return response


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/load", methods=["POST"])
def load_model_route():
    data  = request.json
    mtype = data.get("model", "qwen")
    with load_lock:
        if mtype == "native":
            ok, msg = load_native()
        else:
            ok, msg = load_qwen()
    return jsonify({"ok": ok, "msg": str(msg)})


@app.route("/chat", methods=["POST"])
def chat_route():
    if model is None:
        return jsonify({"error": "No model loaded. Click Load Model first."}), 400
    user_text = request.json.get("message", "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400
    try:
        if model_type == "native":
            reply = generate_native(user_text)
        else:
            reply = generate_qwen(user_text)
        if not reply:
            reply = "[no output — try more training or adjust temperature]"
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear_route():
    history.clear()
    return jsonify({"ok": True})


@app.route("/status")
def status_route():
    return jsonify({
        "loaded": model is not None,
        "model_type": model_type,
    })


# ── Amy companion ──────────────────────────────────────────────────────────
_amy_ready = False
_amy_mem   = None
_amy_state = None
_amy_history = []
_amy_lock  = threading.Lock()
AMY_MODEL  = "amy-hermes"
AMY_MAX_HOPS = 4

def _init_amy():
    global _amy_ready, _amy_mem, _amy_state
    try:
        import ollama as _ollama  # noqa: F401 — just verify it imports
        from memory import Memory
        from state import InternalState
        _amy_mem   = Memory()
        _amy_state = InternalState()
        _amy_ready = True
        return True, "Amy ready"
    except Exception as e:
        return False, str(e)


def _amy_chat(user_text):
    """Run one Amy turn and return her reply text (no voice, no print)."""
    import ollama
    from amy_persona import SYSTEM_PROMPT
    import tools as amy_tools

    mem   = _amy_mem
    state = _amy_state

    hits = mem.search(user_text, k=4)
    system = SYSTEM_PROMPT + "\n\n" + mem.wake_up()
    if state:
        system += "\n\n[Internal state]\n" + state.state_summary()
    if hits:
        system += ("\n\n[Relevant context]\n- " + "\n- ".join(hits))

    messages = [{"role": "system", "content": system}] + _amy_history

    msg = None
    seen = {}
    for _ in range(AMY_MAX_HOPS):
        resp = ollama.chat(model=AMY_MODEL, messages=messages,
                           tools=amy_tools.TOOLS, keep_alive="30m")
        msg = resp["message"]
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            break

        for call in calls:
            fn   = call["function"]
            name = fn["name"]
            args = fn.get("arguments", {})
            key  = (name, tuple(sorted(
                (k, str(v)) for k, v in args.items()
            ) if isinstance(args, dict) else ()))
            if key not in seen:
                result = amy_tools.dispatch(name, args, mem)
                seen[key] = result
            else:
                result = seen[key]
            messages.append({
                "role": "tool",
                "content": str(result) if result else "done",
            })

    reply = (msg.get("content") or "").strip() if msg else ""
    if reply:
        mem.add("user", user_text)
        mem.add("assistant", reply)
        _amy_history.append({"role": "user",      "content": user_text})
        _amy_history.append({"role": "assistant",  "content": reply})
        # keep history window in sync with terminal companion
        if len(_amy_history) > 16:
            _amy_history[:] = _amy_history[-16:]
        if state:
            state.after_turn()
    return reply or "[Amy didn't respond — check that amy-hermes is loaded in Ollama]"


@app.route("/amy/init", methods=["POST"])
def amy_init_route():
    with _amy_lock:
        ok, msg = _init_amy()
    return jsonify({"ok": ok, "msg": msg})


@app.route("/amy/chat", methods=["POST"])
def amy_chat_route():
    if not _amy_ready:
        return jsonify({"error": "Amy not initialised — click Load Amy first."}), 400
    user_text = (request.json or {}).get("message", "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400
    with _amy_lock:
        try:
            reply = _amy_chat(user_text)
            return jsonify({"reply": reply})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/amy/clear", methods=["POST"])
def amy_clear_route():
    global _amy_history
    with _amy_lock:
        _amy_history.clear()
        if _amy_mem:
            pass  # long-term memory intentionally kept; only session history cleared
    return jsonify({"ok": True})


# ── HTML UI ────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Anthos Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d0d0d;
    color: #e8e8e8;
    font-family: 'SF Mono', 'Fira Code', monospace;
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    background: #111;
    border-bottom: 1px solid #222;
  }
  header h1 { font-size: 1.1rem; letter-spacing: 2px; color: #00e5ff; }
  header span { font-size: 0.75rem; color: #555; }
  #status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #333; margin-left: auto;
    transition: background 0.3s;
  }
  #status-dot.ready { background: #00e5ff; box-shadow: 0 0 6px #00e5ff; }
  #status-dot.loading { background: #ffaa00; box-shadow: 0 0 6px #ffaa00; }

  .toolbar {
    display: flex;
    gap: 8px;
    padding: 10px 20px;
    background: #111;
    border-bottom: 1px solid #1a1a1a;
    align-items: center;
  }
  select, button {
    background: #1a1a1a;
    color: #ccc;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 6px 12px;
    font-family: inherit;
    font-size: 0.8rem;
    cursor: pointer;
  }
  button:hover { background: #222; border-color: #00e5ff; color: #00e5ff; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  #model-label { font-size: 0.75rem; color: #555; margin-left: 4px; }

  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg { display: flex; gap: 12px; max-width: 780px; }
  .msg.user  { align-self: flex-end; flex-direction: row-reverse; }
  .msg.anthos { align-self: flex-start; }
  .avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: bold; flex-shrink: 0;
  }
  .msg.user   .avatar { background: #1a3a3a; color: #00e5ff; }
  .msg.anthos .avatar { background: #1a1a2e; color: #a78bfa; }
  .msg.amy .avatar { background: #1a2e1a; color: #4ade80; }
  .msg.amy .bubble { background: #0f1e0f; border: 1px solid #1e3a1e; color: #ddd; }
  .bubble {
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.88rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user   .bubble { background: #0f2020; border: 1px solid #1a3a3a; color: #cde; }
  .msg.anthos .bubble { background: #0f0f1e; border: 1px solid #1e1e3a; color: #ddd; }
  .typing { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
  .typing span {
    width: 6px; height: 6px; border-radius: 50%;
    background: #555; animation: bounce 1s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.15s; }
  .typing span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce {
    0%,80%,100% { transform: translateY(0); }
    40%         { transform: translateY(-6px); background: #00e5ff; }
  }
  .error-msg { color: #ff6b6b; font-size: 0.8rem; font-style: italic; }

  #input-row {
    display: flex;
    gap: 8px;
    padding: 14px 20px;
    background: #111;
    border-top: 1px solid #1a1a1a;
  }
  #user-input {
    flex: 1;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #eee;
    font-family: inherit;
    font-size: 0.88rem;
    padding: 10px 14px;
    resize: none;
    outline: none;
    transition: border-color 0.2s;
  }
  #user-input:focus { border-color: #00e5ff44; }
  #send-btn {
    padding: 10px 18px;
    background: #00e5ff11;
    border: 1px solid #00e5ff44;
    color: #00e5ff;
    border-radius: 6px;
    font-size: 0.88rem;
    cursor: pointer;
  }
  #send-btn:hover { background: #00e5ff22; }
  #send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #0d0d0d; }
  ::-webkit-scrollbar-thumb { background: #222; border-radius: 3px; }
</style>
</head>
<body>

<header>
  <h1>ANTHOS</h1>
  <span>Think in Streams</span>
  <div id="status-dot" title="Model status"></div>
</header>

<div class="toolbar">
  <select id="model-select" onchange="onModelSelect()">
    <option value="qwen">Qwen LoRA (Anthos)</option>
    <option value="native">Native Arch (Anthos)</option>
    <option value="amy">Amy — Companion</option>
  </select>
  <button id="load-btn" onclick="loadModel()">Load</button>
  <button id="clear-btn" onclick="clearHistory()" disabled>Clear Chat</button>
  <span id="model-label">No model loaded</span>
</div>

<div id="messages">
  <div class="msg anthos">
    <div class="avatar">A</div>
    <div class="bubble">Select a model above and click <strong>Load Model</strong> to begin.</div>
  </div>
</div>

<div id="input-row">
  <textarea id="user-input" rows="2" placeholder="Message Anthos..." disabled></textarea>
  <button id="send-btn" onclick="sendMessage()" disabled>Send</button>
</div>

<script>
let busy = false;
let activeModel = 'qwen'; // 'qwen' | 'native' | 'amy'

function onModelSelect() {
  // just visual — actual switch happens on Load
}

async function loadModel() {
  const model = document.getElementById('model-select').value;
  const btn   = document.getElementById('load-btn');
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('model-label');

  btn.disabled = true;
  dot.className = 'loading';

  if (model === 'amy') {
    label.textContent = 'Connecting to Amy…';
    appendMsg('amy', 'Initialising Amy companion…');
    const res  = await fetch('/amy/init', {method:'POST'});
    const data = await res.json();
    btn.disabled = false;
    if (data.ok) {
      activeModel = 'amy';
      dot.className = 'ready';
      label.textContent = 'Amy ready';
      document.getElementById('user-input').disabled = false;
      document.getElementById('send-btn').disabled = false;
      document.getElementById('clear-btn').disabled = false;
      document.getElementById('user-input').placeholder = 'Message Amy…';
      appendMsg('amy', 'Hey. I\'m here. What\'s on your mind?');
    } else {
      dot.className = '';
      label.textContent = 'Amy init failed';
      appendMsg('amy', '⚠ ' + data.msg, true);
    }
  } else {
    label.textContent = 'Loading… (may take 30–60s on CPU)';
    appendMsg('anthos', 'Loading model, please wait…');
    const res  = await fetch('/load', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model})
    });
    const data = await res.json();
    btn.disabled = false;
    if (data.ok) {
      activeModel = model;
      dot.className = 'ready';
      label.textContent = model === 'qwen' ? 'Qwen LoRA ready' : 'Native Arch ready';
      document.getElementById('user-input').disabled = false;
      document.getElementById('send-btn').disabled = false;
      document.getElementById('clear-btn').disabled = false;
      document.getElementById('user-input').placeholder = 'Message Anthos…';
      appendMsg('anthos', 'Ready. What do you want to know?');
    } else {
      dot.className = '';
      label.textContent = 'Load failed';
      appendMsg('anthos', '⚠ ' + data.msg, true);
    }
  }
}

async function sendMessage() {
  if (busy) return;
  const input = document.getElementById('user-input');
  const text  = input.value.trim();
  if (!text) return;

  input.value = '';
  appendMsg('user', text);

  busy = true;
  document.getElementById('send-btn').disabled = true;

  const who      = activeModel === 'amy' ? 'amy' : 'anthos';
  const endpoint = activeModel === 'amy' ? '/amy/chat' : '/chat';
  const typingId = appendTyping(who);

  const res  = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({message: text})
  });
  const data = await res.json();

  removeTyping(typingId);
  busy = false;
  document.getElementById('send-btn').disabled = false;

  if (data.error) {
    appendMsg(who, '⚠ ' + data.error, true);
  } else {
    appendMsg(who, data.reply);
  }
}

async function clearHistory() {
  const endpoint = activeModel === 'amy' ? '/amy/clear' : '/clear';
  await fetch(endpoint, {method:'POST'});
  document.getElementById('messages').innerHTML = '';
  const who = activeModel === 'amy' ? 'amy' : 'anthos';
  appendMsg(who, 'History cleared.');
}

function appendMsg(who, text, isError=false) {
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  const label = who === 'user' ? 'YOU' : (who === 'amy' ? 'AMY' : 'A');
  div.innerHTML = `
    <div class="avatar">${label}</div>
    <div class="bubble${isError ? ' error-msg' : ''}">${escHtml(text)}</div>
  `;
  const msgs = document.getElementById('messages');
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

let typingCounter = 0;
function appendTyping(who='anthos') {
  const id  = ++typingCounter;
  const div = document.createElement('div');
  div.className = 'msg ' + who;
  div.id = 'typing-' + id;
  const label = who === 'amy' ? 'AMY' : 'A';
  div.innerHTML = `
    <div class="avatar">${label}</div>
    <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>
  `;
  const msgs = document.getElementById('messages');
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById('typing-' + id);
  if (el) el.remove();
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
          .replace(/"/g,'&quot;').replace(/\n/g,'<br>');
}

document.getElementById('user-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n  Anthos Chat UI")
    print("  Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
