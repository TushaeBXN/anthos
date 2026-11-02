"""Pod-side watcher: evaluates each new checkpoint, scans the training log, prints EVAL / ALERT / HEARTBEAT lines."""
import os, re, sys, time, json, glob, subprocess
RUN, TOTAL = sys.argv[1], int(sys.argv[2]); D = f"checkpoints/{RUN}"; LOG = f"{D}/train.log"
PY = "/venv/main/bin/python"
seen_ck, sizes, evals, last_alert, pos, lows = set(), {}, [], {}, 0, []
slow = 0; last_beat = time.time(); last_step = 0; done = False
def say(s): print(s, flush=True)
def alert(key, msg):
    if time.time() - last_alert.get(key, 0) > 900: last_alert[key] = time.time(); say("ALERT " + msg)
def running(): return subprocess.run(["pgrep", "-f", "python train[.]py"], capture_output=True).returncode == 0
say(f"WATCH start run={RUN} total_steps={TOTAL}")
while not done:
    time.sleep(30)
    try:
        with open(LOG, errors="ignore") as f:
            f.seek(pos); new = f.read(); pos = f.tell()
    except FileNotFoundError: new = ""
    for ln in new.splitlines():
        if "Traceback" in ln or "Error" in ln and "step" not in ln[:5]: alert("err", f"log error: {ln[:160]}")
        m = re.match(r"step\s+(\d+) \| loss ([\w.]+) .*moe_top1 ([\d.]+)%.* ([\d,]+) tok/s", ln)
        if not m: continue
        st, loss, top1, tps = int(m[1]), float(m[2]) if m[2] not in ("nan", "inf") else float("nan"), float(m[3]), int(m[4].replace(",", ""))
        last_step = st
        if loss != loss or loss > 1e6: alert("nan", f"loss is {m[2]} at step {st}")
        else:
            lows.append(loss); lows[:] = lows[-12:]
            if len(lows) > 6 and loss > min(lows) * 1.25 and st > 3000: alert("spike", f"loss spike at step {st}: {loss:.2f} vs recent best {min(lows):.2f}")
        if top1 > 40: alert("moe", f"expert collapse at step {st}: top-1 expert {top1:.0f}%")
        slow = slow + 1 if tps < 11000 else 0
        if slow >= 3: alert("slow", f"throughput {tps:,} tok/s for 3 logs at step {st}")
        if st >= TOTAL - 100: done = True
    # new checkpoints (size stable across polls => fully written)
    for ck in sorted(glob.glob(f"{D}/step_*.pt")):
        sz = os.path.getsize(ck)
        if ck in seen_ck: continue
        if sizes.get(ck) == sz and sz > 5e8:
            seen_ck.add(ck)
            r = subprocess.run([PY, "eval_ckpt.py", ck, D], capture_output=True, text=True)
            try:
                j = json.loads(r.stdout.strip().splitlines()[-1]); evals.append(j)
                say(f"EVAL step={j['step']} held-out_CE={j['ce']} (unigram {j['unigram']}) sample_repeat={j['sample_rep']}  text: {D}/samples_{j['step']:06d}.txt")
                if j["step"] >= 10000 and j["ce"] > 5.0: alert("ce", f"held-out CE {j['ce']} still above 5.5 at step {j['step']}")
                if len(evals) >= 3 and j["step"] >= 5000 and j["step"] < 0.8 * TOTAL and evals[-3]["ce"] - j["ce"] < 0.02: alert("stall", f"held-out CE improved <0.02 over last 2 checkpoints ({evals[-3]['ce']} -> {j['ce']})")
                if len(evals) >= 2 and j["ce"] > evals[-2]["ce"] + 0.05: alert("worse", f"held-out CE got worse: {evals[-2]['ce']} -> {j['ce']}")
                if j["sample_rep"] > 0.5 and j["step"] >= 7500: alert("rep", f"sampled text is {j['sample_rep']:.0%} repeated 4-grams at step {j['step']}")
            except Exception as e:
                alert("evalfail", f"checkpoint eval failed: {(r.stderr or str(e))[-200:]}")
        sizes[ck] = sz
    if not running() and last_step < TOTAL - 100:
        alert("died", f"training process is not running (last logged step {last_step} of {TOTAL})")
    if time.time() - last_beat > 1800:
        last_beat = time.time(); say(f"HEARTBEAT step={last_step} running={running()}")
say(f"WATCH done: training reached step {last_step}")
