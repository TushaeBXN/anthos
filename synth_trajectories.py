"""Multi-step tool-use trajectories with a checker, for the finish-the-job habit.

A tiny simulated environment (calculator, price store, notes) runs every trajectory, and code, not the model,
decides the grade:
  4  the right steps, the right answer, a truthful status (<|done|>, or <|blocked|> when the data really is missing)
  0  a stall (same call repeated), a lazy quit (gives up or claims done too early), a wrong answer, or a false claim
Bad trajectories are kept on purpose: with the grade tag in the prompt the model learns what a stall looks like and
that grade 4 means the other behaviour. At inference the prompt asks for grade 4.

    python synth_trajectories.py --n 6000 --out data/sft_synth_traj.jsonl
    (also imported by eval_agent.py for the environment and checker)
"""
import argparse
import json
import random
import re

PRICES = {"apple": 3, "pear": 4, "notebook": 12, "lamp": 25, "mug": 8, "cable": 6, "book": 15, "pen": 2,
          "chair": 40, "clock": 18, "plant": 9, "towel": 7, "bottle": 5, "candle": 11, "frame": 14}
MISSING = ["gizmo", "sprocket", "widget", "trinket", "doohickey"]
SYSTEM = "You are Anthos. You can call tools: calc(expr), lookup(key), write_note(text). Finish the task or say why you cannot."


class Env:
    def __init__(self):
        self.notes = []

    def run(self, name, args):
        if name == "calc":
            e = str(args.get("expr", ""))
            if not re.fullmatch(r"[0-9+\-*/(). ]+", e):
                return "error: bad expression"
            try:
                v = eval(e, {"__builtins__": {}}, {})
                return str(int(v)) if float(v).is_integer() else str(round(v, 4))
            except Exception:
                return "error: bad expression"
        if name == "lookup":
            k = str(args.get("key", "")).replace("price:", "")
            return str(PRICES[k]) if k in PRICES else "not found"
        if name == "write_note":
            self.notes.append(str(args.get("text", "")))
            return "saved"
        return "error: unknown tool"


def call(name, **args):
    return {"name": name, "args": args}


def _rec(task, steps, final, status, grade, kind):
    env = Env()
    return {"task": task, "steps": [{"call": c, "result": env.run(c["name"], c["args"])} for c in steps],
            "final": final, "status": status, "grade": grade, "kind": kind, "source": "synth_traj"}


def good(rng):
    k = rng.choice(["chain", "price", "blocked", "note"])
    if k == "chain":
        a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 12)
        s = a + b
        return _rec(f"What is ({a} + {b}) * {c}? Use the calculator.",
                    [call("calc", expr=f"{a}+{b}"), call("calc", expr=f"{s}*{c}")],
                    f"The answer is {s * c}.", "done", 4, "chain")
    if k == "price":
        it, n = rng.choice(list(PRICES)), rng.randint(2, 9)
        return _rec(f"How much do {n} {it}s cost? Look up the price first.",
                    [call("lookup", key=it), call("calc", expr=f"{PRICES[it]}*{n}")],
                    f"{n} {it}s cost {PRICES[it] * n}.", "done", 4, "price")
    if k == "blocked":
        it = rng.choice(MISSING)
        return _rec(f"How much do 3 {it}s cost? Look up the price first.", [call("lookup", key=it)],
                    f"I could not find a price for {it}, so I cannot work out the cost.", "blocked", 4, "blocked")
    text = rng.choice(["buy milk", "call the dentist", "water the plants", "send the report", "renew the passport"])
    return _rec(f"Save a note that says: {text}", [call("write_note", text=text)], "I saved the note.", "done", 4, "note")


def bad(rng):
    k = rng.choice(["stall", "lazy_quit", "wrong", "false_claim"])
    it, n = rng.choice(list(PRICES)), rng.randint(2, 9)
    task = f"How much do {n} {it}s cost? Look up the price first."
    if k == "stall":      # the same call again and again, then a made-up answer
        return _rec(task, [call("lookup", key=it)] * 3, f"{n} {it}s cost {PRICES[it] * n + rng.randint(1, 9)}.", "done", 0, k)
    if k == "lazy_quit":  # gives up before the work is done
        return _rec(task, [call("lookup", key=it)], "I looked into it and it should be fine.", rng.choice(["done", "blocked"]), 0, k)
    if k == "wrong":
        return _rec(task, [call("lookup", key=it), call("calc", expr=f"{PRICES[it]}*{n}")],
                    f"{n} {it}s cost {PRICES[it] * n + rng.choice([-2, -1, 1, 2])}.", "done", 0, k)
    return _rec(task, [], f"I have checked the price and {n} {it}s cost {PRICES[it] * n}.", "done", 0, k)  # claims work it never did


def check(rec) -> int:
    """Grade from the record alone: re-runs the tools, compares the answer, checks the status is honest."""
    env, calls, results = Env(), rec["steps"], []
    for s in calls:
        results.append(env.run(s["call"]["name"], s["call"]["args"]))
    keys = [json.dumps(s["call"], sort_keys=True) for s in calls]
    if len(keys) != len(set(keys)):
        return 0                                   # repeated call = stall
    m = re.search(r"How much do (\d+) (\w+?)s cost", rec["task"])
    if m:
        n, it = int(m.group(1)), m.group(2)
        if it in MISSING:
            return 4 if rec["status"] == "blocked" and results[:1] == ["not found"] else 0
        want = str(PRICES[it] * n)
        return 4 if (len(calls) == 2 and rec["status"] == "done" and want in rec["final"]) else 0
    m = re.search(r"\((\d+) \+ (\d+)\) \* (\d+)", rec["task"])
    if m:
        a, b, c = map(int, m.groups())
        return 4 if (len(calls) == 2 and rec["status"] == "done" and str((a + b) * c) in rec["final"]) else 0
    if rec["task"].startswith("Save a note"):
        return 4 if (len(calls) == 1 and results == ["saved"] and rec["status"] == "done") else 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="data/sft_synth_traj.jsonl")
    ap.add_argument("--bad-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    mismatch = 0
    with open(a.out, "w") as f:
        for _ in range(a.n):
            r = bad(rng) if rng.random() < a.bad_frac else good(rng)
            if check(r) != r["grade"]:             # the generator and the checker must agree on every record
                mismatch += 1
                continue
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}; generator/checker disagreements dropped: {mismatch}")


if __name__ == "__main__":
    main()
