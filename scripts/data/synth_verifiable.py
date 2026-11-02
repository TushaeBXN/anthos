"""Programmatic tasks with a checker, for the reward-conditioning building block.

Every example has a verifiable answer, so its grade comes from code, never from the model:
  correct answer  -> grade 4
  corrupted answer -> grade 0
Training on both with the grade tag in the prompt teaches the tag to mean something; at inference the prompt
asks for grade 4. At 74M parameters this is a plumbing test, not a capability claim.

    python synth_verifiable.py --n 3000 --out data/sft_synth_verifiable.jsonl --seed 0
"""
import argparse
import json
import random

WORDS = ("river stone cloud lamp garden window paper engine forest copper silver market bridge harbor "
         "meadow thunder candle ribbon orchard lantern compass summit valley anchor basket").split()


def make_task(rng: random.Random):
    kind = rng.choice(["add", "sub", "mul", "reverse", "sort", "count", "json", "upper"])
    if kind == "add":
        a, b = rng.randint(2, 999), rng.randint(2, 999)
        return f"What is {a} + {b}? Answer with just the number.", str(a + b), lambda r: _off(r, a + b, rng)
    if kind == "sub":
        a, b = rng.randint(50, 999), rng.randint(2, 49)
        return f"What is {a} - {b}? Answer with just the number.", str(a - b), lambda r: _off(r, a - b, rng)
    if kind == "mul":
        a, b = rng.randint(2, 30), rng.randint(2, 30)
        return f"What is {a} times {b}? Answer with just the number.", str(a * b), lambda r: _off(r, a * b, rng)
    w = rng.choice(WORDS)
    if kind == "reverse":
        return f"Reverse the word '{w}'. Answer with just the reversed word.", w[::-1], lambda r: _swap(r, rng)
    if kind == "upper":
        return f"Write the word '{w}' in capital letters. Answer with just the word.", w.upper(), lambda r: w.capitalize()
    lst = rng.sample(WORDS, 4)
    if kind == "sort":
        return (f"Sort these words alphabetically, separated by commas: {', '.join(lst)}.",
                ", ".join(sorted(lst)), lambda r: ", ".join(lst))
    if kind == "count":
        sent = " ".join(rng.choices(WORDS, k=rng.randint(3, 9)))
        n = len(sent.split())
        return f"How many words are in this sentence: \"{sent}\"? Answer with just the number.", str(n), lambda r: _off(r, n, rng)
    k, v = rng.choice(WORDS), rng.randint(1, 99)
    return (f"Return a JSON object with the key \"{k}\" set to {v}. Output only the JSON.",
            json.dumps({k: v}), lambda r: json.dumps({k: v + 1}))


def _off(right, val, rng):
    return str(val + rng.choice([-10, -2, -1, 1, 2, 10]))


def _swap(right, rng):
    i = rng.randrange(len(right) - 1)
    return right[:i] + right[i + 1] + right[i] + right[i + 2:] if right[i] != right[i + 1] else right[::-1][1:] + right[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/sft_synth_verifiable.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wrong-frac", type=float, default=0.25)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    with open(a.out, "w") as f:
        for _ in range(a.n):
            q, right, corrupt = make_task(rng)
            wrong = rng.random() < a.wrong_frac
            ans = corrupt(right) if wrong else right
            if wrong and ans == right:      # a corruption that happens to equal the answer is not a wrong example
                continue
            f.write(json.dumps({"conversations": [{"from": "human", "value": q}, {"from": "gpt", "value": ans}],
                                "grade": 0 if wrong else 4, "source": "synth_verifiable"}) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
