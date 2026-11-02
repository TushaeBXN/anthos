#!/usr/bin/env python3
"""
expand_code_dataset.py — Template multiplier for code SFT pairs.

Generates thousands of additional training pairs covering:
 - Many more code snippets with multiple question framings
 - Common error patterns across Python / JS / Bash
 - Step-by-step debugging walkthroughs
 - "How do I..." implementation questions
 - Data structures and algorithms
 - API design patterns
 - Machine learning & PyTorch cookbook

Output: data/code_expanded.jsonl
Usage:  python expand_code_dataset.py [--out data/code_expanded.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

random.seed(7)

SYSTEM = (
    "You are Anthos, an expert AI assistant created by Brian Tushae Thomas. "
    "You have deep knowledge of software engineering: reading code, fixing bugs, "
    "writing clean implementations, wiring modules together, and validating systems "
    "with smoke tests. You reason step-by-step, identify root causes before proposing "
    "fixes, and always verify your fix makes logical sense before presenting it."
)

READ_Q = [
    "What does this code do?\n\n```python\n{code}\n```",
    "Explain this Python code:\n\n```python\n{code}\n```",
    "Walk me through this code line by line:\n\n```python\n{code}\n```",
    "What is the purpose of this code?\n\n```python\n{code}\n```",
    "Can you explain what's happening in this snippet?\n\n```python\n{code}\n```",
]

BUG_Q = [
    "What's wrong with this code?\n\n```python\n{code}\n```",
    "Find the bug in this code:\n\n```python\n{code}\n```",
    "This code isn't working correctly. What's the problem?\n\n```python\n{code}\n```",
    "Review this code and identify any issues:\n\n```python\n{code}\n```",
]

FIX_Q = [
    "Fix this code. Problem: {problem}\n\n```python\n{code}\n```",
    "This code has a bug: {problem}. Show the corrected version.\n\n```python\n{code}\n```",
    "Rewrite this code to fix the following issue: {problem}\n\n```python\n{code}\n```",
]

GEN_Q = [
    "{spec}",
    "Write a Python function to {task}.",
    "Implement the following in Python: {spec}",
    "How do I {task} in Python?",
    "Give me a Python implementation of: {spec}",
]


def pair(human, gpt):
    return {"conversations": [
        {"from": "system",    "value": SYSTEM},
        {"from": "human",     "value": human.strip()},
        {"from": "gpt",       "value": gpt.strip()},
    ]}


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED CODE READING
# ─────────────────────────────────────────────────────────────────────────────

READING_BANK = [
    # (code, explanation)
    ("n = int(input())\nprint(sum(range(1, n+1)))",
     "Reads an integer `n` from stdin, then prints the sum 1+2+…+n using `range(1, n+1)` and `sum()`. O(n) space with range, O(1) with the formula, but Python's sum+range is fast enough for typical inputs."),

    ("stack = []\nfor ch in s:\n    if ch == '(':\n        stack.append(ch)\n    elif ch == ')' and stack:\n        stack.pop()\nprint(len(stack) == 0)",
     "Checks if parentheses in string `s` are balanced. Pushes `(` onto a stack; pops on `)`. Prints True if the stack is empty at the end (every open paren was closed). Doesn't handle `{[]}` — only `()`."),

    ("from itertools import product\ncombinations = list(product('ACGT', repeat=3))",
     "Generates all possible 3-character DNA codons using the 4 nucleotide bases (A, C, G, T). `itertools.product('ACGT', repeat=3)` produces the Cartesian product — 4³ = 64 tuples. Each tuple is one codon."),

    ("def clamp(value, lo, hi):\n    return max(lo, min(value, hi))",
     "Clamps `value` to the range [lo, hi]. `min(value, hi)` caps at the upper bound. `max(lo, ...)` floors at the lower bound. If value is in range, it's returned unchanged. Useful for keeping numbers within valid bounds (e.g. pixel values 0–255)."),

    ("seen = set()\nduplicates = [x for x in lst if x in seen or seen.add(x)]",
     "Finds duplicate values in `lst`. The `seen.add(x)` trick exploits that `set.add()` returns `None` (falsy), so the `or` always evaluates the condition. First occurrence: `x in seen` is False, `seen.add(x)` adds it and returns None (falsy) → not appended. Second occurrence: `x in seen` is True → appended. Clever but arguably unreadable."),

    ("import hashlib\nhash_val = hashlib.sha256(data.encode()).hexdigest()",
     "Computes the SHA-256 cryptographic hash of `data` (a string). `.encode()` converts to bytes (required by hashlib). `.hexdigest()` returns a 64-character lowercase hex string. SHA-256 is a one-way function — fast to compute, practically impossible to reverse."),

    ("from pathlib import Path\nfiles = list(Path('.').rglob('*.py'))",
     "Recursively finds all `.py` files in the current directory tree. `Path('.')` is the current directory. `.rglob('*.py')` walks all subdirectories matching the glob pattern. Returns a generator; `list()` materializes it."),

    ("q = collections.deque(maxlen=100)\nq.appendleft(item)",
     "A double-ended queue with a maximum length of 100. `appendleft` adds to the left end. When `maxlen` is reached, items are automatically dropped from the right end (FIFO eviction). Good for sliding windows, recent-N log buffers."),

    ("class Mixin:\n    def greet(self):\n        return f'Hello, I am {self.name}'",
     "A mixin class that provides a `greet()` method. Mixins don't stand alone — they're designed to be inherited alongside a main class that provides `self.name`. `class Dog(Animal, Mixin): ...` would give Dog instances a `greet()` method."),

    ("data.sort(key=lambda x: (x['priority'], -x['score']))",
     "Sorts `data` in-place by two keys: first by `priority` ascending, then by `score` descending (note the `-`). Python's sort is stable, and tuples compare element by element. Items with the same priority are ordered by highest score first."),

    ("env = os.environ.copy()\nenv['PYTHONPATH'] = '/custom/path'\nsubprocess.run(['python', 'script.py'], env=env)",
     "Creates a copy of the current environment variables, adds/overrides `PYTHONPATH`, then runs `script.py` with that modified environment. Doesn't affect the parent process's environment. Useful for running subprocesses with custom paths or secrets."),

    ("@property\ndef full_name(self):\n    return f'{self.first} {self.last}'",
     "A property that computes `full_name` on demand by joining `self.first` and `self.last`. Accessed as `obj.full_name` (no parentheses) like an attribute, but executes code. If `first` or `last` change, `full_name` reflects the update automatically."),

    ("model.requires_grad_(False)\nfor param in model.classifier.parameters():\n    param.requires_grad = True",
     "Freezes all model weights (disables gradient computation), then unfreezes only the `classifier` layer. This is the standard pattern for fine-tuning — the backbone is frozen (pretrained weights preserved), only the head trains. Drastically reduces compute and prevents catastrophic forgetting."),

    ("logits = logits - logits.max(dim=-1, keepdim=True).values",
     "Subtracts the maximum logit from each row (numerically stable softmax trick). The shift doesn't change the softmax result (they cancel out) but prevents overflow from `exp(large_number)`. `keepdim=True` preserves the dimension for broadcasting."),

    ("loss = loss / accumulation_steps\nloss.backward()",
     "Divides loss by the number of gradient accumulation steps before backprop. Without this, gradients would be `accumulation_steps` times too large when `optimizer.step()` finally fires. This makes effective batch = `batch_size × accumulation_steps` without the memory cost."),

    ("torch.manual_seed(42)\nnp.random.seed(42)\nrandom.seed(42)",
     "Sets random seeds for PyTorch, NumPy, and Python's random module to 42. This makes training reproducible — same seed → same random weight initialization and data shuffling order. Essential for debugging and comparing experiments."),

    ("hidden = torch.cat([h1, h2], dim=-1)",
     "Concatenates two tensors `h1` and `h2` along the last dimension. If both are `[batch, 512]`, result is `[batch, 1024]`. Common in attention mechanisms (concatenating query and context), multi-head attention outputs, and feature fusion."),

    ("scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)",
     "Creates a cosine annealing learning rate scheduler. Over `T_max=10000` steps, LR follows a cosine curve from the initial LR down to near zero. Smoother decay than step decay — avoids abrupt LR drops that can destabilize training."),

    ("with torch.cuda.amp.autocast(dtype=torch.bfloat16):\n    output = model(input_ids)",
     "Runs the forward pass in bfloat16 mixed precision. Weights stored in float32 (for stability) but matrix multiplications run in bfloat16 (faster, uses half the memory). `bfloat16` has the same exponent range as float32 (less overflow risk than float16)."),

    ("from contextlib import suppress\nwith suppress(FileNotFoundError):\n    os.remove('temp.txt')",
     "`contextlib.suppress` silently ignores the specified exception type. Equivalent to try/except with an empty except body, but more readable. Here, deletes `temp.txt` if it exists, does nothing if it doesn't."),

    ("tokens = [tok for tok in tokens if tok not in STOP_WORDS and len(tok) > 2]",
     "Filters a token list by removing stop words (common words like 'the', 'is') and very short tokens (≤ 2 characters). Typical NLP preprocessing step before TF-IDF, word2vec, or embedding-based search."),

    ("while True:\n    try:\n        conn = db.connect(timeout=5)\n        break\n    except ConnectionError:\n        time.sleep(2)",
     "Retry loop that keeps attempting a database connection until successful. `timeout=5` prevents hanging indefinitely on each attempt. `sleep(2)` backs off between retries to avoid hammering the server. No max retry count — can loop forever (consider adding a counter)."),

    ("pairs = list(zip(questions, answers))\nrandom.shuffle(pairs)\nquestions, answers = zip(*pairs)",
     "Shuffles `questions` and `answers` in tandem so they stay aligned. `zip` pairs them, `shuffle` randomizes, `zip(*pairs)` transposes back to two separate sequences. Classic technique for shuffling parallel sequences."),

    ("vocab = {word: idx for idx, word in enumerate(sorted(set(tokens)))}",
     "Builds a vocabulary dict mapping each unique token to an integer index. `set(tokens)` deduplicates, `sorted()` ensures deterministic ordering (same input → same vocab every run), `enumerate` provides the integer index. Foundation of any embedding layer."),

    ("n_tokens = sum(len(tok.encode()) for tok in texts)",
     "Counts total UTF-8 encoded bytes across all texts (as a proxy for token count). Not exact token count (depends on tokenizer), but a fast heuristic. For actual token count, use `tokenizer.encode(text)` and take `len()`."),

    ("class Config:\n    def __init__(self, **kwargs):\n        for k, v in kwargs.items():\n            setattr(self, k, v)",
     "A flexible config class built from keyword arguments. `setattr(self, k, v)` dynamically creates instance attributes. `Config(lr=3e-5, batch_size=8)` gives an object where `cfg.lr == 3e-5`. Simpler than argparse for internal configs."),

    ("assert shape[0] == batch_size, f'Expected batch {batch_size}, got {shape[0]}'",
     "An assertion that verifies the first tensor dimension matches the expected batch size. `assert` raises `AssertionError` with the message if the condition is False. Good for catching shape bugs early during development. Remove from production inference for speed (or use `--OO` flag)."),

    ("total = functools.reduce(lambda acc, x: acc + x['value'], items, 0)",
     "`functools.reduce` folds a list into a single value. Starts with accumulator=0, then for each item adds `x['value']` to the running total. Equivalent to `sum(x['value'] for x in items)`. `reduce` is more general but `sum()` is clearer for simple sums."),

    ("try:\n    idx = lst.index(target)\nexcept ValueError:\n    idx = -1",
     "`list.index(target)` raises `ValueError` if the target isn't found. This wraps it to return -1 instead (like C-style `indexOf`). Alternatively: `idx = lst.index(target) if target in lst else -1` (but checks twice); or `next((i for i, x in enumerate(lst) if x == target), -1)`."),

    ("result = next(filter(lambda x: x['id'] == target_id, records), None)",
     "Finds the first record where `id` matches `target_id`. `filter()` is lazy, `next()` stops at the first match. The second arg to `next()` is the default if nothing matches — returns `None` instead of raising `StopIteration`. Efficient: doesn't scan the whole list if the match is early."),
]


def make_reading_pairs():
    out = []
    for code, explanation in READING_BANK:
        for q_template in READ_Q:
            human = q_template.format(code=code)
            out.append(pair(human, explanation))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED BUG BANK
# ─────────────────────────────────────────────────────────────────────────────

BUG_BANK = [
    # (buggy_code, what_is_wrong, fixed_code, fix_explanation)
    (
        "total = 0\nfor i in range(1, 10):\n    total += i\nprint(total)",
        "The loop goes to 9, not 10 — `range(1, 10)` excludes 10.",
        "total = 0\nfor i in range(1, 11):\n    total += i\nprint(total)",
        "`range(1, 11)` includes 10. Total is now 55."
    ),
    (
        "if 0.1 + 0.2 == 0.3:\n    print('equal')\nelse:\n    print('not equal')",
        "Floating point comparison — `0.1 + 0.2` is `0.30000000000000004` in IEEE 754.",
        "import math\nif math.isclose(0.1 + 0.2, 0.3):\n    print('equal')\nelse:\n    print('not equal')",
        "Use `math.isclose()` (or check `abs(a-b) < 1e-9`) for float equality."
    ),
    (
        "items = {'a': 1, 'b': 2, 'c': 3}\nfor key in items:\n    if items[key] < 2:\n        del items[key]",
        "Mutating a dict while iterating raises `RuntimeError: dictionary changed size during iteration`.",
        "items = {'a': 1, 'b': 2, 'c': 3}\nitems = {k: v for k, v in items.items() if v >= 2}",
        "Dict comprehension builds a new dict — never mutates the original during iteration."
    ),
    (
        "class Node:\n    def __init__(self, val):\n        self.val = val\n        self.next = None\n\ndef append(head, val):\n    head.next = Node(val)",
        "If `head.next` already has a node, the chain is broken — the existing `.next` is overwritten.",
        "def append(head, val):\n    cur = head\n    while cur.next:\n        cur = cur.next\n    cur.next = Node(val)",
        "Walk to the tail of the linked list before setting `.next`."
    ),
    (
        "numbers = [1, 2, 3, 4, 5]\nhalf = numbers / 2",
        "TypeError — you can't divide a list by an integer. Lists don't support `/`.",
        "numbers = [1, 2, 3, 4, 5]\nhalf = numbers[:len(numbers)//2]",
        "Use slicing to get the first half of the list."
    ),
    (
        "import random\nprint(random.randint(1, 10))\nprint(random.randint(1, 10))\nprint(random.randint(1, 10))",
        "Not a bug exactly, but no seed — results are different every run (not reproducible).",
        "import random\nrandom.seed(42)\nprint(random.randint(1, 10))\nprint(random.randint(1, 10))\nprint(random.randint(1, 10))",
        "Setting `random.seed(42)` makes results reproducible across runs."
    ),
    (
        "def count_vowels(s):\n    return len([c for c in s if c in 'aeiou'])",
        "Misses uppercase vowels — 'A', 'E', 'I', 'O', 'U' are not counted.",
        "def count_vowels(s):\n    return len([c for c in s.lower() if c in 'aeiou'])",
        "`.lower()` normalizes the string so uppercase vowels are caught."
    ),
    (
        "text = 'hello world'\nprint(text[0] = 'H')",
        "SyntaxError — strings are immutable, you can't assign to an index with `=`. Also, this is inside `print()` which can't contain assignments.",
        "text = 'hello world'\ntext = 'H' + text[1:]\nprint(text)",
        "Build a new string by concatenating the replacement character with the rest."
    ),
    (
        "def factorial(n):\n    return n * factorial(n - 1)",
        "Missing base case — recursion never stops, causing `RecursionError` for any input.",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "Base case `n <= 1` returns 1, stopping the recursion."
    ),
    (
        "d = {'a': 1}\nprint(d['b'])",
        "KeyError — key `'b'` doesn't exist in the dict.",
        "d = {'a': 1}\nprint(d.get('b', 0))  # or: print(d.get('b'))",
        "`.get(key, default)` returns the default instead of raising KeyError."
    ),
    (
        "import torch\nmodel = torch.nn.Linear(10, 5)\nfor p in model.parameters():\n    p = p.detach()",
        "Reassigning loop variable `p` doesn't modify the actual parameter — `p` is a local reference.",
        "import torch\nmodel = torch.nn.Linear(10, 5)\nmodel.requires_grad_(False)",
        "Use `.requires_grad_(False)` to detach all parameters in-place."
    ),
    (
        "x = torch.randn(3, 4)\ny = x.numpy()\nx[0, 0] = 999\nprint(y[0, 0])",
        "`.numpy()` shares memory with the tensor — modifying `x` changes `y` too.",
        "x = torch.randn(3, 4)\ny = x.detach().clone().numpy()\nx[0, 0] = 999\nprint(y[0, 0])  # still original value",
        "`.clone()` creates a copy before `.numpy()` — the array is independent of the tensor."
    ),
    (
        "df = pd.read_csv('data.csv')\ndf.dropna()\nprint(df.shape)",
        "`dropna()` returns a new DataFrame — it doesn't modify `df` in-place.",
        "df = pd.read_csv('data.csv')\ndf = df.dropna()\nprint(df.shape)",
        "Assign the result back: `df = df.dropna()` (or use `inplace=True`)."
    ),
    (
        "response = requests.get(url)\ndata = json.loads(response.text)\nprint(data.name)",
        "`json.loads` returns a Python dict, not an object — use bracket access.",
        "response = requests.get(url)\ndata = json.loads(response.text)\nprint(data['name'])",
        "Dicts use `data['key']`, not `data.key`."
    ),
    (
        "with open('file.txt', 'r') as f:\n    pass\nlines = f.readlines()",
        "The file is closed after the `with` block exits — `f.readlines()` will raise ValueError.",
        "with open('file.txt', 'r') as f:\n    lines = f.readlines()\nprint(len(lines))",
        "Move `readlines()` inside the `with` block."
    ),
    (
        "for item in inventory:\n    inventory.remove(item)",
        "Modifying a list while iterating skips elements — every other item is skipped.",
        "inventory.clear()  # or: inventory = []",
        "If the goal is to empty the list, use `.clear()` or reassign to `[]`."
    ),
    (
        "class BankAccount:\n    balance = 0\n    def deposit(self, amount):\n        self.balance += amount",
        "`balance = 0` is a class variable — all account instances share the same balance.",
        "class BankAccount:\n    def __init__(self):\n        self.balance = 0\n    def deposit(self, amount):\n        self.balance += amount",
        "Move `balance` to `__init__` so each account gets its own independent balance."
    ),
    (
        "async def process():\n    results = [fetch(url) for url in urls]\n    return results",
        "Missing `await` — `fetch(url)` is a coroutine but not awaited, so `results` is a list of coroutine objects.",
        "async def process():\n    import asyncio\n    results = await asyncio.gather(*[fetch(url) for url in urls])\n    return results",
        "`asyncio.gather` runs all coroutines concurrently and awaits them all."
    ),
    (
        "n = input('Enter number: ')\nprint(n * 2)",
        "`input()` returns a string — `'5' * 2` is `'55'`, not `10`.",
        "n = int(input('Enter number: '))\nprint(n * 2)",
        "Wrap `input()` in `int()` to convert the string to an integer."
    ),
    (
        "matrix = [[0] * 3] * 3\nmatrix[0][0] = 1\nprint(matrix)",
        "All three rows are the SAME list object — `[[0]*3]*3` copies the reference. Changing `[0][0]` changes column 0 in all rows.",
        "matrix = [[0] * 3 for _ in range(3)]\nmatrix[0][0] = 1\nprint(matrix)",
        "List comprehension creates three independent lists."
    ),
]


def make_bug_pairs():
    out = []
    for buggy, what_wrong, fixed, fix_exp in BUG_BANK:
        # Identification pair
        human_id = random.choice(BUG_Q).format(code=buggy)
        gpt_id = f"**Bug:** {what_wrong}\n\n**Fixed:**\n```python\n{fixed}\n```\n\n{fix_exp}"
        out.append(pair(human_id, gpt_id))

        # Fix pair
        human_fix = random.choice(FIX_Q).format(problem=what_wrong, code=buggy)
        out.append(pair(human_fix, gpt_id))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# IMPLEMENTATION Q&A BANK
# ─────────────────────────────────────────────────────────────────────────────

HOWTO_BANK = [
    # (question, answer)
    (
        "How do I reverse a string in Python?",
        "```python\ns = 'hello'\nreversed_s = s[::-1]  # 'olleh'\n```\nThe slice `[::-1]` steps through the string backwards. Alternatively: `''.join(reversed(s))`."
    ),
    (
        "How do I check if a string is a palindrome?",
        "```python\ndef is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('A man a plan a canal Panama'))  # True\n```"
    ),
    (
        "How do I convert a list to a set and back in Python?",
        "```python\nlst = [1, 2, 2, 3, 3, 3]\nunique = set(lst)    # {1, 2, 3} — removes duplicates\nback_to_list = list(unique)  # [1, 2, 3] — order may vary\n```\nSets are unordered and have no duplicates. Conversion to list restores list interface but ordering is not guaranteed."
    ),
    (
        "How do I merge two dictionaries in Python?",
        "```python\na = {'x': 1, 'y': 2}\nb = {'y': 3, 'z': 4}\n\n# Python 3.9+ — dict union operator\nmerged = a | b  # {'x': 1, 'y': 3, 'z': 4}  — b overwrites a on conflicts\n\n# Python 3.5+\nmerged = {**a, **b}\n\n# In-place update\na.update(b)  # a is now merged, b wins on conflicts\n```"
    ),
    (
        "How do I iterate over a dictionary's keys and values at the same time?",
        "```python\nd = {'name': 'Alice', 'age': 30}\nfor key, value in d.items():\n    print(f'{key}: {value}')\n# name: Alice\n# age: 30\n```\n`.items()` returns `(key, value)` tuples. `.keys()` for keys only, `.values()` for values only."
    ),
    (
        "How do I read a file line by line in Python?",
        "```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    for line in f:\n        line = line.rstrip('\\n')\n        process(line)\n```\nIterating the file object reads lazily — memory-efficient for large files. `rstrip('\\n')` removes the newline character."
    ),
    (
        "How do I write JSON to a file in Python?",
        "```python\nimport json\n\ndata = {'name': 'Anthos', 'version': 1}\n\nwith open('output.json', 'w', encoding='utf-8') as f:\n    json.dump(data, f, indent=2, ensure_ascii=False)\n```\n`indent=2` formats it readably. `ensure_ascii=False` preserves Unicode characters."
    ),
    (
        "How do I get the current timestamp in Python?",
        "```python\nfrom datetime import datetime\n\nnow = datetime.now()\nprint(now.isoformat())          # '2026-09-08T14:32:01.123456'\nprint(now.strftime('%Y-%m-%d')) # '2026-09-08'\n\n# Unix timestamp (seconds since epoch)\nimport time\nts = time.time()  # 1757347921.456\n```"
    ),
    (
        "How do I remove duplicates from a list while preserving order?",
        "```python\ndef dedupe(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]\n\n# Example\ndedupe([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])  # [3, 1, 4, 5, 9, 2, 6]\n```\nThe `seen.add(x)` trick adds to the set and returns `None` (falsy). First-seen items pass the filter; duplicates don't."
    ),
    (
        "How do I sort a list of objects by multiple attributes?",
        "```python\npeople = [\n    {'name': 'Alice', 'age': 30},\n    {'name': 'Bob', 'age': 25},\n    {'name': 'Carol', 'age': 30},\n]\n\n# Sort by age ASC, then name ASC\nsorted_people = sorted(people, key=lambda p: (p['age'], p['name']))\n# → Bob, Alice, Carol\n\n# Sort by age DESC, then name ASC\nsorted_people = sorted(people, key=lambda p: (-p['age'], p['name']))\n```"
    ),
    (
        "How do I check if a key exists in a dictionary?",
        "```python\nd = {'a': 1, 'b': None}\n\n# Correct\n'a' in d          # True\n'c' in d          # False\n\n# Also correct (but reads the value)\nd.get('a') is not None   # True for 'a', False for 'b' and 'c' (but misleading for None values)\n\n# Best for existence check:\n'a' in d  # always use this\n```"
    ),
    (
        "How do I use enumerate in Python?",
        "```python\nfruits = ['apple', 'banana', 'cherry']\n\nfor i, fruit in enumerate(fruits):\n    print(f'{i}: {fruit}')\n# 0: apple\n# 1: banana\n# 2: cherry\n\n# Start from 1\nfor i, fruit in enumerate(fruits, start=1):\n    print(f'{i}: {fruit}')\n# 1: apple  ...\n```\nAlways prefer `enumerate` over `range(len(lst))` — cleaner and less error-prone."
    ),
    (
        "How do I handle exceptions gracefully in Python?",
        "```python\ntry:\n    result = risky_operation()\nexcept ValueError as e:\n    print(f'Bad value: {e}')\n    result = default_value\nexcept (IOError, OSError) as e:\n    print(f'I/O error: {e}')\n    raise  # re-raise if you can't recover\nexcept Exception as e:\n    # Catch-all — use sparingly\n    logging.error(f'Unexpected error: {e}', exc_info=True)\nelse:\n    # Runs only if no exception was raised\n    save_result(result)\nfinally:\n    # Always runs — cleanup\n    cleanup()\n```"
    ),
    (
        "How do I use context managers to manage resources?",
        "```python\n# Built-in context managers\nwith open('file.txt') as f:\n    data = f.read()  # file auto-closed after block\n\nwith db.connection() as conn:\n    conn.execute('...')\n\n# Custom context manager\nfrom contextlib import contextmanager\n\n@contextmanager\ndef managed_resource():\n    resource = acquire()\n    try:\n        yield resource\n    finally:\n        release(resource)\n\nwith managed_resource() as r:\n    use(r)\n```\nThe `with` statement guarantees cleanup even if an exception occurs."
    ),
    (
        "How do I profile Python code to find bottlenecks?",
        "```python\n# Quick: time a block\nimport time\nt0 = time.perf_counter()\nexpensive_function()\nprint(f'{time.perf_counter() - t0:.4f}s')\n\n# Full function profiler\nimport cProfile\ncProfile.run('main()', sort='cumulative')\n\n# Line-by-line (install line_profiler)\n# @profile decorator on functions, then: kernprof -l script.py\n\n# Memory\n# pip install memory_profiler\n# @profile decorator, then: python -m memory_profiler script.py\n```"
    ),
    (
        "How do I create a virtual environment and install packages?",
        "```bash\n# Create\npython3 -m venv venv\n\n# Activate (Mac/Linux)\nsource venv/bin/activate\n\n# Activate (Windows)\nvenv\\\\Scripts\\\\activate\n\n# Install\npip install requests numpy pandas\n\n# Save dependencies\npip freeze > requirements.txt\n\n# Install from file\npip install -r requirements.txt\n\n# Deactivate\ndeactivate\n```\nAlways work inside a venv — isolates project dependencies from the system Python."
    ),
    (
        "How do I make an HTTP request in Python?",
        "```python\nimport requests\n\n# GET\nresponse = requests.get('https://api.example.com/users', timeout=10)\nresponse.raise_for_status()  # raises HTTPError if 4xx/5xx\ndata = response.json()\n\n# POST with JSON body\nresponse = requests.post(\n    'https://api.example.com/users',\n    json={'name': 'Alice', 'email': 'alice@example.com'},\n    headers={'Authorization': f'Bearer {token}'},\n    timeout=10,\n)\nresponse.raise_for_status()\nnew_user = response.json()\n```\nAlways set `timeout` — without it, requests can hang forever."
    ),
    (
        "How do I use type hints in Python?",
        "```python\nfrom typing import Optional, List, Dict, Tuple, Union\n\ndef greet(name: str, age: int = 0) -> str:\n    return f'Hello, {name}!'\n\ndef process(items: List[int]) -> Dict[str, int]:\n    return {'sum': sum(items), 'count': len(items)}\n\ndef find_user(user_id: int) -> Optional[Dict]:\n    # Returns a dict or None\n    ...\n\n# Python 3.10+ — union with |\ndef parse(value: int | str) -> int:\n    return int(value)\n```\nType hints don't enforce types at runtime (use mypy for static checking) but make code self-documenting and enable IDE autocomplete."
    ),
    (
        "How do I use dataclasses in Python?",
        "```python\nfrom dataclasses import dataclass, field\nfrom typing import List\n\n@dataclass\nclass TrainConfig:\n    lr: float = 3e-5\n    batch_size: int = 8\n    max_steps: int = 10_000\n    seq_len: int = 512\n    tags: List[str] = field(default_factory=list)\n\ncfg = TrainConfig(lr=1e-4, batch_size=16)\nprint(cfg)  # TrainConfig(lr=0.0001, batch_size=16, max_steps=10000, ...)\nprint(cfg.lr)  # 0.0001\n```\nDataclasses auto-generate `__init__`, `__repr__`, and `__eq__`. Use `field(default_factory=list)` for mutable defaults."
    ),
    (
        "How do I implement a simple LRU cache?",
        "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cap = capacity\n        self.cache = OrderedDict()\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)  # mark as recently used\n        return self.cache[key]\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)  # evict LRU (first item)\n\n# Or just use functools.lru_cache for functions:\nfrom functools import lru_cache\n@lru_cache(maxsize=128)\ndef expensive(n): ...\n```"
    ),
    (
        "How do I implement binary search in Python?",
        "```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n\n# Built-in (returns insertion point, not -1):\nimport bisect\nidx = bisect.bisect_left(arr, target)\nfound = idx < len(arr) and arr[idx] == target\n```\nBinary search requires a sorted array. O(log n) time."
    ),
    (
        "How do I implement a graph and do BFS in Python?",
        "```python\nfrom collections import deque\n\ndef bfs(graph, start):\n    visited = set()\n    queue = deque([start])\n    visited.add(start)\n    order = []\n\n    while queue:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph[node]:\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return order\n\n# Adjacency list\ngraph = {\n    'A': ['B', 'C'],\n    'B': ['D'],\n    'C': ['D', 'E'],\n    'D': [],\n    'E': [],\n}\nprint(bfs(graph, 'A'))  # ['A', 'B', 'C', 'D', 'E']\n```"
    ),
    (
        "How do I implement a simple tokenizer for a chatbot?",
        "```python\nimport re\n\ndef tokenize(text):\n    # Split on whitespace and punctuation, keep punctuation as tokens\n    tokens = re.findall(r\"\\w+|[^\\w\\s]\", text.lower())\n    return tokens\n\ndef build_vocab(texts, min_freq=2):\n    from collections import Counter\n    all_tokens = [tok for text in texts for tok in tokenize(text)]\n    counts = Counter(all_tokens)\n    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}\n    for tok, freq in counts.most_common():\n        if freq >= min_freq:\n            vocab[tok] = len(vocab)\n    return vocab\n\ndef encode(text, vocab):\n    return [vocab.get(tok, vocab['<UNK>']) for tok in tokenize(text)]\n```"
    ),
    (
        "How do I implement gradient accumulation in PyTorch?",
        "```python\naccum_steps = 4  # effective batch = batch_size * accum_steps\noptimizer.zero_grad()\n\nfor step, (ids, labels) in enumerate(loader):\n    logits = model(ids)\n    loss = criterion(logits, labels) / accum_steps  # scale down\n    loss.backward()  # accumulate gradients\n\n    if (step + 1) % accum_steps == 0:\n        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n        optimizer.step()\n        optimizer.zero_grad()\n```\nDividing loss by `accum_steps` keeps the effective gradient magnitude the same as a true large batch."
    ),
    (
        "How do I save and load a model in PyTorch?",
        "```python\nimport torch\n\n# Save — always save state_dict, not the full model object\ntorch.save({\n    'model': model.state_dict(),\n    'optimizer': optimizer.state_dict(),\n    'step': current_step,\n    'loss': loss.item(),\n}, 'checkpoint.pt')\n\n# Load\nckpt = torch.load('checkpoint.pt', map_location='cpu')\nmodel.load_state_dict(ckpt['model'])\noptimizer.load_state_dict(ckpt['optimizer'])\nstep = ckpt['step']\n```\nAlways `map_location='cpu'` when loading — safe regardless of whether checkpoint was saved on GPU."
    ),
    (
        "How do I create a custom PyTorch Dataset?",
        "```python\nfrom torch.utils.data import Dataset\nimport torch\n\nclass ConversationDataset(Dataset):\n    def __init__(self, data, tokenizer, max_len=512):\n        self.data = data\n        self.tokenizer = tokenizer\n        self.max_len = max_len\n\n    def __len__(self):\n        return len(self.data)\n\n    def __getitem__(self, idx):\n        item = self.data[idx]\n        ids = self.tokenizer.encode(\n            item['text'], max_length=self.max_len,\n            truncation=True, padding='max_length'\n        )\n        input_ids = torch.tensor(ids[:-1])   # input: all but last\n        labels    = torch.tensor(ids[1:])    # target: all but first\n        return input_ids, labels\n```\nMust implement `__len__` and `__getitem__`. The DataLoader handles batching."
    ),
    (
        "How do I add learning rate warmup to a PyTorch optimizer?",
        "```python\nfrom torch.optim.lr_scheduler import LambdaLR\n\nwarmup_steps = 500\n\ndef lr_lambda(step):\n    if step < warmup_steps:\n        return step / warmup_steps  # linear warmup: 0 → 1\n    return 1.0  # constant after warmup (or add decay here)\n\nscheduler = LambdaLR(optimizer, lr_lambda)\n\n# In training loop:\nscheduler.step()\n```\nWarmup prevents large gradient updates from bad initial random weights. Combine with cosine decay after warmup for best results."
    ),
    (
        "How do I implement a simple REST API with FastAPI?",
        "```python\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Message(BaseModel):\n    text: str\n    user_id: str = 'anonymous'\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}\n\n@app.post('/chat')\ndef chat(msg: Message):\n    if not msg.text.strip():\n        raise HTTPException(status_code=400, detail='Empty message')\n    response = model.generate(msg.text)\n    return {'response': response, 'user_id': msg.user_id}\n\n# Run: uvicorn main:app --reload\n```\nPydantic models auto-validate request bodies. FastAPI auto-generates `/docs` (Swagger UI)."
    ),
]


def make_howto_pairs():
    out = []
    for question, answer in HOWTO_BANK:
        out.append(pair(question, answer))
        # Add a variation
        if question.startswith("How do I"):
            alt = question.replace("How do I", "What's the best way to", 1)
            out.append(pair(alt, answer))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP-BY-STEP DEBUGGING WALKTHROUGHS
# ─────────────────────────────────────────────────────────────────────────────

WALKTHROUGH_BANK = [
    (
        "I'm getting `TypeError: unsupported operand type(s) for +: 'int' and 'str'`. How do I debug this?",
        """**Step 1 — Read the traceback:**
Find the line number in the error. The traceback shows exactly where it failed.

**Step 2 — Identify the types:**
```python
a = 5
b = '3'
print(a + b)  # TypeError here
```
Python won't auto-convert — you must be explicit.

**Step 3 — Fix:**
```python
# Option A — convert b to int
print(a + int(b))   # 8

# Option B — convert a to str
print(str(a) + b)   # '53'
```

**Step 4 — Find the root cause:**
Ask: where did `b` come from? `input()`, `json.loads()`, and `csv` always return strings. Add type conversion at the boundary:
```python
b = int(input('Enter number: '))  # convert on intake
```

**Prevention:** use type hints and verify at function entry points."""
    ),
    (
        "My for loop isn't doing what I expect. How do I debug it step by step?",
        """**Step 1 — Add print statements inside the loop:**
```python
for i, item in enumerate(items):
    print(f'Step {i}: item={item!r}')
    # your logic
    print(f'  → result: {result!r}')
```

**Step 2 — Check the range:**
```python
print(f'Total iterations: {len(items)}')
print(f'First item: {items[0]!r}')
print(f'Last item: {items[-1]!r}')
```

**Step 3 — Use Python's debugger (pdb):**
```python
import pdb; pdb.set_trace()
# Drops you into an interactive prompt inside the loop
# Commands: n (next), s (step), c (continue), p <var> (print), q (quit)
```

**Step 4 — Check off-by-one errors:**
```python
# Inclusive range? range(1, n+1) not range(1, n)
# Zero-indexed? Remember: last valid index is len(lst)-1
```

**Step 5 — Verify the data:**
```python
assert len(items) > 0, 'items is empty!'
print(type(items[0]))  # make sure items contain what you expect
```"""
    ),
    (
        "My Python script runs fine locally but fails in production. How do I debug this?",
        """**Common causes and how to isolate each:**

**1. Different Python versions:**
```bash
python --version  # check both environments
```

**2. Missing dependencies:**
```bash
pip freeze > requirements.txt   # capture exact versions locally
pip install -r requirements.txt # install same versions in prod
```

**3. Missing environment variables:**
```bash
# Prod server — check all required vars are set
echo $DATABASE_URL
echo $SECRET_KEY
```

**4. File paths:**
Local: `/Users/brian/project/data/file.json`
Production: `/app/data/file.json`
Fix: use `Path(__file__).parent / 'data' / 'file.json'` — relative to the script location.

**5. Data differences:**
Log what the prod data actually looks like:
```python
import logging
logging.info(f'Input type: {type(data)}, value: {data!r}')
```

**6. Permissions:**
Production servers often restrict file writes. Check: `ls -la /path/to/output/`

**Systematic approach:**
1. Add detailed logging to the failing code
2. Deploy a debug version
3. Reproduce the exact production data locally
4. Compare environments: Python version, packages, env vars, file system"""
    ),
    (
        "I get `ImportError` when running my script. Walk me through fixing it.",
        """**Step 1 — Read the full error:**
```
ImportError: cannot import name 'MyClass' from 'mymodule'
```
This tells you: `MyClass` doesn't exist in `mymodule`.

**Step 2 — Check the module file:**
Open `mymodule.py` and verify `MyClass` is defined and spelled correctly (case-sensitive).

**Step 3 — Check circular imports:**
If `module_a` imports `module_b` and `module_b` imports `module_a`, Python may fail with ImportError.
Fix: move the import inside the function that needs it:
```python
def process():
    from module_b import helper  # lazy import
    return helper()
```

**Step 4 — Check `__init__.py`:**
If `mymodule` is a package directory, verify `MyClass` is exported from `mymodule/__init__.py`:
```python
# mymodule/__init__.py
from .core import MyClass
```

**Step 5 — Check installation:**
```bash
pip show mypackage   # verify it's installed
python -c "import mypackage; print(mypackage.__file__)"
```

**Step 6 — Check PYTHONPATH:**
```bash
python -c "import sys; print('\\n'.join(sys.path))"
# Verify your project root is in the list
```"""
    ),
]


def make_walkthrough_pairs():
    out = []
    for question, answer in WALKTHROUGH_BANK:
        out.append(pair(question, answer))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

DS_BANK = [
    ("When should I use a list vs a tuple in Python?",
     "**List** — mutable, use when the collection will change: append, remove, sort.\n**Tuple** — immutable, use when the collection is fixed: function return values, dict keys, named record.\n\nKey differences:\n- Tuples are hashable (can be dict keys or set members): `{(1,2): 'point'}`\n- Tuples are slightly faster to create and iterate\n- Tuples signal intent: 'this won't change'\n- Tuples unpack naturally: `x, y = point`"),

    ("When should I use a dict vs a set in Python?",
     "**Dict** — key→value mapping. Use when you need to look up a value by a key.\n**Set** — unordered collection of unique values. Use when you only need membership testing or deduplication.\n\n```python\n# Dict: word → count\ncounts = {'hello': 3, 'world': 1}\n\n# Set: seen words (no associated value)\nseen = {'hello', 'world'}\n'hello' in seen  # O(1)\n```\nSets are backed by a hash table — O(1) membership test, same as dict key lookup."),

    ("What is the time complexity of common list operations in Python?",
     "| Operation | Average Case |\n|-----------|-------------|\n| `lst[i]` | O(1) |\n| `lst.append(x)` | O(1) amortized |\n| `lst.insert(i, x)` | O(n) — shifts elements |\n| `x in lst` | O(n) — linear scan |\n| `lst.pop()` | O(1) — removes last |\n| `lst.pop(i)` | O(n) — shifts elements |\n| `len(lst)` | O(1) |\n| `lst.sort()` | O(n log n) |\n\nFor fast membership testing, use a `set` (O(1)) instead of a list (O(n))."),

    ("What is the difference between a stack and a queue?",
     "**Stack** — Last In, First Out (LIFO). Push to top, pop from top.\n```python\nstack = []\nstack.append(1)  # push\nstack.pop()      # pop from top\n```\nUse for: function call stack, undo/redo, DFS.\n\n**Queue** — First In, First Out (FIFO). Enqueue at back, dequeue from front.\n```python\nfrom collections import deque\nq = deque()\nq.append(1)      # enqueue\nq.popleft()      # dequeue from front — O(1) with deque\n```\nUse for: BFS, task scheduling, producer-consumer."),

    ("What is a defaultdict and when should I use it?",
     "A `defaultdict` is a dict subclass that creates a default value automatically for missing keys.\n\n```python\nfrom collections import defaultdict\n\n# Frequency count\ncounts = defaultdict(int)  # missing key → 0\nfor word in words:\n    counts[word] += 1      # no KeyError on first occurrence\n\n# Grouping\ngroups = defaultdict(list)\nfor item in items:\n    groups[item['category']].append(item)\n\n# Nested dicts\nd = defaultdict(lambda: defaultdict(int))\nd['section']['subsection'] += 1\n```\nUse it when you find yourself writing `d.setdefault(key, []).append(x)` — `defaultdict(list)` is cleaner."),

    ("What is a heap and how do I use one in Python?",
     "A heap is a tree where every parent ≤ its children (min-heap). Python's `heapq` is a min-heap.\n\n```python\nimport heapq\n\nnums = [5, 1, 8, 3, 2]\nheapq.heapify(nums)       # in-place: O(n)\nsmallest = heapq.heappop(nums)  # O(log n)\nheapq.heappush(nums, 4)   # O(log n)\n\n# N smallest/largest\ntop3 = heapq.nsmallest(3, nums)\n\n# Max-heap — negate values\nheapq.heappush(max_heap, -value)\nmax_val = -heapq.heappop(max_heap)\n```\nUse for: priority queues, Dijkstra's algorithm, top-N queries."),
]


def make_ds_pairs():
    out = []
    for question, answer in DS_BANK:
        out.append(pair(question, answer))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/code_expanded.jsonl')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('Generating expanded code SFT dataset...')
    all_pairs = []

    generators = [
        ('code reading (extended)', make_reading_pairs),
        ('bug identification/fix (extended)', make_bug_pairs),
        ('how-to implementations', make_howto_pairs),
        ('debugging walkthroughs', make_walkthrough_pairs),
        ('data structures', make_ds_pairs),
    ]

    for name, fn in generators:
        batch = fn()
        all_pairs.extend(batch)
        print(f'  {name}: {len(batch)} pairs')

    random.shuffle(all_pairs)

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    size_mb = out_path.stat().st_size / 1e6
    print(f'\nWrote {len(all_pairs):,} pairs → {out_path} ({size_mb:.1f} MB)')
    print('\nTo add to master:')
    print(f'  cat {out_path} >> data/sft_master.jsonl')


if __name__ == '__main__':
    main()
