#!/usr/bin/env python3
"""
generate_code_dataset.py — Code Intelligence SFT dataset for Anthos

Generates ~15,000 high-quality training pairs covering:
  1. Code reading / understanding
  2. Bug identification & fixing
  3. Code generation from spec
  4. Integration & wiring (connecting modules)
  5. Smoke testing & validation
  6. Refactoring
  7. Code review
  8. Debugging process (error → root cause → fix)
  9. Anthos-specific patterns (train.py, preflight, configs)

Output: data/code_sft.jsonl
Usage:  python generate_code_dataset.py [--out data/code_sft.jsonl]
"""

import argparse
import json
import random
import sys
from pathlib import Path

random.seed(42)

SYSTEM = (
    "You are Anthos, an expert AI assistant created by Brian Tushae Thomas. "
    "You have deep knowledge of software engineering: reading code, fixing bugs, "
    "writing clean implementations, wiring modules together, and validating systems "
    "with smoke tests. You reason step-by-step, identify root causes before proposing "
    "fixes, and always verify your fix makes logical sense before presenting it."
)


def pair(human: str, gpt: str) -> dict:
    return {
        "conversations": [
            {"from": "system",  "value": SYSTEM},
            {"from": "human",   "value": human.strip()},
            {"from": "gpt",     "value": gpt.strip()},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. CODE READING / UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

def code_reading_pairs():
    samples = []

    snippets = [
        (
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "This is a recursive Fibonacci function. Given `n`, it returns the nth Fibonacci number. "
            "Base cases: `n=0` returns 0, `n=1` returns 1. For larger `n`, it recursively sums the two "
            "preceding Fibonacci numbers. Time complexity is O(2^n) — exponential — because it recomputes "
            "subtrees. For large inputs, prefer memoization or an iterative approach."
        ),
        (
            "result = [x**2 for x in range(10) if x % 2 == 0]",
            "This list comprehension builds a list of squares of even numbers from 0 to 9. "
            "`range(10)` produces 0–9. The `if x % 2 == 0` filter keeps only even values (0, 2, 4, 6, 8). "
            "`x**2` squares each. Result: `[0, 4, 16, 36, 64]`."
        ),
        (
            "with open('data.txt', 'r') as f:\n    lines = [line.strip() for line in f if line.strip()]",
            "Opens `data.txt` in read mode using a context manager (so the file is auto-closed). "
            "Reads every line, strips leading/trailing whitespace, and keeps only non-empty lines. "
            "Result is a list of cleaned, non-blank strings."
        ),
        (
            "from functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)",
            "Same Fibonacci logic as naive recursion, but decorated with `@lru_cache`. This memoizes "
            "results so each `fib(k)` is computed only once. `maxsize=None` means unlimited cache. "
            "Time complexity drops from O(2^n) to O(n), space is O(n) for the cache."
        ),
        (
            "try:\n    result = 10 / x\nexcept ZeroDivisionError:\n    result = float('inf')",
            "Attempts to divide 10 by `x`. If `x` is zero, a `ZeroDivisionError` would normally crash "
            "the program. The `except` block catches it and assigns `inf` (positive infinity) instead. "
            "This is a safe division pattern — useful when zero denominators are expected input."
        ),
        (
            "d = {}\nfor word in text.split():\n    d[word] = d.get(word, 0) + 1",
            "Counts word frequencies in `text`. `text.split()` tokenizes on whitespace. For each word, "
            "`d.get(word, 0)` returns the current count or 0 if unseen, then increments by 1. "
            "After the loop, `d` is a frequency map `{word: count, ...}`."
        ),
        (
            "sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)",
            "Sorts `items` (a list of dicts) by the `'score'` key in descending order. "
            "`key=lambda x: x['score']` extracts the sort key from each dict. `reverse=True` sorts "
            "highest-to-lowest. `sorted()` returns a new list; original `items` is unchanged."
        ),
        (
            "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
            "Implements the Singleton design pattern. `_instance` is a class-level variable (shared across "
            "all instances). `__new__` is called before `__init__`. If `_instance` is None, it creates one "
            "instance via `super().__new__(cls)` and stores it. On every subsequent call, the existing "
            "instance is returned. This guarantees only one object of this class exists."
        ),
        (
            "import asyncio\n\nasync def fetch(url):\n    await asyncio.sleep(1)\n    return f'data from {url}'",
            "Defines an async coroutine `fetch`. The `async def` makes it awaitable. `await asyncio.sleep(1)` "
            "suspends execution for 1 second without blocking the event loop (other coroutines can run). "
            "Returns a mock string. In real use, you'd replace `sleep` with an async HTTP call like `aiohttp.get(url)`."
        ),
        (
            "nums = [3, 1, 4, 1, 5, 9, 2, 6]\nprint(nums[::2])\nprint(nums[::-1])",
            "`nums[::2]` slices with step 2 — every other element starting at index 0: `[3, 4, 5, 2]`. "
            "`nums[::-1]` reverses the list by stepping -1: `[6, 2, 9, 5, 1, 4, 1, 3]`."
        ),
        (
            "if __name__ == '__main__':\n    main()",
            "This guard means: only run `main()` if this script is executed directly (e.g. `python script.py`). "
            "If the file is imported as a module by another script, `__name__` is the module name, not `'__main__'`, "
            "so `main()` is not called. Best practice for any Python script that defines a main entry point."
        ),
        (
            "data = json.loads(response.text)\nusers = data.get('users', [])\nactive = [u for u in users if u.get('active')]",
            "Parses a JSON string from an HTTP response body into a Python dict. Extracts the `'users'` key "
            "(defaulting to `[]` if absent). Filters to users where `'active'` is truthy. "
            "Result: a list of active user dicts."
        ),
        (
            "path = Path('/data/logs') / 'app.log'\nif path.exists():\n    content = path.read_text()",
            "Uses `pathlib.Path` to build a cross-platform file path `/data/logs/app.log`. "
            "The `/` operator joins path segments. `.exists()` checks if the file is on disk. "
            "If it exists, `.read_text()` reads its entire contents as a string."
        ),
        (
            "import re\nmatches = re.findall(r'\\b\\d{3}-\\d{4}\\b', text)",
            "Uses a regular expression to find all 7-digit phone number fragments (NXX-XXXX format) in `text`. "
            "`\\b` marks word boundaries. `\\d{3}` matches exactly 3 digits, `-` the literal hyphen, "
            "`\\d{4}` exactly 4 digits. `findall` returns a list of all non-overlapping matches."
        ),
        (
            "df = df[df['salary'] > 50000].copy()\ndf['bonus'] = df['salary'] * 0.1",
            "Filters a pandas DataFrame to rows where salary exceeds 50,000. `.copy()` prevents the "
            "SettingWithCopyWarning by creating an independent copy rather than a view. "
            "Then adds a `'bonus'` column equal to 10% of salary."
        ),
        (
            "with torch.no_grad():\n    logits = model(input_ids)\n    probs = torch.softmax(logits, dim=-1)",
            "`torch.no_grad()` disables gradient computation — saves memory and speeds up inference. "
            "Inside, the model does a forward pass on `input_ids` to produce `logits` (raw unnormalized scores). "
            "`torch.softmax(..., dim=-1)` converts the last-dimension logits to a probability distribution "
            "that sums to 1."
        ),
        (
            "checkpoint = torch.load('model.pt', map_location='cpu')\nmodel.load_state_dict(checkpoint['model'])",
            "Loads a PyTorch checkpoint file from disk into CPU memory (safe even if it was saved on GPU). "
            "The checkpoint dict is expected to have a `'model'` key containing the `state_dict`. "
            "`load_state_dict` maps those saved weights into the current model's parameters."
        ),
        (
            "os.makedirs(output_dir, exist_ok=True)",
            "Creates the directory at `output_dir` including any missing parent directories. "
            "`exist_ok=True` suppresses the error if the directory already exists. "
            "Equivalent to `mkdir -p` in bash."
        ),
        (
            "tokens = tokenizer.encode(text, max_length=512, truncation=True, return_tensors='pt')",
            "Tokenizes `text` using a HuggingFace tokenizer. `max_length=512` caps the output length. "
            "`truncation=True` cuts text that exceeds the limit. `return_tensors='pt'` returns a "
            "PyTorch tensor instead of a plain list. Result is a dict with `'input_ids'` and `'attention_mask'`."
        ),
        (
            "loss = criterion(logits.view(-1, vocab_size), labels.view(-1))",
            "Reshapes `logits` from `[batch, seq_len, vocab_size]` to `[batch*seq_len, vocab_size]` "
            "and `labels` from `[batch, seq_len]` to `[batch*seq_len]`. This flattens the batch and "
            "sequence dimensions into one so `CrossEntropyLoss` can compare each token prediction "
            "to its target label."
        ),
    ]

    for code, explanation in snippets:
        human = f"What does this code do?\n\n```python\n{code}\n```"
        gpt = explanation
        samples.append(pair(human, gpt))

        human2 = f"Explain this Python snippet line by line:\n\n```python\n{code}\n```"
        samples.append(pair(human2, gpt))

    # JavaScript reading
    js_snippets = [
        (
            "const doubled = arr.map(x => x * 2).filter(x => x > 10);",
            "Chains two array methods. `.map(x => x * 2)` creates a new array with each element doubled. "
            "`.filter(x => x > 10)` keeps only elements greater than 10. Result is a new array — original `arr` unchanged."
        ),
        (
            "const { name, age = 25 } = user;",
            "Destructures the `user` object, extracting `name` and `age`. If `user.age` is undefined, "
            "`age` defaults to 25. Clean ES6 syntax for pulling specific fields out of an object."
        ),
        (
            "fetch(url)\n  .then(res => res.json())\n  .then(data => console.log(data))\n  .catch(err => console.error(err));",
            "Makes an async HTTP GET to `url`. `.then(res => res.json())` parses the response body as JSON. "
            "The next `.then` receives the parsed data. `.catch` handles any network error or JSON parse failure."
        ),
        (
            "const memoize = fn => {\n  const cache = {};\n  return (...args) => {\n    const key = JSON.stringify(args);\n    return cache[key] ?? (cache[key] = fn(...args));\n  };\n};",
            "Higher-order function that wraps any function `fn` with memoization. "
            "Each unique argument combination (serialized as JSON) is stored in `cache`. "
            "`??` (nullish coalescing) returns the cached value if it exists, otherwise calls `fn`, stores, and returns the result."
        ),
        (
            "const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));",
            "Creates a Promise-based delay. `setTimeout(resolve, ms)` calls `resolve` after `ms` milliseconds, "
            "settling the Promise. Used with `await sleep(1000)` to pause async code for 1 second."
        ),
    ]

    for code, explanation in js_snippets:
        human = f"What does this JavaScript do?\n\n```javascript\n{code}\n```"
        samples.append(pair(human, explanation))

    # SQL reading
    sql_snippets = [
        (
            "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.id\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;",
            "Finds users with more than 5 orders, sorted by order count descending. "
            "LEFT JOIN includes users even with no orders. GROUP BY aggregates per user. "
            "HAVING filters groups after aggregation (unlike WHERE which filters before). "
            "COUNT(o.id) counts non-null order rows per user."
        ),
        (
            "UPDATE products SET price = price * 1.10 WHERE category = 'electronics' AND stock > 0;",
            "Raises prices by 10% for in-stock electronics. `price * 1.10` multiplies the current price. "
            "Both conditions must be true (AND). Only rows where category is 'electronics' AND stock is positive are updated."
        ),
        (
            "SELECT * FROM logs WHERE created_at > NOW() - INTERVAL '7 days';",
            "Retrieves all log rows from the past 7 days. `NOW()` returns the current timestamp. "
            "Subtracting `INTERVAL '7 days'` gives a timestamp 7 days ago. Any row with `created_at` after that point is returned."
        ),
    ]

    for code, explanation in sql_snippets:
        human = f"What does this SQL query do?\n\n```sql\n{code}\n```"
        samples.append(pair(human, explanation))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUG IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def bug_identification_pairs():
    samples = []

    bugs = [
        (
            "def average(nums):\n    return sum(nums) / len(nums)",
            "ZeroDivisionError when `nums` is an empty list — `len([])` is 0. Fix: guard with `if not nums: return 0` (or return `None` / raise `ValueError`)."
        ),
        (
            "for i in range(len(items)):\n    items.append(i * 2)",
            "Infinite loop. `items.append(...)` grows the list while `range(len(items))` re-evaluates each iteration — the loop never ends. Fix: iterate over a copy or capture `n = len(items)` before the loop."
        ),
        (
            "def get_user(id):\n    return db.query(f'SELECT * FROM users WHERE id = {id}')",
            "SQL injection vulnerability. If `id` comes from user input, an attacker can inject arbitrary SQL. "
            "Fix: use parameterized queries — `db.query('SELECT * FROM users WHERE id = %s', (id,))`."
        ),
        (
            "if x == None:\n    print('empty')",
            "Style/correctness issue: use `is None` not `== None`. The `==` comparison calls `__eq__` which can be overridden by custom classes. The correct check is `if x is None:`."
        ),
        (
            "class Counter:\n    count = 0\n    def increment(self):\n        self.count += 1",
            "Subtle class vs instance variable bug. `count = 0` is a class variable shared by all instances. "
            "The first `self.count += 1` creates an instance variable that shadows it — but if any code reads "
            "`Counter.count` it's always 0. Fix: initialize in `__init__`: `self.count = 0`."
        ),
        (
            "result = []\nfor i in range(5):\n    result.append(lambda: i)",
            "All lambdas capture the same `i` variable by reference. When called, they all return 4 (the final value). "
            "Fix: use a default argument — `lambda i=i: i` — to capture the current value."
        ),
        (
            "data = None\nprint(data['key'])",
            "TypeError: `None` is not subscriptable. `data` was never assigned a dict. "
            "Fix: check `if data is not None:` before accessing, or ensure the upstream function returns a dict."
        ),
        (
            "with open('file.txt') as f:\n    content = f.read()\nprint(f.read())",
            "`f.read()` after the `with` block fails — the file is closed when the `with` exits. "
            "Fix: move the print inside the block, or keep `content` and print that."
        ),
        (
            "import threading\n\ncount = 0\ndef increment():\n    global count\n    for _ in range(100000):\n        count += 1\n\nthreads = [threading.Thread(target=increment) for _ in range(10)]\nfor t in threads: t.start()\nfor t in threads: t.join()\nprint(count)",
            "Race condition — `count += 1` is not atomic (it's read-modify-write). Multiple threads interleave, "
            "causing lost updates. `count` will be less than 1,000,000. Fix: use `threading.Lock()` around the increment."
        ),
        (
            "def process(items=[]):\n    items.append('done')\n    return items",
            "Mutable default argument bug. The default `[]` is created once at function definition and shared across all calls. "
            "Each call without an argument accumulates `'done'` entries. Fix: `def process(items=None): if items is None: items = []`."
        ),
        (
            "model.train()\nwith torch.no_grad():\n    loss = criterion(model(x), y)\nloss.backward()",
            "Can't call `.backward()` inside or after `torch.no_grad()` — gradients aren't tracked. "
            "Fix: only use `no_grad` during inference/evaluation, not during training loss computation."
        ),
        (
            "optimizer.zero_grad()\nfor batch in dataloader:\n    loss = model(batch)\n    loss.backward()\noptimizer.step()",
            "Gradient accumulation bug: `zero_grad()` should be called every iteration, not once before the loop — "
            "gradients from all batches will accumulate into one giant update. Fix: move `optimizer.zero_grad()` inside the loop."
        ),
        (
            "x = torch.tensor([1, 2, 3])\ny = x\ny[0] = 99\nprint(x[0])",
            "Prints 99, not 1. PyTorch tensors (like numpy arrays) use reference semantics by default. "
            "`y = x` makes `y` point to the same storage. Fix: `y = x.clone()` to get an independent copy."
        ),
        (
            "n_lines = sum(1 for _ in open(path))\nfor line in open(path):\n    process(line)",
            "Opens the file twice — once for the count, once for processing. If the file is large this wastes I/O. "
            "Also, neither `open()` call is in a `with` block, so file handles may leak. "
            "Fix: use `with open(path) as f: lines = f.readlines()` once."
        ),
        (
            "results = {}\nfor key, value in pairs:\n    results[key].append(value)",
            "KeyError on the first occurrence of any key — `results[key]` raises because the key doesn't exist yet. "
            "Fix: use `collections.defaultdict(list)` or `results.setdefault(key, []).append(value)`."
        ),
        (
            "async def main():\n    result = fetch_data()\n    print(result)",
            "Missing `await`. `fetch_data()` is presumably an async coroutine — calling it without `await` returns a coroutine object, not the actual data. Fix: `result = await fetch_data()`."
        ),
        (
            "df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))\ndf.sort_values('date')",
            "`sort_values` returns a new DataFrame but the result isn't assigned back — original `df` is unchanged. "
            "Fix: `df = df.sort_values('date')` or `df.sort_values('date', inplace=True)`."
        ),
        (
            "import json\nwith open('config.json') as f:\n    cfg = json.load(f)\nprint(cfg.api_key)",
            "AttributeError — `json.load` returns a Python dict, not an object. Dicts use bracket access, not dot access. "
            "Fix: `print(cfg['api_key'])` or use `argparse.Namespace(**cfg)` if dot access is desired."
        ),
        (
            "password = 'secret123'\nif user_input == password:\n    grant_access()",
            "Security issue — plain text string comparison is vulnerable to timing attacks. "
            "Fix: use `hmac.compare_digest(user_input, password)` which runs in constant time regardless of match length."
        ),
        (
            "print('Starting...')\nsys.exit(0)\nprint('Done')",
            "`print('Done')` is unreachable — `sys.exit(0)` terminates the process immediately. "
            "Not a crash, but dead code. Fix: remove the `sys.exit(0)` call or move it after the final print."
        ),
    ]

    for code, explanation in bugs:
        human = f"What is wrong with this code?\n\n```python\n{code}\n```"
        samples.append(pair(human, explanation))

        human2 = f"Find the bug in this code:\n\n```python\n{code}\n```"
        samples.append(pair(human2, explanation))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUG FIXING
# ─────────────────────────────────────────────────────────────────────────────

def bug_fixing_pairs():
    samples = []

    fixes = [
        (
            "def divide(a, b):\n    return a / b",
            "ZeroDivisionError when b is 0",
            "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
            "Added a guard that raises a descriptive `ValueError` before the division. This is better than returning "
            "`None` or `inf` because callers see a clear error message."
        ),
        (
            "def process(items=[]):\n    items.append('done')\n    return items",
            "Mutable default argument — list persists between calls",
            "def process(items=None):\n    if items is None:\n        items = []\n    items.append('done')\n    return items",
            "Replaced `[]` default with `None`. A fresh list is created inside the function body each time. "
            "This is the standard Python fix for mutable defaults."
        ),
        (
            "results = {}\nfor key, value in pairs:\n    results[key].append(value)",
            "KeyError on first occurrence of each key",
            "from collections import defaultdict\n\nresults = defaultdict(list)\nfor key, value in pairs:\n    results[key].append(value)",
            "Used `defaultdict(list)` — it automatically creates an empty list for any new key, so `.append()` "
            "never raises a KeyError."
        ),
        (
            "for i in range(len(items)):\n    if items[i] < 0:\n        items.remove(items[i])",
            "Skips elements when removing from a list being iterated by index",
            "items = [x for x in items if x >= 0]",
            "List comprehension is the cleanest fix — builds a new list keeping only non-negative values. "
            "No mutation-during-iteration problem."
        ),
        (
            "import threading\ncount = 0\ndef increment():\n    global count\n    count += 1",
            "Race condition — count += 1 is not atomic",
            "import threading\ncount = 0\nlock = threading.Lock()\ndef increment():\n    global count\n    with lock:\n        count += 1",
            "Added a `threading.Lock()`. The `with lock:` block ensures only one thread executes the read-modify-write "
            "at a time, eliminating the race condition."
        ),
        (
            "optimizer.zero_grad()\nfor batch in loader:\n    loss = model(batch)\n    loss.backward()\noptimizer.step()",
            "zero_grad called once before the loop; gradients accumulate",
            "for batch in loader:\n    optimizer.zero_grad()\n    loss = model(batch)\n    loss.backward()\n    optimizer.step()",
            "Moved `zero_grad()` and `step()` inside the loop. Each batch now gets a fresh gradient accumulator "
            "and an independent weight update."
        ),
        (
            "x = torch.tensor([1, 2, 3])\ny = x\ny[0] = 99",
            "y and x share storage — mutating y mutates x",
            "x = torch.tensor([1, 2, 3])\ny = x.clone()\ny[0] = 99",
            "`x.clone()` creates an independent tensor with its own storage. Mutations to `y` no longer affect `x`."
        ),
        (
            "df.sort_values('date')\nprint(df.head())",
            "sort_values returns a new DataFrame; original df is unchanged",
            "df = df.sort_values('date')\nprint(df.head())",
            "Assigned the result back to `df`. `sort_values` does not sort in-place by default."
        ),
        (
            "result = []\nfor i in range(5):\n    result.append(lambda: i)",
            "All lambdas capture the same i; return 4 when called",
            "result = []\nfor i in range(5):\n    result.append(lambda i=i: i)",
            "Default argument `i=i` captures the current value of `i` at definition time. "
            "Each lambda now holds its own copy."
        ),
        (
            "async def get_data():\n    response = fetch(url)\n    return response.json()",
            "Missing await on async coroutine fetch",
            "async def get_data():\n    response = await fetch(url)\n    return response.json()",
            "Added `await` before `fetch(url)`. Without it, `response` is a coroutine object, not the HTTP response."
        ),
        (
            "class Dog:\n    tricks = []\n    def add_trick(self, trick):\n        self.tricks.append(trick)",
            "tricks is a class variable; all Dog instances share the same list",
            "class Dog:\n    def __init__(self):\n        self.tricks = []\n    def add_trick(self, trick):\n        self.tricks.append(trick)",
            "Moved `tricks` into `__init__` so each instance gets its own independent list."
        ),
        (
            "n = int(input('Enter a number: '))\nprint(10 / n)",
            "Crashes on non-integer input and on n=0",
            "try:\n    n = int(input('Enter a number: '))\n    if n == 0:\n        print('Cannot divide by zero')\n    else:\n        print(10 / n)\nexcept ValueError:\n    print('Please enter a valid integer')",
            "Wrapped in try/except to catch `ValueError` from bad input. Added explicit zero check before division."
        ),
        (
            "with open('output.txt', 'w') as f:\n    for line in data:\n        f.write(line)",
            "Missing newlines — all data written on one line",
            "with open('output.txt', 'w') as f:\n    for line in data:\n        f.write(line + '\\n')",
            "Appended `'\\n'` to each line. Alternatively use `f.writelines(line + '\\n' for line in data)`."
        ),
        (
            "model.eval()\nloss = criterion(model(x), y)\noptimizer.zero_grad()\nloss.backward()\noptimizer.step()",
            "model.eval() disables dropout/batchnorm AND implies no training — but then backward/step are called",
            "model.train()\noptimizer.zero_grad()\nlogits = model(x)\nloss = criterion(logits, y)\nloss.backward()\noptimizer.step()",
            "Replaced `model.eval()` with `model.train()` for training loops. Moved `zero_grad()` before the forward pass."
        ),
        (
            "path = '/data/' + folder + '/' + filename",
            "Manual string concatenation for paths is fragile and OS-dependent",
            "from pathlib import Path\npath = Path('/data') / folder / filename",
            "Used `pathlib.Path` with `/` operator for cross-platform path joining. Handles edge cases like trailing slashes."
        ),
    ]

    for broken, error_desc, fixed, explanation in fixes:
        human = (
            f"Fix this code. The problem is: {error_desc}\n\n"
            f"```python\n{broken}\n```"
        )
        gpt = f"**Fixed code:**\n\n```python\n{fixed}\n```\n\n**Why:** {explanation}"
        samples.append(pair(human, gpt))

        human2 = (
            f"This code has a bug: {error_desc}. Show me the corrected version.\n\n"
            f"```python\n{broken}\n```"
        )
        samples.append(pair(human2, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. CODE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def code_generation_pairs():
    samples = []

    specs = [
        (
            "Write a Python function that takes a list of integers and returns the two numbers that sum to a target value.",
            "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        complement = target - n\n        if complement in seen:\n            return [seen[complement], i]\n        seen[n] = i\n    return []\n\n# Example\n# two_sum([2, 7, 11, 15], 9)  # → [0, 1]",
            "Uses a hash map to track seen numbers. For each number, checks if its complement (target - n) was seen. O(n) time, O(n) space."
        ),
        (
            "Write a Python class for a stack with push, pop, peek, and is_empty methods.",
            "class Stack:\n    def __init__(self):\n        self._data = []\n\n    def push(self, item):\n        self._data.append(item)\n\n    def pop(self):\n        if self.is_empty():\n            raise IndexError('pop from empty stack')\n        return self._data.pop()\n\n    def peek(self):\n        if self.is_empty():\n            raise IndexError('peek from empty stack')\n        return self._data[-1]\n\n    def is_empty(self):\n        return len(self._data) == 0\n\n    def __len__(self):\n        return len(self._data)",
            "Backed by a Python list. All operations are O(1). Raises `IndexError` on empty access to match Python built-in behavior."
        ),
        (
            "Write a function that reads a JSONL file and returns a list of parsed dicts.",
            "import json\nfrom pathlib import Path\n\ndef load_jsonl(path):\n    path = Path(path)\n    records = []\n    with open(path, encoding='utf-8') as f:\n        for lineno, line in enumerate(f, 1):\n            line = line.strip()\n            if not line:\n                continue\n            try:\n                records.append(json.loads(line))\n            except json.JSONDecodeError as e:\n                print(f'Warning: bad JSON on line {lineno}: {e}')\n    return records",
            "Skips blank lines, catches malformed JSON with a warning, and reads line by line (memory-efficient for large files)."
        ),
        (
            "Write a Python decorator that retries a function up to 3 times on exception.",
            "import time\nimport functools\n\ndef retry(max_attempts=3, delay=1.0):\n    def decorator(fn):\n        @functools.wraps(fn)\n        def wrapper(*args, **kwargs):\n            for attempt in range(1, max_attempts + 1):\n                try:\n\n                    return fn(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts:\n                        raise\n                    print(f'Attempt {attempt} failed: {e}. Retrying in {delay}s...')\n                    time.sleep(delay)\n        return wrapper\n    return decorator\n\n# Usage\n@retry(max_attempts=3, delay=2)\ndef fetch_data(url):\n    ...",
            "Parameterized decorator with configurable attempts and delay. Re-raises on the final attempt so the caller still sees the exception."
        ),
        (
            "Write a Python function to flatten a nested list of arbitrary depth.",
            "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n\n# Example\n# flatten([1, [2, [3, 4]], 5])  # → [1, 2, 3, 4, 5]",
            "Recursively descends into nested lists. `isinstance(item, list)` detects nesting. `extend` adds the flattened sub-result."
        ),
        (
            "Write a function that counts word frequency in a string and returns the top N words.",
            "from collections import Counter\n\ndef top_words(text, n=10):\n    words = text.lower().split()\n    # strip punctuation\n    words = [w.strip('.,!?;:\\\"') for w in words]\n    words = [w for w in words if w]\n    counter = Counter(words)\n    return counter.most_common(n)",
            "Lowercases, strips common punctuation, filters empties, then uses `Counter.most_common(n)` which returns the top N `(word, count)` tuples."
        ),
        (
            "Write a binary search function in Python.",
            "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n\n# arr must be sorted\n# Returns index of target or -1 if not found",
            "Classic iterative binary search. O(log n) time. Integer division `//` avoids float mid. Returns -1 (not None) to match C-style convention."
        ),
        (
            "Write a Python context manager that times a code block.",
            "import time\nfrom contextlib import contextmanager\n\n@contextmanager\ndef timer(label=''):\n    start = time.perf_counter()\n    try:\n        yield\n    finally:\n        elapsed = time.perf_counter() - start\n        print(f'{label}: {elapsed:.4f}s')\n\n# Usage\nwith timer('loading data'):\n    data = load_big_file()",
            "Uses `@contextmanager` for brevity. `perf_counter()` is the most precise clock for short intervals. "
            "`finally` guarantees the elapsed time is printed even if the block raises."
        ),
        (
            "Write a function that validates an email address using a regex.",
            "import re\n\nEMAIL_RE = re.compile(\n    r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'\n)\n\ndef is_valid_email(email):\n    return bool(EMAIL_RE.match(email.strip()))",
            "Pre-compiled regex for performance. Anchored with `^` and `$`. Covers common email characters. "
            "For production, consider the `email-validator` library for full RFC 5322 compliance."
        ),
        (
            "Write a Python async function that fetches multiple URLs concurrently.",
            "import asyncio\nimport aiohttp\n\nasync def fetch_one(session, url):\n    async with session.get(url) as resp:\n        return await resp.text()\n\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_one(session, url) for url in urls]\n        return await asyncio.gather(*tasks)\n\n# Usage\n# results = asyncio.run(fetch_all(['https://...', 'https://...']))",
            "Uses `aiohttp` for async HTTP and `asyncio.gather` to run all requests concurrently. "
            "A single `ClientSession` is reused across requests for connection pooling."
        ),
        (
            "Write a function to merge two sorted lists into one sorted list.",
            "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result",
            "Two-pointer merge — the core of merge sort. Compares the front of each list, takes the smaller. "
            "After one list is exhausted, appends the rest of the other. O(n+m) time."
        ),
        (
            "Write a Python function that deep-copies a nested dictionary.",
            "import copy\n\ndef deep_copy_dict(d):\n    return copy.deepcopy(d)\n\n# Or manual implementation:\ndef deep_copy_manual(obj):\n    if isinstance(obj, dict):\n        return {k: deep_copy_manual(v) for k, v in obj.items()}\n    if isinstance(obj, list):\n        return [deep_copy_manual(x) for x in obj]\n    return obj  # primitives are immutable — safe to share",
            "`copy.deepcopy` is the idiomatic answer and handles all edge cases. "
            "The manual version shows how it works: recursively copy dicts and lists, primitives are shared (safe since immutable)."
        ),
        (
            "Write a function that batches a list into chunks of size n.",
            "def batch(lst, n):\n    for i in range(0, len(lst), n):\n        yield lst[i:i + n]\n\n# Usage\n# list(batch([1,2,3,4,5], 2))  # → [[1,2],[3,4],[5]]",
            "Generator function — yields slices without building all batches in memory at once. "
            "The last chunk may be smaller than `n` if `len(lst)` isn't divisible."
        ),
        (
            "Write a simple rate limiter class in Python.",
            "import time\n\nclass RateLimiter:\n    def __init__(self, max_calls, period):\n        self.max_calls = max_calls\n        self.period = period\n        self.calls = []\n\n    def __call__(self, fn):\n        import functools\n        @functools.wraps(fn)\n        def wrapper(*args, **kwargs):\n            now = time.monotonic()\n            self.calls = [t for t in self.calls if now - t < self.period]\n            if len(self.calls) >= self.max_calls:\n                sleep_time = self.period - (now - self.calls[0])\n                time.sleep(max(0, sleep_time))\n            self.calls.append(time.monotonic())\n            return fn(*args, **kwargs)\n        return wrapper\n\n# Usage\n@RateLimiter(max_calls=5, period=1.0)\ndef api_call(): ...",
            "Sliding window rate limiter. Keeps timestamps of recent calls, drops old ones, sleeps if at the limit."
        ),
        (
            "Write a function that parses a CSV file into a list of dicts.",
            "import csv\nfrom pathlib import Path\n\ndef load_csv(path):\n    records = []\n    with open(Path(path), newline='', encoding='utf-8') as f:\n        reader = csv.DictReader(f)\n        for row in reader:\n            records.append(dict(row))\n    return records",
            "`csv.DictReader` uses the first row as column names. `newline=''` prevents universal newline translation "
            "that can corrupt Windows-style CSV files. Returns a list of plain dicts."
        ),
    ]

    for spec, code, explanation in specs:
        human = spec
        gpt = f"```python\n{code}\n```\n\n{explanation}"
        samples.append(pair(human, gpt))

        human2 = f"Implement this in Python: {spec}"
        samples.append(pair(human2, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 5. INTEGRATION & WIRING
# ─────────────────────────────────────────────────────────────────────────────

def integration_pairs():
    samples = []

    wiring = [
        (
            "How do I connect a Flask route to a SQLAlchemy database model?",
            """Here's the full wiring pattern:

```python
# models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
```

```python
# app.py
from flask import Flask, jsonify
from models import db, User

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db.init_app(app)

@app.route('/users')
def get_users():
    users = User.query.all()
    return jsonify([{'id': u.id, 'name': u.name} for u in users])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
```

**Key wiring steps:**
1. `db = SQLAlchemy()` in models.py (no app yet)
2. `db.init_app(app)` in app.py to bind them
3. `app.app_context()` needed for db operations outside request context
4. All model queries happen inside route handlers (which run in request context automatically)"""
        ),
        (
            "How do I connect a PyTorch model to a HuggingFace tokenizer for inference?",
            """Complete inference wiring:

```python
from transformers import AutoTokenizer
import torch
from anthos.main import Anthos
from anthos.configs import get_training_config

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('data/anthos_tokenizer')

# 2. Load model
cfg, _ = get_training_config('sft')
model = Anthos(cfg)
ckpt = torch.load('checkpoints/mansa_sovereign/step_010000.pt', map_location='cpu')
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# 3. Tokenize input
def generate(prompt, max_new_tokens=200):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(generate('What is quantum computing?'))
```

**Wiring checklist:**
- Tokenizer vocab size must match `cfg.vocab_size`
- `map_location='cpu'` prevents crash if checkpoint was saved on GPU
- `model.eval()` disables dropout/training-mode batchnorm
- `torch.no_grad()` skips gradient tracking — faster and less memory"""
        ),
        (
            "How do I wire a Redis cache in front of a Python function?",
            """Cache-aside pattern:

```python
import redis
import json
import functools

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache(ttl=300):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = f'{fn.__name__}:{json.dumps(args)}:{json.dumps(kwargs, sort_keys=True)}'
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)
            result = fn(*args, **kwargs)
            redis_client.setex(key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage
@cache(ttl=600)
def fetch_user_profile(user_id):
    return db.query('SELECT * FROM users WHERE id = %s', (user_id,))
```

**How it works:**
1. Build a deterministic cache key from function name + arguments
2. `redis_client.get(key)` — cache hit: deserialize and return
3. Cache miss: call the real function, store result with TTL, return
4. `setex` atomically sets value + expiry in one command"""
        ),
        (
            "How do I wire server-sent events (SSE) for token streaming from a Flask backend to JavaScript?",
            """**Backend (Flask):**

```python
from flask import Flask, Response, stream_with_context

app = Flask(__name__)

@app.route('/chat/stream')
def stream():
    def generate():
        for token in model.generate_stream(prompt):
            yield f'data: {json.dumps({"token": token})}\\n\\n'
        yield 'data: [DONE]\\n\\n'
    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})
```

**Frontend (JavaScript):**

```javascript
const evtSource = new EventSource('/chat/stream');

evtSource.onmessage = (e) => {
    if (e.data === '[DONE]') {
        evtSource.close();
        return;
    }
    const { token } = JSON.parse(e.data);
    document.getElementById('output').textContent += token;
};

evtSource.onerror = () => evtSource.close();
```

**Key wiring notes:**
- `text/event-stream` MIME type is required for SSE
- Each event is `data: <payload>\\n\\n` (double newline terminates the event)
- `X-Accel-Buffering: no` prevents nginx from buffering the stream
- `[DONE]` sentinel signals the client to close the connection"""
        ),
        (
            "How do I connect a Python training script to Weights & Biases for experiment tracking?",
            """```python
import wandb
import torch

# 1. Initialize at the start of training
wandb.init(
    project='anthos-training',
    name='sft-run-001',
    config={
        'lr': 3e-5,
        'batch_size': 8,
        'max_steps': 10000,
        'model': 'anthos-47M',
    }
)

# 2. Log metrics inside the training loop
for step, (ids, labels) in enumerate(loader):
    optimizer.zero_grad()
    logits, _, loop_counts = model(ids)
    loss = loss_fn(logits, labels, loop_counts)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        wandb.log({
            'train/loss': loss.item(),
            'train/step': step,
            'train/lr': scheduler.get_last_lr()[0],
        }, step=step)

    # 3. Save checkpoint artifact
    if step % 1000 == 0:
        torch.save({'model': model.state_dict(), 'step': step}, f'checkpoints/step_{step:06d}.pt')
        wandb.save(f'checkpoints/step_{step:06d}.pt')

wandb.finish()
```

**Wiring steps:**
1. `wandb.init(...)` — creates the run (set `WANDB_API_KEY` env var)
2. `wandb.log(dict, step=step)` — any metrics, any frequency
3. `wandb.save(path)` — uploads checkpoint files to the run
4. `wandb.finish()` — marks the run complete"""
        ),
        (
            "How do I wire a LangChain retriever to inject context into a chat model?",
            """```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Build the vector store (do this once)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
vectorstore.save_local('faiss_index')

# 2. Load and wire at query time
vectorstore = FAISS.load_local('faiss_index', embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True
)

# 3. Query
result = chain({'query': 'What is the patient treatment plan?'})
print(result['result'])
print(result['source_documents'])
```

**Flow:** query → retriever fetches top-4 docs → 'stuff' chain concatenates them into the prompt → LLM generates answer with context injected."""
        ),
        (
            "Show me how to wire a dataloader to a training loop in PyTorch.",
            """Complete training loop wiring:

```python
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class TextDataset(Dataset):
    def __init__(self, ids_list, labels_list):
        self.ids = ids_list
        self.labels = labels_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]

# Wire: Dataset → DataLoader → Training loop
dataset = TextDataset(all_ids, all_labels)
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,         # 0 on Mac/CPU to avoid fork issues
    pin_memory=False,      # True only if using CUDA
)

model = MyModel()
optimizer = AdamW(model.parameters(), lr=3e-5)

model.train()
for epoch in range(num_epochs):
    for step, (ids, labels) in enumerate(loader):
        optimizer.zero_grad()
        logits = model(ids)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f'Epoch {epoch} Step {step} Loss {loss.item():.4f}')
```

**Key wiring points:**
- `DataLoader` wraps any `Dataset` with batching, shuffling, and optional multiprocessing
- `shuffle=True` randomizes order each epoch — important for training quality
- `clip_grad_norm_` prevents exploding gradients (max norm=1.0 is standard)"""
        ),
        (
            "How do I connect a Python module across files without circular imports?",
            """The standard approach: initialize the shared resource in a separate `core.py` or use dependency injection.

**Pattern 1 — Separate init module:**
```python
# core.py
from transformers import AutoTokenizer
tokenizer = None

def init(path='data/anthos_tokenizer'):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
```

```python
# app.py
import core
core.init()  # called once at startup

from routes import users_bp
app.register_blueprint(users_bp)
```

```python
# routes/users.py
import core

@bp.route('/tokenize')
def tokenize():
    ids = core.tokenizer.encode(request.json['text'])
    return jsonify({'ids': ids})
```

**Pattern 2 — Lazy import (break circular chains):**
```python
# If module A needs module B and B needs A:
# In A, import B inside the function that needs it:
def process():
    from module_b import helper  # delayed import
    return helper()
```

**Rule of thumb:** if two modules need each other, extract the shared part into a third module that both import."""
        ),
        (
            "How do I wire environment variables into a Python app safely?",
            """```python
# config.py — single source of truth for all env config
import os
from dataclasses import dataclass

@dataclass
class Config:
    db_url: str
    secret_key: str
    debug: bool
    port: int
    hf_token: str

def load_config() -> Config:
    missing = []
    def require(key):
        val = os.environ.get(key)
        if val is None:
            missing.append(key)
        return val

    cfg = Config(
        db_url      = require('DATABASE_URL'),
        secret_key  = require('SECRET_KEY'),
        debug       = os.environ.get('DEBUG', 'false').lower() == 'true',
        port        = int(os.environ.get('PORT', '8000')),
        hf_token    = os.environ.get('HF_TOKEN', ''),  # optional
    )
    if missing:
        raise EnvironmentError(f'Missing required env vars: {missing}')
    return cfg

# Usage: once at startup
cfg = load_config()
```

**Wiring steps:**
1. Never scatter `os.environ.get(...)` throughout the codebase
2. Centralize in `config.py`, load once at startup
3. Fail fast on missing required vars — better than mysterious None errors later
4. Use `.env` file + `python-dotenv` in development:
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # loads .env into os.environ before load_config()
   ```"""
        ),
        (
            "How do I wire a checkpoint save/load into a training loop so I can resume from where I left off?",
            """```python
import torch
from pathlib import Path

CKPT_DIR = Path('checkpoints/mansa_sovereign')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(model, optimizer, step, loss):
    path = CKPT_DIR / f'step_{step:06d}.pt'
    torch.save({
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step':      step,
        'loss':      loss,
    }, path)
    print(f'  Saved checkpoint: {path}')
    return path

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('step', 0)

# Training loop with resume support
start_step = 0
if args.resume and Path(args.resume).exists():
    start_step = load_checkpoint(args.resume, model, optimizer)
    print(f'Resumed from step {start_step}')

for step in range(start_step, max_steps):
    # ... training ...
    if step % save_every == 0:
        save_checkpoint(model, optimizer, step, loss.item())
```

**What each field does:**
- `model` state dict: all weights (required to resume)
- `optimizer` state dict: momentum/Adam state (skip this → optimizer cold-starts, loss may spike)
- `step`: where to resume iteration count
- `loss`: optional, useful for logging/debugging"""
        ),
    ]

    for human, gpt in wiring:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 6. SMOKE TESTING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def smoke_testing_pairs():
    samples = []

    tests = [
        (
            "Write a smoke test for a Flask API endpoint that returns user data.",
            """```python
# test_smoke.py
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as c:
        yield c

def test_get_users_returns_200(client):
    response = client.get('/users')
    assert response.status_code == 200

def test_get_users_returns_json(client):
    response = client.get('/users')
    data = response.get_json()
    assert isinstance(data, list)

def test_get_user_by_id(client):
    response = client.get('/users/1')
    assert response.status_code in (200, 404)  # 404 ok if no seed data

def test_create_user(client):
    response = client.post('/users', json={'name': 'Test', 'email': 'test@test.com'})
    assert response.status_code in (200, 201)
    data = response.get_json()
    assert 'id' in data
```

**Run:**
```bash
pip install pytest
pytest test_smoke.py -v
```

**What this catches:** import errors, routing bugs, JSON serialization crashes, missing fields in response."""
        ),
        (
            "Write a smoke test for a PyTorch model to verify it loads and runs a forward pass.",
            """```python
# test_model_smoke.py
import torch
import pytest

def test_model_import():
    from anthos.main import Anthos
    from anthos.configs import get_training_config
    cfg, _ = get_training_config('sft')
    assert cfg is not None

def test_model_init():
    from anthos.main import Anthos
    from anthos.configs import get_training_config
    cfg, _ = get_training_config('sft')
    model = Anthos(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 1_000_000, f'Model too small: {n_params}'

def test_forward_pass():
    from anthos.main import Anthos
    from anthos.configs import get_training_config
    cfg, _ = get_training_config('sft')
    model = Anthos(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        out = model(x)
    assert out is not None
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 32, cfg.vocab_size)

def test_loss_is_finite():
    from anthos.main import Anthos
    from anthos.configs import get_training_config
    import torch.nn.functional as F
    cfg, _ = get_training_config('sft')
    model = Anthos(cfg)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    labels = x.clone()
    logits = model(x)[0]
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))
    assert torch.isfinite(loss), f'Loss is not finite: {loss}'
    assert loss.item() > 0
```

**Run:** `pytest test_model_smoke.py -v`

These four tests catch: bad imports, wrong parameter count, shape mismatches, NaN/inf losses."""
        ),
        (
            "How do I write a smoke test for a dataset loader?",
            """```python
# test_data_smoke.py
from pathlib import Path
import json
import pytest

DATA_PATH = Path('data/sft_master.jsonl')

def test_file_exists():
    assert DATA_PATH.exists(), f'{DATA_PATH} not found'

def test_file_not_empty():
    assert DATA_PATH.stat().st_size > 1_000_000, 'File suspiciously small'

def test_first_100_lines_valid():
    bad = 0
    with open(DATA_PATH) as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            try:
                obj = json.loads(line)
                convs = obj.get('conversations', [])
                has_human = any(c['from'] in ('human','user') for c in convs)
                has_gpt = any(c['from'] in ('gpt','assistant') for c in convs)
                if not (has_human and has_gpt):
                    bad += 1
            except Exception:
                bad += 1
    assert bad == 0, f'{bad}/100 lines failed validation'

def test_dataloader_produces_batches():
    from anthos.data import get_chat_dataloader
    loader = get_chat_dataloader(
        seq_len=128, batch_size=2, num_workers=0,
        tokenizer_path='data/anthos_tokenizer',
        max_samples=50, dataset_name=str(DATA_PATH),
    )
    ids, labels = next(iter(loader))
    assert ids.shape == (2, 128)
    assert labels.shape == (2, 128)
```

**Run:** `pytest test_data_smoke.py -v`"""
        ),
        (
            "How do I write a smoke test to verify a REST API is up and returning correct data?",
            """```python
# test_api_smoke.py
import requests
import pytest

BASE = 'http://localhost:5000'

@pytest.mark.smoke
def test_health_endpoint():
    r = requests.get(f'{BASE}/health', timeout=5)
    assert r.status_code == 200

@pytest.mark.smoke
def test_status_endpoint():
    r = requests.get(f'{BASE}/status', timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data.get('loaded') is True

@pytest.mark.smoke
def test_chat_endpoint_responds():
    r = requests.post(f'{BASE}/chat', json={'message': 'Hello'}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert 'response' in data
    assert len(data['response']) > 0

@pytest.mark.smoke
def test_chat_does_not_return_empty():
    r = requests.post(f'{BASE}/chat', json={'message': 'What is 2+2?'}, timeout=30)
    response_text = r.json().get('response', '')
    assert len(response_text.strip()) > 5

@pytest.mark.smoke
def test_memory_endpoint():
    r = requests.get(f'{BASE}/memory', timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert 'turns' in data
```

**Run:** Start the server first, then `pytest test_api_smoke.py -m smoke -v`"""
        ),
        (
            "Walk me through how to smoke test a new module before wiring it into the main app.",
            """**Step-by-step smoke test process for a new module:**

**1. Test the import alone:**
```python
python -c "from anthos.retrieval import retrieve, needs_retrieval; print('Import OK')"
```
If this fails — missing deps, syntax error, circular import — fix it before anything else.

**2. Test the core function with known inputs:**
```python
python -c "
from anthos.retrieval import needs_retrieval
assert needs_retrieval('current price of Bitcoin') == True
assert needs_retrieval('write me a poem') == False
print('Logic checks PASS')
"
```

**3. Test integration with its dependencies:**
```python
python -c "
from anthos.retrieval import retrieve
ctx = retrieve('What is the speed of light', use_web=False, use_wiki=True)
assert isinstance(ctx, str)
assert len(ctx) > 0
print('Retrieved:', ctx[:100])
"
```

**4. Write a pytest file for repeatable testing:**
```python
# test_retrieval_smoke.py
from anthos.retrieval import needs_retrieval, retrieve

def test_heuristic_true():
    assert needs_retrieval('stock price today')

def test_heuristic_false():
    assert not needs_retrieval('explain recursion')

def test_retrieve_returns_string():
    ctx = retrieve('capital of France', use_web=False, use_wiki=True)
    assert isinstance(ctx, str)

def test_retrieve_empty_fallback():
    ctx = retrieve('', use_web=False, use_wiki=False)
    assert isinstance(ctx, str)  # should return '' not raise
```

**5. Only after all pass: wire into the main app.**

This process prevents broken modules from crashing the main app and makes debugging faster — you know exactly which module failed."""
        ),
        (
            "How do I verify that a checkpoint saved correctly and can be loaded?",
            """```python
# verify_checkpoint.py
import torch
from pathlib import Path
import sys

def verify(ckpt_path):
    path = Path(ckpt_path)
    if not path.exists():
        print(f'FAIL: {path} does not exist')
        return False

    size_mb = path.stat().st_size / 1e6
    print(f'File: {path} ({size_mb:.1f} MB)')

    # Load
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f'FAIL: Cannot load: {e}')
        return False

    # Check keys
    expected_keys = {'model', 'step'}
    missing = expected_keys - set(ckpt.keys())
    if missing:
        print(f'WARN: Missing keys: {missing}')
    else:
        print(f'Keys: {list(ckpt.keys())}')
        print(f'Step: {ckpt["step"]}')

    # Check model weights
    model_sd = ckpt['model']
    print(f'Weight tensors: {len(model_sd)}')

    # Verify no NaN/inf in weights
    bad_tensors = [k for k, v in model_sd.items() if not torch.isfinite(v).all()]
    if bad_tensors:
        print(f'FAIL: NaN/inf in weights: {bad_tensors[:5]}')
        return False

    print('PASS: Checkpoint is valid')
    return True

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/mansa_sovereign/step_010000.pt'
    ok = verify(path)
    sys.exit(0 if ok else 1)
```

**Run:** `python verify_checkpoint.py checkpoints/mansa_sovereign/step_010000.pt`

**What it checks:** file exists, loadable, expected keys present, step number, no NaN/inf in weights."""
        ),
        (
            "How do I write a smoke test that proves my tokenizer and model are compatible?",
            """```python
# test_tokenizer_model_compat.py
import torch
import pytest

def test_vocab_sizes_match():
    from transformers import AutoTokenizer
    from anthos.configs import get_training_config
    tok = AutoTokenizer.from_pretrained('data/anthos_tokenizer')
    cfg, _ = get_training_config('sft')
    assert tok.vocab_size == cfg.vocab_size, (
        f'Tokenizer vocab {tok.vocab_size} != model vocab {cfg.vocab_size}'
    )

def test_tokenizer_encodes_and_model_accepts():
    from transformers import AutoTokenizer
    from anthos.main import Anthos
    from anthos.configs import get_training_config
    tok = AutoTokenizer.from_pretrained('data/anthos_tokenizer')
    cfg, _ = get_training_config('sft')
    model = Anthos(cfg)
    model.eval()

    text = 'Hello, what can you do?'
    ids = tok.encode(text, return_tensors='pt')
    assert ids.max() < cfg.vocab_size, 'Token id exceeds vocab size'

    with torch.no_grad():
        out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape[-1] == cfg.vocab_size

def test_decode_is_readable():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('data/anthos_tokenizer')
    original = 'The quick brown fox'
    ids = tok.encode(original, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert original.lower() in decoded.lower()

def test_max_seq_len_ok():
    from transformers import AutoTokenizer
    from anthos.configs import get_training_config
    tok = AutoTokenizer.from_pretrained('data/anthos_tokenizer')
    cfg, _ = get_training_config('sft')
    # Long text should not crash with truncation
    long_text = 'word ' * 1000
    ids = tok.encode(long_text, max_length=cfg.max_seq_len, truncation=True, return_tensors='pt')
    assert ids.shape[1] <= cfg.max_seq_len
```"""
        ),
    ]

    for human, gpt in tests:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 7. REFACTORING
# ─────────────────────────────────────────────────────────────────────────────

def refactoring_pairs():
    samples = []

    refactors = [
        (
            "Refactor this repetitive code:\n\n```python\nif animal == 'dog':\n    sound = 'woof'\nelif animal == 'cat':\n    sound = 'meow'\nelif animal == 'cow':\n    sound = 'moo'\nelif animal == 'duck':\n    sound = 'quack'\n```",
            """Use a lookup dict:

```python
SOUNDS = {
    'dog': 'woof',
    'cat': 'meow',
    'cow': 'moo',
    'duck': 'quack',
}

sound = SOUNDS.get(animal, 'unknown')
```

**Why:** dict lookup is O(1) and trivially extensible — add a new animal by adding one line to `SOUNDS`, not a new `elif`. `dict.get()` handles unknown animals gracefully."""
        ),
        (
            "Refactor this nested code to reduce indentation:\n\n```python\ndef process_order(order):\n    if order is not None:\n        if order['status'] == 'active':\n            if order['total'] > 0:\n                items = order['items']\n                if len(items) > 0:\n                    return sum(i['price'] for i in items)\n```",
            """Use early returns (guard clauses):

```python
def process_order(order):
    if order is None:
        return None
    if order['status'] != 'active':
        return None
    if order['total'] <= 0:
        return None
    items = order.get('items', [])
    if not items:
        return None
    return sum(i['price'] for i in items)
```

**Why:** each failure condition returns immediately. The "happy path" is at the left margin, unindented. Easier to read, easier to add conditions."""
        ),
        (
            "Refactor this imperative loop to be more Pythonic:\n\n```python\nresult = []\nfor item in items:\n    if item['active']:\n        result.append(item['name'].upper())\n```",
            """List comprehension:

```python
result = [item['name'].upper() for item in items if item['active']]
```

One line, no mutable accumulator, clearly communicates intent: filter then transform."""
        ),
        (
            "Refactor this function that is doing too many things:\n\n```python\ndef handle_request(data):\n    # validate\n    if 'name' not in data:\n        return {'error': 'missing name'}\n    # save to db\n    db.execute('INSERT INTO users ...')\n    # send email\n    send_email(data['email'], 'Welcome!')\n    # return response\n    return {'ok': True}\n```",
            """Split into focused functions:

```python
def validate_user_data(data):
    if 'name' not in data:
        raise ValueError('missing name')
    if 'email' not in data:
        raise ValueError('missing email')

def save_user(data):
    db.execute('INSERT INTO users (name, email) VALUES (%s, %s)',
               (data['name'], data['email']))

def welcome_user(email):
    send_email(email, 'Welcome!')

def handle_request(data):
    try:
        validate_user_data(data)
        save_user(data)
        welcome_user(data['email'])
        return {'ok': True}
    except ValueError as e:
        return {'error': str(e)}
```

**Why:** each function has one job. Unit-testable independently. Failures in one step don't silently skip others."""
        ),
        (
            "Refactor this class to use properties instead of raw attribute access:\n\n```python\nclass Circle:\n    def __init__(self, radius):\n        self.radius = radius\n        self.area = 3.14159 * radius ** 2\n```",
            """Use a `@property` so area stays in sync when radius changes:

```python
import math

class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property\n    def radius(self):\n        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError('Radius cannot be negative')
        self._radius = value

    @property
    def area(self):
        return math.pi * self._radius ** 2

    @property
    def circumference(self):
        return 2 * math.pi * self._radius
```

**Why:** `area` was a stale cached value before — if `radius` changed, `area` wouldn't update. Properties compute on demand and stay correct."""
        ),
    ]

    for human, gpt in refactors:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 8. DEBUGGING PROCESS (error → root cause → fix)
# ─────────────────────────────────────────────────────────────────────────────

def debugging_pairs():
    samples = []

    debugs = [
        (
            "I'm getting `ModuleNotFoundError: No module named 'anthos'` when running `python train.py`. What do I do?",
            """**Root cause:** Python can't find the `anthos` package because it's not installed and the directory isn't in `sys.path`.

**Fix — Option 1 (development install):**
```bash
pip install -e .
```
Requires a `setup.py` or `pyproject.toml` in the repo root. Installs in editable mode so changes take effect immediately.

**Fix — Option 2 (run from repo root):**
```bash
# Always run from /path/to/anthos-repo, not a subdirectory
cd /Users/dadsmacpro/Desktop/anthos-repo
python train.py
```

**Fix — Option 3 (PYTHONPATH):**
```bash
PYTHONPATH=/Users/dadsmacpro/Desktop/anthos-repo python train.py
```

**Check:** `python -c "import anthos; print(anthos.__file__)"` — if this prints a path, the import works."""
        ),
        (
            "My training loss is NaN after a few steps. How do I debug this?",
            """NaN loss is almost always one of four things:

**1. Exploding gradients — check first:**
```python
# Add this after loss.backward(), before optimizer.step()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if total_norm > 10:
    print(f'Warning: large gradient norm {total_norm:.2f}')
```
Fix: lower learning rate, add gradient clipping.

**2. Bad data — check your dataset:**
```python
for ids, labels in loader:
    if torch.isnan(ids.float()).any():
        print('NaN in input!')
    if (ids >= vocab_size).any():
        print('Token ID out of range!')
    break
```

**3. Learning rate too high:**
Start with `lr=1e-4`, monitor first 100 steps. If loss spikes → explodes, cut lr by 10x.

**4. Mixed precision with incompatible ops:**
If using `autocast`, some ops produce NaN on bfloat16. Try `torch.float32` to isolate.

**Diagnostic loop:**
```python
for step, (ids, labels) in enumerate(loader):
    logits = model(ids)[0]
    if torch.isnan(logits).any():
        print(f'NaN in logits at step {step}')
        break
    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
    if torch.isnan(loss):
        print(f'NaN in loss at step {step}, logits range: {logits.min():.2f}–{logits.max():.2f}')
        break
```"""
        ),
        (
            "I get `CUDA out of memory` during training. How do I fix it?",
            """**Root cause:** batch doesn't fit in GPU VRAM.

**Immediate fixes (in order of impact):**

**1. Reduce batch size:**
```python
# In configs.py or train args
batch_size = 4  # try halving: 8 → 4 → 2
```

**2. Enable gradient checkpointing (trades compute for memory):**
```python
model.gradient_checkpointing_enable()
```

**3. Use mixed precision (bfloat16 uses half the memory of float32):**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    loss = model(ids)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**4. Increase gradient accumulation instead of batch size:**
```python
# Effective batch = batch_size * grad_accum
accum_steps = 8
for step, (ids, labels) in enumerate(loader):
    loss = model(ids) / accum_steps
    loss.backward()
    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**5. Clear cache between batches:**
```python
torch.cuda.empty_cache()
```

**Diagnose:** `nvidia-smi` shows current VRAM usage. Target: leave 10-15% free."""
        ),
        (
            "My Flask app returns 500 errors and I can't see why. How do I debug it?",
            """**Step 1: Enable detailed error output:**
```python
app.run(debug=True)  # shows full traceback in browser
```
Never in production — but essential for debugging.

**Step 2: Check the terminal/logs:**
Flask prints the full traceback to stdout. Read it top-to-bottom, focus on YOUR code (not library internals).

**Step 3: Add logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/chat')
def chat():
    app.logger.debug(f'Request: {request.json}')
    try:
        result = process(request.json)
        app.logger.debug(f'Result: {result}')
        return jsonify(result)
    except Exception as e:
        app.logger.error(f'Error in /chat: {e}', exc_info=True)
        raise  # let Flask return 500 with traceback
```

**Step 4: Test the route in isolation:**
```python
# In Python console or test file
with app.test_client() as c:
    r = c.post('/chat', json={'message': 'test'})
    print(r.status_code)
    print(r.data)
```

**Step 5: Common causes:**
- `None` returned from route (Flask needs a Response or string/dict)
- JSON serialization error (object not JSON-serializable)
- Missing database connection
- Import error in a module loaded lazily by the route"""
        ),
        (
            "My model generates garbage output — just repeating tokens or outputting nonsense. What's wrong?",
            """**Possible causes:**

**1. Temperature too high or too low:**
```python
# Too high (>1.5) → random nonsense
# Too low (near 0) → degenerate repetition
# Good starting range: 0.7–1.0
temperature = 0.8
logits = logits / temperature
```

**2. Missing EOS/stop condition — model never knows to stop:**
```python
eos_id = tokenizer.eos_token_id
# In generation loop:
if next_token == eos_id:
    break
```

**3. Checkpoint from too early in training — model hasn't learned yet:**
Load a later checkpoint. Check training loss: if it never decreased below 3.0, the model didn't learn.

**4. Tokenizer mismatch — wrong tokenizer for the model:**
```python
# Verify
tokenizer = AutoTokenizer.from_pretrained('data/anthos_tokenizer')
print(tokenizer.vocab_size)  # must match cfg.vocab_size
```

**5. model.eval() not called — dropout is adding noise:**
```python
model.eval()  # ALWAYS call before inference
with torch.no_grad():
    ...
```

**6. Repetition penalty not applied:**
```python
# Penalize tokens already generated
for prev_token in generated_ids:
    logits[prev_token] /= repetition_penalty  # e.g. 1.3
```

**Diagnosis:** generate on a simple test prompt like "2 + 2 = " — if the model can't get that right, it hasn't trained enough."""
        ),
        (
            "How do I find which line in my code is causing a slow training loop?",
            """**Profile with Python's built-in tools:**

**Quick method — time individual steps:**
```python
import time

t_data = t_forward = t_backward = t_step = 0

for ids, labels in loader:
    t0 = time.perf_counter()
    # data is ready
    t_data += time.perf_counter() - t0

    t0 = time.perf_counter()
    logits = model(ids)
    t_forward += time.perf_counter() - t0

    t0 = time.perf_counter()
    loss.backward()
    t_backward += time.perf_counter() - t0

    t0 = time.perf_counter()
    optimizer.step()
    t_step += time.perf_counter() - t0

print(f'Data loading: {t_data:.2f}s')
print(f'Forward pass: {t_forward:.2f}s')
print(f'Backward pass: {t_backward:.2f}s')
print(f'Optimizer step: {t_step:.2f}s')
```

**If data loading is slow:** increase `num_workers`, preload into RAM, use faster storage.
**If forward is slow:** model is too large for batch size; use gradient accumulation.
**If backward is slow:** same as forward — reduce memory pressure.

**CUDA profiler (GPU only):**
```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        loss = model(ids)
        loss.backward()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
```"""
        ),
    ]

    for human, gpt in debugs:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 9. CODE REVIEW
# ─────────────────────────────────────────────────────────────────────────────

def code_review_pairs():
    samples = []

    reviews = [
        (
            "Review this login function:\n\n```python\ndef login(username, password):\n    user = db.query(f'SELECT * FROM users WHERE username = \"{username}\"')\n    if user and user['password'] == password:\n        return True\n    return False\n```",
            """**Critical issues — fix immediately:**

**1. SQL Injection (Critical):**
```python
# VULNERABLE — attacker can input: admin\" OR 1=1 --
f'SELECT * FROM users WHERE username = \"{username}\"'

# FIXED — parameterized query:
user = db.query('SELECT * FROM users WHERE username = %s', (username,))
```

**2. Plain text password comparison (Critical):**
Never store or compare passwords in plain text. Use bcrypt:
```python
import bcrypt

def login(username, password):
    user = db.query('SELECT * FROM users WHERE username = %s', (username,))
    if not user:
        return False
    return bcrypt.checkpw(password.encode(), user['password_hash'].encode())
```

**3. Timing attack (Security):**
`user['password'] == password` short-circuits. Use `hmac.compare_digest()` or bcrypt (which is constant-time).

**4. Missing rate limiting:**
Brute-force protection needed. After 5 failures, lock account or add delay.

**Summary:** This code has two critical security vulnerabilities. Do not deploy without the parameterized query and password hashing fixes."""
        ),
        (
            "Review this API handler:\n\n```python\n@app.route('/upload', methods=['POST'])\ndef upload():\n    file = request.files['file']\n    filename = file.filename\n    file.save(f'uploads/{filename}')\n    return 'ok'\n```",
            """**Issues found:**

**1. Path traversal vulnerability (Critical):**
An attacker can upload a file named `../../etc/cron.d/evil` and write anywhere the process has permission.
```python
from werkzeug.utils import secure_filename
filename = secure_filename(file.filename)
```

**2. No file type validation:**
Accepts any file including executables.
```python
ALLOWED = {'png', 'jpg', 'pdf', 'txt'}
ext = filename.rsplit('.', 1)[-1].lower()
if ext not in ALLOWED:
    return 'File type not allowed', 400
```

**3. No size limit:**
Huge uploads can fill disk.
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

**4. Original filename exposed in path:**
Use a UUID for the stored filename:
```python
import uuid
stored_name = f'{uuid.uuid4().hex}.{ext}'
file.save(f'uploads/{stored_name}')
```

**Fixed version:**
```python
from werkzeug.utils import secure_filename
import uuid, os

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or not file.filename:
        return 'No file', 400
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in {'png', 'jpg', 'pdf', 'txt'}:
        return 'File type not allowed', 400
    stored = f'{uuid.uuid4().hex}.{ext}'
    file.save(os.path.join('uploads', stored))
    return jsonify({'file_id': stored})
```"""
        ),
        (
            "Review this PyTorch training loop for correctness:\n\n```python\nmodel = MyModel()\noptimizer = Adam(model.parameters(), lr=0.01)\n\nfor epoch in range(10):\n    for batch in loader:\n        ids, labels = batch\n        logits = model(ids)\n        loss = F.cross_entropy(logits, labels)\n        loss.backward()\n        optimizer.step()\n```",
            """**Issues found:**

**1. Missing `optimizer.zero_grad()` (Bug — gradients accumulate):**
```python
optimizer.zero_grad()  # add before loss.backward()
loss.backward()
optimizer.step()
```

**2. Learning rate 0.01 is very high for transformers (Will cause divergence):**
For language models, typical range is 1e-5 to 5e-4. Start with 3e-4 or 3e-5.
```python
optimizer = Adam(model.parameters(), lr=3e-4)
```

**3. Missing gradient clipping (Exploding gradient risk):**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**4. Missing `model.train()` call:**
If eval() was ever called, dropout remains off.
```python
model.train()
for epoch in range(10):
    ...
```

**5. No loss logging — can't tell if training is working:**
```python
if step % 100 == 0:
    print(f'Epoch {epoch} Step {step} Loss {loss.item():.4f}')
```

**Corrected loop:**
```python
model.train()
for epoch in range(10):
    for step, (ids, labels) in enumerate(loader):
        optimizer.zero_grad()
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f'Epoch {epoch} Step {step} Loss {loss.item():.4f}')
```"""
        ),
    ]

    for human, gpt in reviews:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 10. ANTHOS-SPECIFIC PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

def anthos_specific_pairs():
    samples = []

    anthos = [
        (
            "How do I add a new training tier to Anthos's train.py?",
            """**Step 1 — Add config in `anthos/configs.py`:**
```python
def get_training_config(tier='sft'):
    # ... existing tiers ...
    elif tier == 'my_new_tier':
        train_cfg = TrainConfig(
            dataset   = 'data/my_new_data.jsonl',
            lr        = 2e-5,
            batch_size = 4,
            max_steps  = 5000,
            seq_len    = 512,
        )
```

**Step 2 — Add the `elif` branch in `train.py` where datasets are loaded:**
```python
elif tier == 'my_new_tier':
    local_data = train_cfg.dataset
    if Path(local_data).exists():
        dataset_name = local_data
        max_samples  = 0
        n_lines = sum(1 for _ in open(local_data))
        print(f'  ✓ my_new_tier: loading {local_data} ({n_lines:,} pairs)')
    else:
        raise FileNotFoundError(f'Data not found: {local_data}')
```

**Step 3 — Prepare the dataset:**
```bash
python generate_my_new_data.py --out data/my_new_data.jsonl
```

**Step 4 — Smoke test:**
```bash
python preflight.py --fast
python train.py --tier my_new_tier  # watch first 10 steps
```

**Step 5 — Run:**
```bash
python train.py --tier my_new_tier --resume checkpoints/mansa_sovereign/step_010000.pt
```"""
        ),
        (
            "How does Anthos's Synapse memory system work and how do I use it in my code?",
            """**Synapse** is Anthos's persistent memory engine. It stores:
- **turns**: recent conversation history (capped, then compressed)
- **user_facts**: long-term facts about the user
- **compressed_batches**: summaries of older turns

**Loading:**
```python
from anthos.synapse import Synapse

brain = Synapse.load()  # loads from disk (.anthos/synapse.json)
```

**Adding a turn:**
```python
brain.add_turn('user', 'What is machine learning?')
brain.add_turn('assistant', 'Machine learning is...')
brain.save()
```

**Injecting into a system prompt:**
```python
system_prompt = BASE_SYSTEM + brain.context_injection()
```
`context_injection()` returns a formatted string of recent turns + user facts.

**Storing a user fact:**
```python
brain.add_user_fact('user prefers Python over JavaScript')
brain.save()
```

**Starting a new session:**
```python
brain.new_session()  # clears turns, keeps user_facts
brain.save()
```

**Full chat wiring:**
```python
brain = Synapse.load()
while True:
    user_msg = input('You: ')
    sys_prompt = BASE_SYSTEM + brain.context_injection()
    response = model.generate(sys_prompt, user_msg)
    brain.add_turn('user', user_msg)
    brain.add_turn('assistant', response)
    brain.save()
    print(f'Anthos: {response}')
```"""
        ),
        (
            "How do I interpret Anthos's training loss curve? What numbers are good?",
            """**Loss scale reference for language models:**

| Loss | What it means |
|------|---------------|
| > 8.0 | Random noise — model hasn't learned anything |
| 5.0–8.0 | Early training, just starting to learn |
| 3.0–5.0 | Learning structure and common patterns |
| 2.0–3.0 | Good progress — starting to understand language |
| 1.5–2.0 | Strong learning — model is useful |
| 1.0–1.5 | Very well trained on this data |
| < 1.0 | Likely memorizing / overfitting |

**What to watch for:**

1. **Loss should decrease** in the first 500–1000 steps. If it's flat or going up, your LR is wrong or data is broken.

2. **Sudden spikes then recovery** = gradient instability. Add `clip_grad_norm_(model.parameters(), 1.0)`.

3. **Loss stuck at ~4.0** on Anthos (47M params, SFT tier) is normal — small model, diverse data. The model is still useful for conversation.

4. **Check every 500 steps:**
```bash
tail -f /workspace/train_stdout.txt
```

5. **Healthy Anthos SFT run (47M, 2.46M pairs):**
- Step 0: loss ≈ 10.5 (random)
- Step 500: loss ≈ 4.0–5.0
- Step 2000: loss ≈ 3.0–3.5
- Step 10000: loss ≈ 2.5–3.0"""
        ),
        (
            "How do I add a new tool to Anthos's toolbox so it can answer a new type of question?",
            """**Step 1 — Add detection regex in `anthos/toolbox.py`:**
```python
def detect_tool(query):
    q = query.lower().strip()
    # ... existing patterns ...

    # New tool: weather
    weather_match = re.search(r'weather\\s+(?:in\\s+)?(.+)', q)
    if weather_match:
        return 'weather', weather_match.group(1).strip()

    return None, None
```

**Step 2 — Add the tool function:**
```python
def _run_weather(city):
    try:
        import requests
        r = requests.get(f'https://wttr.in/{city}?format=3', timeout=5)
        return r.text.strip()
    except Exception as e:
        return f'Could not get weather: {e}'
```

**Step 3 — Add to the dispatch in `run_tool`:**
```python
def run_tool(tool, arg):
    if tool == 'calc':
        return _run_calc(arg)
    elif tool == 'time':
        return _run_time()
    elif tool == 'weather':       # ADD THIS
        return _run_weather(arg)  # ADD THIS
    elif tool == 'search':
        return _run_search(arg)
    return f'Unknown tool: {tool}'
```

**Step 4 — Update `format_tool_context`:**
```python
def format_tool_context(tool, result):
    labels = {
        'calc': 'Calculation result',
        'time': 'Current time',
        'weather': 'Weather info',  # ADD THIS
        'search': 'Web search results',
        'wiki': 'Wikipedia summary',
    }
    label = labels.get(tool, 'Tool result')
    return f'[{label}]: {result}\\n'
```

**Step 5 — Smoke test:**
```python
python -c "
from anthos.toolbox import detect_tool, run_tool
tool, arg = detect_tool('what is the weather in New York')
print(f'Detected: {tool}({arg})')
result = run_tool(tool, arg)
print(f'Result: {result}')
"
```"""
        ),
        (
            "How do I run the full preflight check and what does each step mean?",
            """**Run it:**
```bash
cd /Users/dadsmacpro/Desktop/anthos-repo
source venv/bin/activate
python preflight.py          # full (~5 min)
python preflight.py --fast   # skip smoke training (~30 sec)
```

**What each step checks:**

**1. Dataset spot-check** — reads first 500 lines of `sft_master.jsonl`, verifies each is valid JSON with human+gpt turns, checks file > 1GB. Catches: corrupted data, wrong format, empty file.

**2. Tokenizer** — loads `data/anthos_tokenizer`, does encode/decode round-trip. Catches: missing tokenizer directory, vocab mismatch.

**3. Model init + forward pass** — builds Anthos from configs, runs a random input through it. Catches: import errors, architecture bugs, shape mismatches.

**4. Dataloader** — loads 5 real batches from sft_master.jsonl. Catches: dataset format errors, tokenization failures, OOM on tiny CPU.

**5. Smoke training** (skipped with `--fast`) — 100 gradient steps on CPU. Catches: loss not decreasing, optimizer bugs, gradient issues.

**6. Checkpoint round-trip** (skipped with `--fast`) — saves and reloads a checkpoint. Catches: serialization bugs, missing keys.

**7. Inference stack** — imports and tests Synapse, toolbox, multilingual detector, retrieval heuristic. Catches: missing pip packages, import errors in any runtime module.

**Interpreting output:**
- `✓` = passed
- `✗` = failed — **fix before training**
- `⚠` = warning — may still work, but investigate"""
        ),
        (
            "How do I merge a new JSONL dataset into Anthos's sft_master.jsonl?",
            """**Check the new file first:**
```bash
# Count lines
wc -l data/new_dataset.jsonl

# Spot check format (first 3 lines)
head -3 data/new_dataset.jsonl | python3 -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"

# Verify it has the right structure
python3 -c "
import json
bad = 0
with open('data/new_dataset.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        convs = obj.get('conversations', [])
        if not any(c['from'] in ('human','user') for c in convs):
            bad += 1
print(f'Bad lines: {bad}')
"
```

**Merge:**
```bash
cat data/new_dataset.jsonl >> data/sft_master.jsonl
```

**Verify the merge:**
```bash
wc -l data/sft_master.jsonl    # line count increased
du -sh data/sft_master.jsonl   # file size looks right
tail -3 data/sft_master.jsonl  # new data is at the end
```

**Re-validate:**
```bash
python validate_dataset.py --file data/sft_master.jsonl
```

**Important:** the training loop reads the whole file and shuffles, so appending order doesn't matter for training quality."""
        ),
    ]

    for human, gpt in anthos:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 11. GENERAL PROGRAMMING CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────

def concepts_pairs():
    samples = []

    concepts = [
        ("What is the difference between a list and a generator in Python?",
         "A **list** stores all values in memory at once — `[x**2 for x in range(1000000)]` allocates memory for a million integers immediately. A **generator** yields values lazily one at a time — `(x**2 for x in range(1000000))` uses constant memory regardless of size. Use lists when you need to index/slice or iterate multiple times. Use generators when you process large data streams once."),
        ("What is the difference between `==` and `is` in Python?",
         "`==` compares **value** (calls `__eq__`). `is` compares **identity** (same object in memory). `[1,2] == [1,2]` is `True` (same content). `[1,2] is [1,2]` is `False` (two different list objects). Use `is` only for `None`, `True`, `False` checks — `if x is None:` — never for value comparison."),
        ("What does `*args` and `**kwargs` do in Python?",
         "`*args` collects extra positional arguments into a tuple. `**kwargs` collects extra keyword arguments into a dict. Example: `def fn(*args, **kwargs): print(args, kwargs)` called as `fn(1, 2, key='val')` gives `(1, 2)` and `{'key': 'val'}`. Used to write functions that accept any number of arguments, or to pass through arguments to another function."),
        ("What is the GIL in Python and why does it matter?",
         "The Global Interpreter Lock (GIL) is a mutex in CPython that allows only one thread to execute Python bytecode at a time. This means Python threads don't achieve true CPU parallelism for CPU-bound work. For I/O-bound work (network, disk), threads are still effective because the GIL is released during I/O. For CPU-bound work, use `multiprocessing` (separate processes, no shared GIL) or write the hot loop in C (NumPy/PyTorch bypass the GIL)."),
        ("What is the difference between `deepcopy` and `copy` in Python?",
         "`copy.copy()` creates a **shallow copy** — a new container but the elements inside are still references to the same objects. `copy.deepcopy()` creates a **deep copy** — recursively copies every nested object. Example: `copy([[1,2],[3,4]])` gives two lists sharing the inner lists; mutating `inner[0]` affects both. `deepcopy` gives fully independent nested objects."),
        ("Explain Big O notation with examples.",
         "Big O describes how an algorithm's runtime grows as input size `n` increases.\n- **O(1)** — constant: dict lookup, list append. Same time regardless of n.\n- **O(log n)** — logarithmic: binary search. Doubles input → one more step.\n- **O(n)** — linear: iterating a list. Doubles input → doubles time.\n- **O(n log n)** — linearithmic: merge sort, Python's `sorted()`.\n- **O(n²)** — quadratic: nested loops over same data. 1000 items → 1,000,000 ops.\n- **O(2^n)** — exponential: naive Fibonacci. Avoid for large n.\nPick the algorithm with the lowest Big O for your bottleneck."),
        ("What is the difference between synchronous and asynchronous code?",
         "**Synchronous** code executes line by line — each operation must complete before the next starts. If you fetch 10 URLs synchronously, you wait for URL 1, then URL 2, etc. Total time = sum of all fetch times.\n\n**Asynchronous** code (async/await) can start an operation and immediately move on to another while waiting. Fetching 10 URLs async fires all 10 simultaneously — total time ≈ time of the slowest single fetch.\n\nPython's `asyncio` is single-threaded but uses an event loop to juggle many pending operations. Best for I/O-bound concurrency (HTTP, database, file I/O). For CPU-bound work, async doesn't help — use multiprocessing."),
        ("What is dependency injection and why use it?",
         "Dependency injection means passing a component's dependencies into it rather than having it create them internally.\n\n**Without DI (hard to test):**\n```python\nclass UserService:\n    def __init__(self):\n        self.db = PostgresDB()  # hardcoded\n```\n\n**With DI (easy to test):**\n```python\nclass UserService:\n    def __init__(self, db):  # injected\n        self.db = db\n\n# Production\nservice = UserService(PostgresDB())\n# Testing\nservice = UserService(FakeDB())\n```\n\nBenefits: swap implementations, mock in tests, make dependencies explicit and auditable."),
        ("What is the difference between a process and a thread?",
         "A **process** is an independent program with its own memory space. Processes don't share memory — communication requires IPC (pipes, sockets, shared memory). Safer but slower to create and communicate.\n\nA **thread** is a lightweight unit of execution within a process, sharing the same memory. Faster to create and communicate, but shared memory means race conditions.\n\n**In Python:** `multiprocessing` for CPU-bound parallelism (bypasses GIL), `threading` for I/O-bound concurrency."),
        ("What is memoization and when should I use it?",
         "Memoization caches the result of expensive function calls so repeated calls with the same arguments return the cached result instead of recomputing.\n\n```python\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef expensive(n):\n    # Only computed once per unique n\n    return sum(range(n))\n```\n\n**Use when:** the function is pure (same inputs always give same output), called repeatedly with the same args, and the computation is expensive. **Don't use when:** the function has side effects, accesses external state (current time, DB), or the result set is too large to fit in memory."),
    ]

    for human, gpt in concepts:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 12. BASH / SHELL
# ─────────────────────────────────────────────────────────────────────────────

def bash_pairs():
    samples = []

    bash = [
        ("What does `2>&1` mean in a bash command?",
         "`2>&1` redirects stderr (file descriptor 2) to wherever stdout (fd 1) is currently pointing. Combined with `> file.log`, the command `cmd > file.log 2>&1` sends both stdout and stderr to `file.log`. Order matters: `cmd 2>&1 > file.log` redirects stderr to the screen (stdout's original destination) then redirects stdout to the file — stderr still goes to screen."),
        ("What is the difference between `&&` and `;` in bash?",
         "`&&` runs the second command only if the first succeeds (exit code 0). `cmd1 && cmd2` — if cmd1 fails, cmd2 is skipped. `;` always runs both commands regardless of exit code. `cmd1 ; cmd2` — cmd2 runs even if cmd1 fails. Use `&&` to chain commands that depend on each other (like `make && ./run`), `;` for independent sequential commands."),
        ("How do I find all Python files in a directory and grep for a string?",
         "```bash\n# Find all .py files and grep for 'import torch'\ngrep -r 'import torch' --include='*.py' .\n\n# With line numbers\ngrep -rn 'import torch' --include='*.py' .\n\n# Using find + xargs (more control)\nfind . -name '*.py' -print0 | xargs -0 grep -l 'import torch'\n```"),
        ("How do I run a Python script in the background and capture its output?",
         "```bash\n# Run in background, capture stdout+stderr to log file\nnohup python3 train.py > train.log 2>&1 &\necho \"PID: $!\"\n\n# Follow the log live\ntail -f train.log\n\n# Check if it's still running\nps aux | grep train.py\n\n# Stop it\nkill $(cat train.pid)  # if you saved the PID\n```"),
        ("How do I check disk usage and find large files in bash?",
         "```bash\n# Total disk usage of current directory\ndu -sh .\n\n# Size of each file/dir sorted by size\ndu -sh * | sort -h\n\n# Find files larger than 1GB\nfind . -size +1G -type f\n\n# Find largest files anywhere\nfind . -type f -exec du -sh {} + | sort -rh | head -20\n\n# Overall disk space\ndf -h\n```"),
        ("How do I safely delete files matching a pattern in bash?",
         "```bash\n# ALWAYS test with echo first (dry run)\nfind . -name '*.pyc' -print\n\n# Once confirmed, delete\nfind . -name '*.pyc' -delete\n\n# Or with xargs\nfind . -name '*.pyc' -print0 | xargs -0 rm -f\n\n# Remove __pycache__ directories\nfind . -type d -name '__pycache__' -exec rm -rf {} +\n```\n**Rule:** always run with `-print` first to see what would be deleted before actually deleting."),
    ]

    for human, gpt in bash:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 13. GIT WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def git_pairs():
    samples = []

    git = [
        ("How do I undo the last git commit without losing my changes?",
         "```bash\ngit reset --soft HEAD~1\n```\nThis moves HEAD back one commit but keeps your changes staged. The files are unchanged — only the commit is undone. You can then edit and recommit.\n\n`--soft` = keep staged. `--mixed` (default) = keep unstaged. `--hard` = discard changes entirely (destructive)."),
        ("How do I see what changed in a specific file over the last 5 commits?",
         "```bash\ngit log -5 --oneline -- path/to/file.py  # last 5 commits touching this file\ngit diff HEAD~3..HEAD -- path/to/file.py   # diff over last 3 commits\ngit log -p --follow -- path/to/file.py     # full patch history\n```"),
        ("How do I stash my current changes to switch branches?",
         "```bash\n# Save current changes (including untracked with -u)\ngit stash push -u -m 'wip: feature x'\n\n# Switch branch\ngit checkout main\n\n# Later, restore\ngit stash pop       # apply and drop the stash\n# or\ngit stash apply     # apply but keep the stash\n\n# List stashes\ngit stash list\n```"),
        ("How do I squash my last 3 commits into one before pushing?",
         "```bash\ngit rebase -i HEAD~3\n```\nIn the editor that opens, change `pick` to `squash` (or `s`) on the 2nd and 3rd lines, keep `pick` on the first. Save and close. Edit the combined commit message in the next prompt.\n\n**Only do this on commits that haven't been pushed to a shared branch.**"),
        ("How do I find which commit introduced a bug?",
         "```bash\ngit bisect start\ngit bisect bad          # current commit is broken\ngit bisect good v1.0    # this tag/commit was good\n\n# Git checks out the midpoint — test it\npython test_the_feature.py\ngit bisect good   # or: git bisect bad\n\n# Repeat until git identifies the exact commit\ngit bisect reset  # done — go back to HEAD\n```\nBinary search through history — finds the bug commit in O(log n) steps."),
    ]

    for human, gpt in git:
        samples.append(pair(human, gpt))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/code_sft.jsonl')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('Generating code intelligence SFT dataset...')

    all_pairs = []

    generators = [
        ('code reading',          code_reading_pairs),
        ('bug identification',    bug_identification_pairs),
        ('bug fixing',            bug_fixing_pairs),
        ('code generation',       code_generation_pairs),
        ('integration & wiring',  integration_pairs),
        ('smoke testing',         smoke_testing_pairs),
        ('refactoring',           refactoring_pairs),
        ('debugging process',     debugging_pairs),
        ('code review',           code_review_pairs),
        ('Anthos-specific',       anthos_specific_pairs),
        ('concepts',              concepts_pairs),
        ('bash / shell',          bash_pairs),
        ('git workflow',          git_pairs),
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
    print('\nTo merge into sft_master.jsonl:')
    print(f'  cat {out_path} >> data/sft_master.jsonl')


if __name__ == '__main__':
    main()
