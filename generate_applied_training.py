#!/usr/bin/env python3
"""
generate_applied_training.py — Generate "know it → use it" applied training pairs

Philosophy: Reading a concept is level 1. Implementing it is level 2. Applying it
to a real user problem is level 3. This script generates level 2 and 3 pairs
across Anthos's key domains so the model doesn't just describe things — it builds them.

Domains covered:
  - Python / data engineering
  - AI / ML pipelines (RAG, agents, fine-tuning)
  - System design and APIs
  - Math / algorithmic problem solving
  - Business logic and automation

Usage:
    python generate_applied_training.py
    python generate_applied_training.py --n 1000 --output data/applied_sft.jsonl
"""

import json
import random
import argparse
from pathlib import Path

OUTPUT = Path("data/applied_sft.jsonl")

ANTHOS_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "When someone asks you to build, write, or implement something — you do it. "
    "You produce working code, real plans, and actual outputs. "
    "You don't describe what could be done. You show what to do."
)

ANTHOS_CODE_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are an expert Python developer. When asked to write code, you write "
    "complete, working implementations — not outlines or pseudocode. "
    "Include imports. Make it runnable."
)

ANTHOS_ML_SYSTEM = (
    "You are Anthos, an AI assistant created by Brian Tushae Thomas. "
    "You are built from scratch as a custom AI model. You understand how to train, "
    "fine-tune, evaluate, and deploy language models. When asked to implement "
    "something in ML, you write the actual code."
)


# ─────────────────────────────────────────────────────────────────────────────
# Domain: Python / Data Engineering — "build it" pairs
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_TASKS = [
    (
        "Write a Python function that reads a JSONL file and returns a list of dicts.",
        '''def read_jsonl(path: str) -> list[dict]:
    import json
    from pathlib import Path
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

# Usage
data = read_jsonl("data/training.jsonl")
print(f"Loaded {len(data):,} records")'''
    ),
    (
        "Write a Python script that walks a folder, finds all PDF files, and prints their names and sizes.",
        '''from pathlib import Path

def find_pdfs(folder: str) -> list[dict]:
    root = Path(folder)
    results = []
    for pdf in sorted(root.rglob("*.pdf")):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        results.append({"name": pdf.name, "path": str(pdf), "size_mb": round(size_mb, 2)})
    return results

pdfs = find_pdfs(".")
for p in pdfs:
    print(f"{p['size_mb']:6.2f} MB  {p['name']}")
print(f"\\nTotal: {len(pdfs)} PDFs")'''
    ),
    (
        "Write a Python function to chunk a long string into overlapping segments for RAG.",
        '''def chunk_with_overlap(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks with overlap so context isn\'t lost at boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step back by overlap amount
    return chunks

text = "Your long document text here..."
chunks = chunk_with_overlap(text, chunk_size=500, overlap=50)
print(f"{len(chunks)} chunks generated")'''
    ),
    (
        "Write a Python class that wraps a FAISS index for simple vector search.",
        '''import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        import faiss
        self.dim   = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embedding: np.ndarray, text: str):
        vec = np.array([embedding], dtype="float32")
        self.index.add(vec)
        self.texts.append(text)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[float, str]]:
        vec = np.array([query_embedding], dtype="float32")
        distances, indices = self.index.search(vec, k)
        return [(float(distances[0][i]), self.texts[indices[0][i]])
                for i in range(k) if indices[0][i] >= 0]

# Usage
store = VectorStore(dim=768)
# store.add(embedding_vector, "Document text...")
# results = store.search(query_vector, k=3)'''
    ),
    (
        "Write a Python function that retries a function call with exponential backoff.",
        '''import time
import functools

def retry(max_attempts: int = 3, base_delay: float = 1.0, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, base_delay=1.0, exceptions=(ConnectionError, TimeoutError))
def fetch_data(url: str) -> dict:
    import urllib.request, json
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())'''
    ),
    (
        "Write a Python script that converts a folder of .docx files to plain text.",
        '''from pathlib import Path
import docx

def docx_to_text(docx_path: str) -> str:
    doc = docx.Document(docx_path)
    return "\\n\\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

def convert_folder(folder: str, output_folder: str = None):
    src = Path(folder)
    dst = Path(output_folder) if output_folder else src / "text_output"
    dst.mkdir(exist_ok=True)

    for docx_file in sorted(src.glob("*.docx")):
        text = docx_to_text(str(docx_file))
        out  = dst / (docx_file.stem + ".txt")
        out.write_text(text)
        print(f"  {docx_file.name} → {out.name} ({len(text):,} chars)")

convert_folder("./docs")'''
    ),
    (
        "Write a Python function that merges multiple JSONL files into one, deduplicating by a key field.",
        '''import json
from pathlib import Path

def merge_jsonl(files: list[str], output: str, dedup_key: str = None) -> int:
    seen = set()
    kept = 0
    with open(output, "w") as out:
        for file_path in files:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if dedup_key:
                        key = str(obj.get(dedup_key, ""))
                        if key in seen:
                            continue
                        seen.add(key)
                    out.write(line + "\\n")
                    kept += 1
    return kept

kept = merge_jsonl(
    ["data/set1.jsonl", "data/set2.jsonl", "data/set3.jsonl"],
    "data/merged.jsonl",
    dedup_key="id"
)
print(f"Merged: {kept:,} unique records")'''
    ),
    (
        "Write a Python async function that fetches multiple URLs concurrently and returns their text.",
        '''import asyncio
import aiohttp

async def fetch_all(urls: list[str], timeout: int = 10) -> list[dict]:
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url, timeout) for url in urls]
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            results.append(result)
    return results

async def fetch_one(session, url: str, timeout: int) -> dict:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            text = await resp.text()
            return {"url": url, "status": resp.status, "text": text[:1000]}
    except Exception as e:
        return {"url": url, "error": str(e)}

# Usage
urls = ["https://example.com", "https://httpbin.org/get"]
results = asyncio.run(fetch_all(urls))
for r in results:
    print(r["url"], r.get("status", "ERROR"))'''
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Domain: AI / ML — "implement it" pairs
# ─────────────────────────────────────────────────────────────────────────────

ML_TASKS = [
    (
        "Write Python code to build a simple RAG pipeline: load text, embed it, store in FAISS, then query it.",
        '''# Simple RAG pipeline using sentence-transformers + FAISS
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json

model   = SentenceTransformer("all-MiniLM-L6-v2")
dim     = 384
index   = faiss.IndexFlatL2(dim)
store   = []

def add_document(text: str, metadata: dict = None):
    embedding = model.encode([text])[0].astype("float32")
    index.add(np.array([embedding]))
    store.append({"text": text, "meta": metadata or {}})

def query(question: str, k: int = 3) -> list[dict]:
    embedding = model.encode([question])[0].astype("float32")
    distances, indices = index.search(np.array([embedding]), k)
    return [{"score": float(distances[0][i]), **store[indices[0][i]]}
            for i in range(k) if indices[0][i] >= 0]

# Load and index documents
docs = ["Python is a high-level programming language.", "RAG combines retrieval with generation.",
        "FAISS is a library for efficient similarity search.", "Anthos is a custom AI model."]
for doc in docs:
    add_document(doc)

# Query
results = query("What is RAG?")
for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")'''
    ),
    (
        "Write a Python training loop for a simple language model using PyTorch.",
        '''import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: str = "cpu") -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        labels     = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)                          # (B, T, vocab)
        loss   = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100                              # ignore padding
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

# Training loop
# for epoch in range(num_epochs):
#     loss = train_epoch(model, train_loader, optimizer, device)
#     print(f"Epoch {epoch+1} | Loss: {loss:.4f}")'''
    ),
    (
        "Write Python code to fine-tune a HuggingFace model using the Trainer API.",
        '''from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

MODEL_NAME = "microsoft/phi-2"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

# Build dataset from your JSONL
import json
raw = [json.loads(l) for l in open("data/sft_master.jsonl")][:1000]  # sample
texts = [" ".join(m["value"] for m in ex["conversations"]) for ex in raw]
dataset = Dataset.from_dict({"text": texts}).map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

args = TrainingArguments(
    output_dir="checkpoints/phi-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    save_steps=500,
    logging_steps=50,
    fp16=False,  # CPU: keep False
)

trainer = Trainer(model=model, args=args,
                  train_dataset=dataset["train"], eval_dataset=dataset["test"])
trainer.train()'''
    ),
    (
        "Write a Python function to evaluate a language model's perplexity on a dataset.",
        '''import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_perplexity(model_name: str, texts: list[str], max_length: int = 512) -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"]
            labels    = input_ids.clone()

            outputs   = model(input_ids, labels=labels)
            loss      = outputs.loss.item()
            n_tokens  = input_ids.numel()

            total_loss   += loss * n_tokens
            total_tokens += n_tokens

    avg_loss    = total_loss / total_tokens
    perplexity  = math.exp(avg_loss)
    return perplexity

# ppl = compute_perplexity("gpt2", ["The quick brown fox jumps over the lazy dog."] * 10)
# print(f"Perplexity: {ppl:.2f}")'''
    ),
    (
        "Write a Python class that manages a simple tool-calling AI agent loop.",
        '''import json
from typing import Callable

class SimpleAgent:
    """Agent loop: LLM → tool call → result → LLM → ... → final answer."""

    def __init__(self, llm_fn: Callable, tools: dict[str, Callable], max_steps: int = 5):
        self.llm      = llm_fn    # fn(messages) -> str
        self.tools    = tools     # {"tool_name": callable}
        self.max_steps = max_steps

    def run(self, user_message: str) -> str:
        messages = [{"role": "user", "content": user_message}]
        tool_desc = "\\n".join(f"- {name}: {fn.__doc__ or name}" for name, fn in self.tools.items())
        system = f"You have access to these tools:\\n{tool_desc}\\n\\nTo use a tool, respond with JSON: {{\"tool\": \"name\", \"args\": {{...}}}}\\nWhen done, respond with your final answer."
        messages.insert(0, {"role": "system", "content": system})

        for step in range(self.max_steps):
            response = self.llm(messages)
            messages.append({"role": "assistant", "content": response})
            try:
                call = json.loads(response)
                tool_name = call.get("tool")
                args      = call.get("args", {})
                if tool_name and tool_name in self.tools:
                    result = str(self.tools[tool_name](**args))
                    messages.append({"role": "user", "content": f"Tool result: {result}"})
                    continue
            except json.JSONDecodeError:
                return response  # Final answer (not JSON)
        return response'''
    ),
    (
        "Write Python code to implement cosine similarity search without FAISS.",
        '''import numpy as np

class SimpleSimilaritySearch:
    def __init__(self):
        self.embeddings = []
        self.documents  = []

    def add(self, embedding: np.ndarray, document: str):
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        self.embeddings.append(embedding / norm if norm > 0 else embedding)
        self.documents.append(document)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[float, str]]:
        if not self.embeddings:
            return []
        query_norm = np.linalg.norm(query_embedding)
        q = query_embedding / query_norm if query_norm > 0 else query_embedding
        matrix = np.array(self.embeddings)
        scores  = matrix @ q                # dot product = cosine similarity (pre-normalized)
        top_k   = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.documents[i]) for i in top_k]

# Usage example
store = SimpleSimilaritySearch()
store.add(np.random.randn(384), "Anthos is a custom AI model")
store.add(np.random.randn(384), "Python is great for data science")
results = store.search(np.random.randn(384), k=2)
for score, doc in results:
    print(f"{score:.3f}  {doc}")'''
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Domain: Applied "real problem" → solution pairs
# These are 3rd-level: someone brings a REAL problem, Anthos solves it
# ─────────────────────────────────────────────────────────────────────────────

APPLIED_PROBLEMS = [
    (
        "I have 1 million JSONL training examples and want to shuffle them randomly without loading them all into memory. How?",
        '''For a 1M-line file that won\'t fit in RAM, use reservoir sampling or a two-pass approach:

**Option 1: Line-index shuffle (low memory)**
```python
import random, json
from pathlib import Path

def shuffle_jsonl(input_path: str, output_path: str, seed: int = 42):
    random.seed(seed)
    # Pass 1: collect byte offsets of each line
    offsets = []
    with open(input_path, "rb") as f:
        while True:
            offset = f.tell()
            line   = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)

    random.shuffle(offsets)

    # Pass 2: write in shuffled order
    with open(input_path, "rb") as f_in, open(output_path, "w") as f_out:
        for offset in offsets:
            f_in.seek(offset)
            f_out.write(f_in.readline().decode())

    print(f"Shuffled {len(offsets):,} lines → {output_path}")

shuffle_jsonl("data/sft_master.jsonl", "data/sft_shuffled.jsonl")
```

**Option 2: Split → shuffle → merge** (simpler but uses temp disk)
```python
import subprocess
# GNU shuf on Linux, or gshuf on Mac via homebrew
subprocess.run(["shuf", "data/sft_master.jsonl", "-o", "data/sft_shuffled.jsonl"])
```

For macOS: `brew install coreutils` then `gshuf`.

Option 1 is the safest — works everywhere, uses ~8 bytes per line of RAM.'''
    ),
    (
        "I want to build a chatbot that answers questions from my PDF files. Give me the full implementation.",
        '''Here\'s a complete local RAG chatbot that reads your PDFs and answers questions:

```python
#!/usr/bin/env python3
# pdf_chatbot.py — RAG chatbot over local PDFs
# pip install pdfplumber sentence-transformers faiss-cpu ollama

import pdfplumber, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import ollama

MODEL   = SentenceTransformer("all-MiniLM-L6-v2")
DIM     = 384
INDEX   = faiss.IndexFlatL2(DIM)
CHUNKS  = []

def load_pdf(path: str, chunk_size: int = 400):
    with pdfplumber.open(path) as pdf:
        text = " ".join(page.extract_text() or "" for page in pdf.pages)
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk) > 100:
            CHUNKS.append({"text": chunk, "source": Path(path).name})
            vec = MODEL.encode([chunk])[0].astype("float32")
            INDEX.add(np.array([vec]))

def retrieve(query: str, k: int = 3) -> str:
    vec = MODEL.encode([query])[0].astype("float32")
    _, indices = INDEX.search(np.array([vec]), k)
    return "\\n\\n".join(CHUNKS[i]["text"] for i in indices[0] if i >= 0)

def chat(question: str) -> str:
    context = retrieve(question)
    response = ollama.chat(model="llama3.2", messages=[
        {"role": "system", "content": f"Answer using this context:\\n\\n{context}"},
        {"role": "user",   "content": question}
    ])
    return response["message"]["content"]

# Load all PDFs in current folder
for pdf in Path(".").glob("*.pdf"):
    print(f"Loading {pdf.name}...")
    load_pdf(str(pdf))

print(f"Indexed {len(CHUNKS)} chunks from {len(list(Path(\'.\').glob(\'*.pdf\')))} PDFs")

# Chat loop
while True:
    q = input("\\nYou: ").strip()
    if q.lower() in ("quit", "exit"):
        break
    print("Anthos:", chat(q))
```

Run: `python pdf_chatbot.py` in the folder with your PDFs.'''
    ),
    (
        "I have a Python script that takes 10 minutes to process files one at a time. How do I parallelize it?",
        '''Use `concurrent.futures.ProcessPoolExecutor` for CPU-bound work or `ThreadPoolExecutor` for I/O-bound work:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def process_file(path: str) -> dict:
    """Your existing processing logic — put it here."""
    import time
    time.sleep(0.1)  # simulate work
    return {"file": path, "result": f"processed {Path(path).name}"}

def process_all_parallel(folder: str, max_workers: int = 4) -> list[dict]:
    files = [str(p) for p in Path(folder).glob("*") if p.is_file()]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"Done: {result[\'file\']}")
            except Exception as e:
                print(f"Failed {futures[future]}: {e}")

    return results

results = process_all_parallel("./data", max_workers=4)
print(f"Processed {len(results)} files")
```

**Rules of thumb:**
- I/O-bound (reading files, network requests) → `ThreadPoolExecutor`
- CPU-bound (data processing, encoding) → `ProcessPoolExecutor`
- For pandas/numpy heavy work → use `ProcessPoolExecutor` with `max_workers=os.cpu_count()`

With 4 workers, 10 minutes → ~2.5 minutes. With 8 workers → ~1.25 minutes.'''
    ),
    (
        "My PyTorch model is training too slowly on CPU. What can I do without a GPU?",
        '''Several ways to speed up CPU training significantly:

**1. Reduce numerical precision (use bfloat16 if supported)**
```python
# Check if your CPU supports bfloat16
import torch
print(torch.backends.cpu.get_cpu_capability())  # 'AVX2' or 'AVX512'
# If AVX2+, bfloat16 matmuls are ~2x faster on modern Intel/AMD
model = model.to(torch.bfloat16)
```

**2. Enable torch.compile() (PyTorch 2.0+)**
```python
model = torch.compile(model)  # ~10-30% speedup, first batch is slow
```

**3. Increase batch size (fill CPU cache)**
```python
# Bigger batches → fewer Python loop iterations → more time in C++
per_device_batch_size = 32  # try 16, 32, 64 — watch RAM
```

**4. Use DataLoader workers**
```python
loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=False)
# num_workers=0 on macOS if you get errors; 4 on Linux
```

**5. Gradient accumulation (simulate larger batch)**
```python
accumulation_steps = 8  # effective batch = batch_size * 8
optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**6. Flash Attention alternative for CPU**
```python
# Use scaled_dot_product_attention with math backend (avoids loops)
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):
    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

Realistically: if you need to train a 70M+ param model, rent a RunPod A100 for $1-2/hour. CPU training beyond proof-of-concept isn\'t worth the time.'''
    ),
    (
        "Build me a script that automatically generates training data from any text document.",
        '''```python
#!/usr/bin/env python3
# doc_to_training.py — Convert any text document into SFT training pairs
# Works with .txt, .md, .pdf (if pdfplumber installed)

import json, re
from pathlib import Path

SYSTEM = "You are Anthos, an AI assistant created by Brian Tushae Thomas. Answer clearly and helpfully."

def extract_text(path: str) -> str:
    p = Path(path)
    if p.suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\\n\\n".join(pg.extract_text() or "" for pg in pdf.pages)
    return p.read_text(errors="ignore")

def chunk(text: str, size: int = 600, overlap: int = 80) -> list[str]:
    paras = [p.strip() for p in text.split("\\n\\n") if len(p.strip()) > 60]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) > size:
            if len(buf) > 100:
                chunks.append(buf.strip())
            buf = p
        else:
            buf = buf + "\\n\\n" + p if buf else p
    if len(buf) > 100:
        chunks.append(buf.strip())
    return chunks

def make_pairs(chunks: list[str], topic: str) -> list[dict]:
    pairs = []
    for i, chunk in enumerate(chunks):
        first_line = chunk.splitlines()[0][:70].strip()
        q = f"Explain this from {topic}: {first_line}"
        pairs.append({
            "conversations": [
                {"from": "system", "value": SYSTEM},
                {"from": "human",  "value": q},
                {"from": "gpt",    "value": chunk[:2000]},
            ]
        })
    return pairs

def convert(input_path: str, output_path: str = None):
    path   = Path(input_path)
    output = Path(output_path) if output_path else path.with_suffix(".jsonl")
    topic  = path.stem.replace("_", " ").replace("-", " ")

    text  = extract_text(input_path)
    text  = re.sub(r\'\\n{3,}\', \'\\n\\n\', text).strip()
    pairs = make_pairs(chunk(text), topic)

    with open(output, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\\n")

    print(f"{path.name} → {len(pairs)} pairs → {output}")
    return pairs

# Usage
import sys
if __name__ == "__main__":
    for doc in sys.argv[1:]:
        convert(doc)
```

Run it on any file:
```bash
python doc_to_training.py my_book.pdf notes.md report.txt
```'''
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Domain: System design — "design it" pairs
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_TASKS = [
    (
        "Write a Python script that sends a daily email report with a summary of a CSV file.",
        '''import smtplib, csv, os
from email.mime.text import MIMEText
from datetime import date

def summarize_csv(path: str) -> str:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    cols  = list(rows[0].keys()) if rows else []
    return f"Date: {date.today()}\\nRows: {total:,}\\nColumns: {', '.join(cols)}\\nSample: {rows[0] if rows else 'empty'}"

def send_report(to: str, subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = os.environ["EMAIL_FROM"]
    msg["To"]      = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(os.environ["EMAIL_FROM"], os.environ["EMAIL_PASSWORD"])
        smtp.send_message(msg)

summary = summarize_csv("data/daily_report.csv")
send_report("brian@example.com", f"Daily Report {date.today()}", summary)
print("Report sent.")'''
    ),
    (
        "Write a Python function to scrape product prices from a webpage and save them to CSV.",
        '''import csv, time, random
import urllib.request
from html.parser import HTMLParser

class PriceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.prices = []
        self._in_price = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")
        if "price" in cls.lower():
            self._in_price = True

    def handle_data(self, data):
        if self._in_price and data.strip().startswith("$"):
            self.prices.append(data.strip())
            self._in_price = False

def scrape_prices(url: str) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    parser = PriceParser()
    parser.feed(html)
    return parser.prices

def save_to_csv(prices: list[str], output: str):
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["price"])
        writer.writerows([[p] for p in prices])

prices = scrape_prices("https://example.com/products")
save_to_csv(prices, "prices.csv")
print(f"Saved {len(prices)} prices")'''
    ),
    (
        "Write a Python script that monitors a folder for new files and processes each one automatically.",
        '''import time
from pathlib import Path

WATCH_FOLDER = Path("./incoming")
DONE_FOLDER  = Path("./processed")
WATCH_FOLDER.mkdir(exist_ok=True)
DONE_FOLDER.mkdir(exist_ok=True)

def process_file(path: Path):
    """Your custom processing logic here."""
    print(f"Processing: {path.name}")
    content = path.read_text(errors="ignore")
    # Example: count words
    word_count = len(content.split())
    result = DONE_FOLDER / (path.stem + "_result.txt")
    result.write_text(f"Source: {path.name}\\nWords: {word_count}")
    path.rename(DONE_FOLDER / path.name)   # move to done
    print(f"  Done → {result.name}")

seen = set()
print(f"Watching {WATCH_FOLDER}... (Ctrl+C to stop)")
while True:
    for file in WATCH_FOLDER.iterdir():
        if file.is_file() and str(file) not in seen:
            seen.add(str(file))
            try:
                process_file(file)
            except Exception as e:
                print(f"  ERROR {file.name}: {e}")
    time.sleep(2)'''
    ),
    (
        "Write a Python CLI tool with argument parsing that converts between JSON and CSV formats.",
        '''#!/usr/bin/env python3
"""json2csv — Convert between JSON and CSV formats."""
import argparse, csv, json, sys
from pathlib import Path

def json_to_csv(src: str, dst: str):
    data = json.loads(Path(src).read_text())
    if isinstance(data, dict):
        data = [data]
    if not data:
        print("Empty input"); return
    with open(dst, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"JSON → CSV: {len(data)} rows → {dst}")

def csv_to_json(src: str, dst: str, pretty: bool = False):
    with open(src) as f:
        rows = list(csv.DictReader(f))
    indent = 2 if pretty else None
    Path(dst).write_text(json.dumps(rows, indent=indent))
    print(f"CSV → JSON: {len(rows)} rows → {dst}")

parser = argparse.ArgumentParser(description="Convert JSON ↔ CSV")
parser.add_argument("input",  help="Input file (.json or .csv)")
parser.add_argument("output", help="Output file (.csv or .json)")
parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
args = parser.parse_args()

ext = Path(args.input).suffix.lower()
if ext == ".json":
    json_to_csv(args.input, args.output)
elif ext == ".csv":
    csv_to_json(args.input, args.output, args.pretty)
else:
    print(f"Unsupported format: {ext}")
    sys.exit(1)'''
    ),
]

MORE_PYTHON_TASKS = [
    (
        "Write a Python decorator that logs how long any function takes to run.",
        '''import time, functools, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

@timer
def slow_function(n: int) -> int:
    time.sleep(0.1)
    return sum(range(n))

result = slow_function(1_000_000)
# Logs: slow_function took 0.112s'''
    ),
    (
        "Write a Python context manager that catches and logs exceptions without crashing the program.",
        '''import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.ERROR)

@contextmanager
def safe_run(label: str = "operation"):
    try:
        yield
    except Exception as e:
        logging.error(f"{label} failed: {type(e).__name__}: {e}")

# Usage — program keeps running even if each block fails
with safe_run("reading config"):
    config = open("config.json").read()   # might fail

with safe_run("fetching data"):
    import urllib.request
    data = urllib.request.urlopen("http://example.com", timeout=5).read()

with safe_run("processing results"):
    results = [1, 2, 3][99]              # IndexError — caught and logged

print("All done — no crash")'''
    ),
    (
        "Write a Python function to flatten a nested dictionary into dot-notation keys.",
        '''def flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> dict:
    result = {}
    for key, value in d.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, full_key, sep))
        else:
            result[full_key] = value
    return result

nested = {
    "user": {
        "name": "Brian",
        "address": {
            "city": "Washington",
            "zip": "20001"
        }
    },
    "settings": {"theme": "dark"}
}

flat = flatten_dict(nested)
# {"user.name": "Brian", "user.address.city": "Washington",
#  "user.address.zip": "20001", "settings.theme": "dark"}
for k, v in flat.items():
    print(f"{k}: {v}")'''
    ),
    (
        "Write a Python script that generates a PDF report from a dict of data.",
        '''# pip install reportlab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import date

def generate_report(data: dict, output_path: str = "report.pdf"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, data.get("title", "Report"))
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Generated: {date.today()}")
    c.line(50, height - 90, width - 50, height - 90)

    # Body
    y = height - 120
    c.setFont("Helvetica", 12)
    for key, value in data.get("fields", {}).items():
        c.drawString(50, y, f"{key}:")
        c.drawString(200, y, str(value))
        y -= 20
        if y < 80:
            c.showPage()
            y = height - 60

    c.save()
    print(f"Report saved: {output_path}")

generate_report({
    "title": "Monthly AI Training Summary",
    "fields": {
        "Model": "Anthos v2",
        "Training Steps": "10,000",
        "Final Loss": "1.42",
        "Dataset Size": "1,196,366 examples",
        "Creator": "Brian Tushae Thomas"
    }
})'''
    ),
    (
        "Write a Python function to clean and normalize text for NLP processing.",
        '''import re
import unicodedata

def normalize_text(text: str, lowercase: bool = True, remove_urls: bool = True,
                   remove_extra_spaces: bool = True) -> str:
    """Clean text for NLP: normalize unicode, remove noise, standardize whitespace."""
    # Normalize unicode (convert accented chars to ASCII where possible)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    if remove_urls:
        text = re.sub(r"https?://\\S+|www\\.\\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove special characters (keep letters, numbers, punctuation)
    text = re.sub(r"[^\\w\\s\\.,!?;:\\-\\'\\\"()]", " ", text)

    if lowercase:
        text = text.lower()

    if remove_extra_spaces:
        text = re.sub(r"\\s+", " ", text).strip()

    return text

raw = "Check out https://example.com! It\'s <b>AMAZING</b>... really    great stuff  ."
clean = normalize_text(raw)
print(clean)  # "check out it's amazing... really great stuff ."'''
    ),
    (
        "Write a Python script to split a large JSONL file into train/val/test sets.",
        '''import json, random
from pathlib import Path

def split_jsonl(input_path: str, train: float = 0.8, val: float = 0.1, seed: int = 42):
    """Split JSONL file into train/val/test. test = 1 - train - val."""
    assert train + val < 1.0, "train + val must be < 1"
    random.seed(seed)

    # Load all lines
    with open(input_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    random.shuffle(lines)

    n        = len(lines)
    n_train  = int(n * train)
    n_val    = int(n * val)

    splits = {
        "train": lines[:n_train],
        "val":   lines[n_train:n_train + n_val],
        "test":  lines[n_train + n_val:],
    }

    base = Path(input_path).stem
    for split_name, split_lines in splits.items():
        out = Path(f"data/{base}_{split_name}.jsonl")
        out.write_text("\\n".join(split_lines))
        print(f"  {split_name}: {len(split_lines):,} examples → {out}")

split_jsonl("data/sft_master.jsonl", train=0.9, val=0.05)'''
    ),
    (
        "Write a Python class to build and query a simple inverted index for keyword search.",
        '''import re
from collections import defaultdict

class InvertedIndex:
    """Simple keyword search index — like a book\'s back-of-book index."""

    def __init__(self):
        self.index = defaultdict(set)
        self.docs  = {}

    def add(self, doc_id: str, text: str):
        self.docs[doc_id] = text
        for word in self._tokenize(text):
            self.index[word].add(doc_id)

    def search(self, query: str, require_all: bool = True) -> list[str]:
        words = self._tokenize(query)
        if not words:
            return []
        sets = [self.index.get(w, set()) for w in words]
        if require_all:
            result = set.intersection(*sets) if sets else set()
        else:
            result = set.union(*sets) if sets else set()
        return sorted(result)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

idx = InvertedIndex()
idx.add("doc1", "Python machine learning tutorial")
idx.add("doc2", "Python web scraping with requests")
idx.add("doc3", "Machine learning with PyTorch and Python")

print(idx.search("python machine learning"))   # ["doc1", "doc3"]
print(idx.search("scraping"))                  # ["doc2"]'''
    ),
    (
        "Write a Python function that polls an API endpoint until a job completes.",
        '''import time
import urllib.request
import json

def poll_until_done(url: str, check_fn, max_wait: int = 300,
                    interval: int = 5) -> dict:
    """Poll a URL until check_fn(response) returns True or timeout."""
    start = time.monotonic()
    attempt = 0

    while time.monotonic() - start < max_wait:
        attempt += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python/poller"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            print(f"  Poll #{attempt}: status={data.get(\'status\', \'?\')} elapsed={time.monotonic()-start:.1f}s")
            if check_fn(data):
                return data
        except Exception as e:
            print(f"  Poll #{attempt} error: {e}")
        time.sleep(interval)

    raise TimeoutError(f"Job did not complete within {max_wait}s")

# Usage: poll a training job API
result = poll_until_done(
    url="https://api.example.com/jobs/abc123",
    check_fn=lambda r: r.get("status") in ("completed", "failed"),
    max_wait=300,
    interval=10
)
print(f"Job finished: {result[\'status\']}")'''
    ),
]

ALGORITHM_TASKS = [
    (
        "Implement binary search in Python and explain when to use it.",
        '''def binary_search(arr: list, target) -> int:
    """Search sorted list for target. Returns index or -1 if not found. O(log n)."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Example
nums = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
print(binary_search(nums, 7))    # 3
print(binary_search(nums, 6))    # -1

# When to use:
# - List is SORTED (required)
# - You\'re searching repeatedly (O(log n) vs O(n) for linear search)
# - 1M items: binary search = 20 comparisons, linear search = 1,000,000
# - Built-in: import bisect; bisect.bisect_left(arr, target)'''
    ),
    (
        "Implement a simple LRU cache in Python from scratch.",
        '''from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache — O(1) get and put."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache    = OrderedDict()  # maintains insertion order

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # mark as recently used
        return self.cache[key]

    def put(self, key: int, value: int):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # evict least recently used

# Usage
cache = LRUCache(3)
cache.put(1, "one")
cache.put(2, "two")
cache.put(3, "three")
print(cache.get(1))    # "one" (now most recent)
cache.put(4, "four")   # evicts key 2 (least recently used)
print(cache.get(2))    # -1 (evicted)

# Real use: caching API responses, database query results, computed embeddings'''
    ),
    (
        "Write a Python function that finds the most frequent words in a text document.",
        '''from collections import Counter
import re

def top_words(text: str, n: int = 20, min_length: int = 4) -> list[tuple[str, int]]:
    """Return top n most frequent words, ignoring short words and stopwords."""
    STOPWORDS = {"this", "that", "with", "from", "have", "been", "they",
                 "their", "there", "would", "could", "should", "which", "about"}
    words = re.findall(r\'[a-z]+\', text.lower())
    filtered = [w for w in words if len(w) >= min_length and w not in STOPWORDS]
    return Counter(filtered).most_common(n)

# Usage
text = open("my_document.txt").read()
results = top_words(text, n=10)
for word, count in results:
    print(f"{count:5d}  {word}")

# Output example:
#   847  learning
#   612  model
#   589  training
#   421  python
#   388  function'''
    ),
    (
        "Implement a simple rate limiter in Python for API calls.",
        '''import time
from collections import deque

class RateLimiter:
    """Sliding window rate limiter. Allows max_calls per period seconds."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period    = period
        self.calls     = deque()

    def is_allowed(self) -> bool:
        now = time.monotonic()
        # Remove calls outside the window
        while self.calls and self.calls[0] <= now - self.period:
            self.calls.popleft()
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False

    def wait_and_call(self):
        """Block until a call is allowed."""
        while not self.is_allowed():
            time.sleep(0.05)

# Usage: 10 calls per second
limiter = RateLimiter(max_calls=10, period=1.0)

for i in range(20):
    limiter.wait_and_call()
    print(f"API call {i+1} at {time.monotonic():.2f}")'''
    ),
]

DESIGN_TASKS = [
    (
        "Design the data flow for a system where users upload PDFs and can ask questions about them.",
        '''Here\'s the full data flow:

```
User uploads PDF
      ↓
[Upload Service]
  - Accept multipart/form-data
  - Validate (size < 50MB, type = PDF)
  - Save to object storage (S3/local)
  - Emit job event: {doc_id, path}
      ↓
[Ingestion Worker] (async)
  - Extract text with pdfplumber
  - Split into chunks (~400 tokens each, 50-token overlap)
  - Embed each chunk with sentence-transformer
  - Store embeddings + chunk text in vector DB (Chroma/FAISS/pgvector)
  - Mark doc_id as ready in metadata DB
      ↓
User sends question: "What does it say about X?"
      ↓
[Query Service]
  - Embed the question
  - Retrieve top-k matching chunks from vector DB (k=5)
  - Build prompt: system + context chunks + user question
  - Call LLM (local Ollama or API)
  - Stream response back to user
      ↓
[Response]
  - Stream tokens to UI
  - Include source citations (chunk source + page number)
```

**Key design decisions:**
- Chunking strategy matters most — 400 tokens with 50-token overlap works well for general docs
- Use async ingestion so the upload returns immediately
- Store doc_id + chunk_index in vector DB metadata for citation tracking
- Rate limit query service per user to prevent runaway LLM costs'''
    ),
    (
        "Design a training data pipeline for a custom language model from scratch.",
        '''Here\'s the complete pipeline:

```
[Source Data] → [Filter] → [Tokenize] → [Pack] → [Train]
```

**Step-by-step:**

**1. Data collection**
- Identify sources: books, code, conversations, domain docs
- Download raw files (.jsonl, .txt, .parquet)
- Target: 10M+ tokens minimum; 1B+ for real capability

**2. Deduplication (critical)**
```python
# Near-dedup using MinHash or exact hash
seen = set()
for doc in corpus:
    h = hashlib.md5(doc[:500].encode()).hexdigest()
    if h not in seen:
        seen.add(h)
        yield doc
```

**3. Quality filtering**
- Remove docs < 100 words
- Remove docs with >30% non-ASCII
- Keep perplexity-filtered content (high ppl = garbled; too low = boilerplate)

**4. Formatting for SFT**
- Convert to conversations format: [{from: system}, {from: human}, {from: gpt}]
- Label: prompt tokens = -100 (masked), response tokens = actual labels

**5. Tokenization + packing**
- Tokenize all conversations
- Pack multiple conversations into fixed-length sequences (e.g., 2048 tokens)
- This maximizes GPU utilization (no wasted padding tokens)

**6. Training**
- Use AdamW optimizer, cosine LR schedule, warmup 2k steps
- Clip gradients at 1.0
- Save checkpoint every 500 steps
- Evaluate perplexity on held-out set

**Data ratios that work:**
- General instruction: 60%
- Code: 20%
- Domain-specific: 15%
- Safety/identity: 5%'''
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Build training pairs
# ─────────────────────────────────────────────────────────────────────────────

def make_conv(question: str, answer: str, system: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human",  "value": question.strip()},
            {"from": "gpt",    "value": answer.strip()},
        ]
    }


TASK_TEMPLATES = [
    # Pattern: "Write a Python function to X"
    ("Write a Python function to count the number of lines, words, and characters in a text file.",
     '''def count_stats(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    lines = text.count("\\n") + (1 if text and not text.endswith("\\n") else 0)
    words = len(text.split())
    chars = len(text)
    return {"lines": lines, "words": words, "chars": chars}

stats = count_stats("my_document.txt")
print(f"Lines: {stats[\'lines\']}, Words: {stats[\'words\']}, Chars: {stats[\'chars\']}")'''),

    ("Write a Python function to find duplicate files in a folder by comparing their content hash.",
     '''import hashlib
from pathlib import Path
from collections import defaultdict

def find_duplicates(folder: str) -> dict[str, list[str]]:
    hash_to_files = defaultdict(list)
    for path in Path(folder).rglob("*"):
        if path.is_file():
            h = hashlib.md5(path.read_bytes()).hexdigest()
            hash_to_files[h].append(str(path))
    return {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}

dupes = find_duplicates("./data")
for h, files in dupes.items():
    print(f"Duplicate ({h[:8]}...):")
    for f in files:
        print(f"  {f}")'''),

    ("Write a Python script that takes a list of URLs and checks which ones return 200 OK.",
     '''import urllib.request
from concurrent.futures import ThreadPoolExecutor

def check_url(url: str) -> tuple[str, int, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return url, resp.status, "OK"
    except Exception as e:
        return url, 0, str(e)[:60]

urls = [
    "https://example.com",
    "https://httpbin.org/status/404",
    "https://github.com",
]

with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(check_url, urls))

for url, status, msg in results:
    icon = "✓" if status == 200 else "✗"
    print(f"{icon} [{status}] {url}")'''),

    ("Write a Python function to generate a random secure password.",
     '''import secrets
import string

def generate_password(length: int = 16, use_symbols: bool = True) -> str:
    chars = string.ascii_letters + string.digits
    if use_symbols:
        chars += "!@#$%^&*()_+-=[]{};"

    # Guarantee at least one of each required type
    password = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
    ]
    if use_symbols:
        password.append(secrets.choice("!@#$%^&*"))

    # Fill the rest randomly
    password += [secrets.choice(chars) for _ in range(length - len(password))]
    secrets.SystemRandom().shuffle(password)
    return "".join(password)

for _ in range(5):
    print(generate_password(length=20))'''),

    ("Write a Python function to convert a list of dicts to a formatted Markdown table.",
     '''def to_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    widths  = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}

    def fmt_row(values):
        return "| " + " | ".join(str(v).ljust(widths[h]) for h, v in zip(headers, values)) + " |"

    header_row = fmt_row(headers)
    sep_row    = "| " + " | ".join("-" * widths[h] for h in headers) + " |"
    data_rows  = [fmt_row([row.get(h, "") for h in headers]) for row in rows]

    return "\\n".join([header_row, sep_row] + data_rows)

data = [
    {"Model": "Anthos", "Params": "73.6M", "Loss": "1.42"},
    {"Model": "Smoke",  "Params": "6.9M",  "Loss": "2.10"},
]
print(to_markdown_table(data))'''),

    ("Write a Python class that implements a simple event emitter / observer pattern.",
     '''from collections import defaultdict
from typing import Callable

class EventEmitter:
    def __init__(self):
        self._listeners = defaultdict(list)

    def on(self, event: str, callback: Callable):
        self._listeners[event].append(callback)
        return self  # allow chaining

    def emit(self, event: str, *args, **kwargs):
        for callback in self._listeners.get(event, []):
            callback(*args, **kwargs)

    def off(self, event: str, callback: Callable = None):
        if callback:
            self._listeners[event] = [c for c in self._listeners[event] if c != callback]
        else:
            self._listeners.pop(event, None)

# Usage
emitter = EventEmitter()
emitter.on("data", lambda d: print(f"Got data: {d}"))
emitter.on("data", lambda d: print(f"Also got: {d}"))
emitter.on("done", lambda: print("All done!"))

emitter.emit("data", {"key": "value"})
emitter.emit("done")'''),

    ("Write a Python function to parse a log file and extract error messages with timestamps.",
     '''import re
from datetime import datetime

def parse_errors(log_path: str, pattern: str = r"ERROR|CRITICAL|FATAL") -> list[dict]:
    errors = []
    # Common log format: 2026-09-05 12:34:56 ERROR message here
    log_regex = re.compile(r"(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}).*?(" + pattern + r")(.+)")

    with open(log_path) as f:
        for lineno, line in enumerate(f, 1):
            match = log_regex.search(line)
            if match:
                errors.append({
                    "line":    lineno,
                    "time":    match.group(1),
                    "level":   match.group(2),
                    "message": match.group(3).strip()
                })
    return errors

errors = parse_errors("app.log")
for e in errors[:10]:
    print(f"[{e[\'time\']}] {e[\'level\']}: {e[\'message\']}")
print(f"\\nTotal errors: {len(errors)}")'''),

    ("Write a Python function that implements memoization (caching) with a time-to-live expiry.",
     '''import time
import functools
from typing import Any

def ttl_cache(seconds: int = 60):
    """Cache function results for `seconds` seconds, then recompute."""
    def decorator(func):
        cache = {}   # key → (value, expires_at)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            if key in cache:
                value, expires_at = cache[key]
                if now < expires_at:
                    return value        # cache hit
            result = func(*args, **kwargs)
            cache[key] = (result, now + seconds)
            return result

        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator

@ttl_cache(seconds=5)
def get_data(query: str) -> str:
    print(f"  (fetching: {query})")    # only prints on cache miss
    return f"result for {query}"

print(get_data("hello"))   # fetches
print(get_data("hello"))   # cache hit
time.sleep(6)
print(get_data("hello"))   # expired → fetches again'''),

    ("Write a Python script that reads environment variables and validates required ones are set.",
     '''import os
import sys

REQUIRED = {
    "DATABASE_URL": "PostgreSQL connection string",
    "API_KEY":      "External API authentication key",
    "SECRET_KEY":   "Application secret for sessions",
}

OPTIONAL = {
    "DEBUG":    ("false", "Enable debug mode"),
    "LOG_LEVEL": ("INFO", "Logging verbosity"),
    "PORT":     ("8000", "Server port"),
}

def load_config() -> dict:
    missing = []
    config  = {}

    for key, description in REQUIRED.items():
        val = os.environ.get(key)
        if not val:
            missing.append(f"  {key}: {description}")
        else:
            config[key] = val

    if missing:
        print("ERROR: Missing required environment variables:")
        print("\\n".join(missing))
        sys.exit(1)

    for key, (default, _) in OPTIONAL.items():
        config[key] = os.environ.get(key, default)

    return config

config = load_config()
print(f"Config loaded: {list(config.keys())}")'''),

    ("Write a Python function to recursively convert all keys in a dict from camelCase to snake_case.",
     '''import re

def camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\\1_\\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s1).lower()

def convert_keys(obj):
    if isinstance(obj, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_keys(item) for item in obj]
    return obj

data = {
    "userId": 123,
    "firstName": "Brian",
    "contactInfo": {
        "emailAddress": "brian@example.com",
        "phoneNumber": "555-1234",
    },
    "trainingConfig": {
        "learningRate": 0.001,
        "maxSteps": 10000,
    }
}

result = convert_keys(data)
import json; print(json.dumps(result, indent=2))
# {"user_id": 123, "first_name": "Brian", "contact_info": {...}, ...}'''),

    ("Write a Python function to stream a large file line by line without loading it all into memory.",
     '''from pathlib import Path

def stream_jsonl(path: str, batch_size: int = 1000):
    """Yield batches of parsed JSONL lines without loading the whole file."""
    import json
    batch = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                batch.append(json.loads(line))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            except json.JSONDecodeError:
                continue
    if batch:
        yield batch

# Usage: process 1M JSONL file in 1000-row batches
total = 0
for batch in stream_jsonl("data/sft_master.jsonl", batch_size=1000):
    # Process each batch
    total += len(batch)
    if total % 100_000 == 0:
        print(f"Processed {total:,} rows...")

print(f"Done: {total:,} total rows")'''),

    ("Write a Python function to validate an email address using regex.",
     '''import re

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$"
)

def validate_email(email: str) -> bool:
    return bool(EMAIL_REGEX.match(email.strip()))

def validate_many(emails: list[str]) -> dict[str, bool]:
    return {email: validate_email(email) for email in emails}

tests = [
    "brian@example.com",        # valid
    "brian.thomas@ai.io",       # valid
    "not-an-email",             # invalid
    "@missing-local.com",       # invalid
    "missing@.com",             # invalid
    "brian+tag@company.co.uk",  # valid
]

for email, valid in validate_many(tests).items():
    icon = "✓" if valid else "✗"
    print(f"{icon}  {email}")'''),
]


def generate(n: int, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)

    base_pairs = []

    for q, a in PYTHON_TASKS:
        base_pairs.append(make_conv(q, a, ANTHOS_CODE_SYSTEM))
    for q, a in ML_TASKS:
        base_pairs.append(make_conv(q, a, ANTHOS_ML_SYSTEM))
    for q, a in APPLIED_PROBLEMS:
        base_pairs.append(make_conv(q, a, ANTHOS_SYSTEM))
    for q, a in DESIGN_TASKS:
        base_pairs.append(make_conv(q, a, ANTHOS_SYSTEM))
    for q, a in BUSINESS_TASKS:
        base_pairs.append(make_conv(q, a, ANTHOS_CODE_SYSTEM))
    for q, a in MORE_PYTHON_TASKS:
        base_pairs.append(make_conv(q, a, ANTHOS_CODE_SYSTEM))
    for q, a in ALGORITHM_TASKS:
        base_pairs.append(make_conv(q, a, ANTHOS_CODE_SYSTEM))
    for q, a in TASK_TEMPLATES:
        base_pairs.append(make_conv(q, a, ANTHOS_CODE_SYSTEM))

    # If n > base_pairs, sample with replacement to hit target
    if n <= len(base_pairs):
        pairs = random.sample(base_pairs, n)
    else:
        pairs = base_pairs + random.choices(base_pairs, k=n - len(base_pairs))

    random.shuffle(pairs)

    with open(output, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Generated {len(pairs)} applied training pairs → {output}")
    print(f"  Python tasks:    {len(PYTHON_TASKS)}")
    print(f"  ML tasks:        {len(ML_TASKS)}")
    print(f"  Applied problems:{len(APPLIED_PROBLEMS)}")
    print(f"  Design tasks:    {len(DESIGN_TASKS)}")
    print(f"  Business tasks:  {len(BUSINESS_TASKS)}")
    print(f"  More Python:     {len(MORE_PYTHON_TASKS)}")
    print(f"  Algorithm tasks: {len(ALGORITHM_TASKS)}")
    print(f"  Templates:       {len(TASK_TEMPLATES)}")


def main():
    parser = argparse.ArgumentParser()
    all_tasks = PYTHON_TASKS + ML_TASKS + APPLIED_PROBLEMS + DESIGN_TASKS + BUSINESS_TASKS + MORE_PYTHON_TASKS + ALGORITHM_TASKS + TASK_TEMPLATES
    parser.add_argument("--n",      type=int, default=500)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    generate(args.n, Path(args.output))
    print(f"\nTo merge into master:")
    print(f"  cat Desktop/anthos-repo/data/sft_master.jsonl {args.output} > /tmp/merged.jsonl && mv /tmp/merged.jsonl Desktop/anthos-repo/data/sft_master.jsonl")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Variation generator — takes base pairs and creates context variations
# Gets us from ~37 unique → 500+ without repetitive manual writing
# ─────────────────────────────────────────────────────────────────────────────
