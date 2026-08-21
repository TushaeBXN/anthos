# Anthos Code Training Dataset Package

This package is a reproducible builder for three public code-instruction datasets:

1. `ise-uiuc/Magicoder-OSS-Instruct-75K`
2. `m-a-p/CodeFeedback-Filtered-Instruction`
3. `ise-uiuc/Magicoder-Evol-Instruct-110K`

It downloads the exact upstream JSONL files, verifies SHA-256 hashes, normalizes them to a common `messages` schema, and creates an exact-text-deduplicated Anthos SFT corpus.

## Important
The large upstream JSONL files are **not bundled in this archive** because the execution environment used to build this package cannot retrieve Hugging Face's 200–400 MB Xet objects directly. Run the builder on a machine with internet access.

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
python download_and_build.py
```

Outputs:

- `raw/` — verified upstream files
- `normalized/` — common Anthos conversation format
- `combined/anthos_code_sft_deduped.jsonl` — deduplicated training corpus
- `manifests/dataset_manifest.json` — source URLs, hashes, licenses, and counts

The source repositories report approximately 75,197 OSS-Instruct rows, 156,526 CodeFeedback rows, and 106,878 Evol-Instruct rows.
