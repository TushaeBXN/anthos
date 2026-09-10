#!/usr/bin/env python3
"""
anthos_acquire.py — Anthos Intelligence Dataset Acquisition
"""

import json
from datetime import datetime
from pathlib import Path
from huggingface_hub import snapshot_download

BASE = Path("/Volumes/1TB Drive/anthos-data")

DIRS = ["datasets", "pdfs", "processed", "checkpoints", "models", "pipelines"]

DATASETS = [
    {"id": "openai/gsm8k",                  "name": "gsm8k",            "priority": 1},
    {"id": "lighteval/MATH",                "name": "math-lighteval",   "priority": 1},
    {"id": "HuggingFaceFW/fineweb-edu",     "name": "fineweb-edu",      "priority": 1},
    {"id": "teknium/OpenHermes-2.5",        "name": "openhermes-2.5",   "priority": 2},
    {"id": "cais/mmlu",                     "name": "mmlu",             "priority": 2},
    {"id": "bigcode/the-stack-dedup",       "name": "the-stack-dedup",  "priority": 3},
]

MANIFEST_PATH = BASE / "manifest.json"

def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}

def save_manifest(manifest):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

def setup_dirs():
    for d in DIRS:
        (BASE / d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directory structure ready at {BASE}")

def already_downloaded(manifest, name):
    return manifest.get(name, {}).get("status") == "complete" and (BASE / "datasets" / name).exists()

def download_dataset(entry, manifest):
    name = entry["name"]
    hf_id = entry["id"]
    dest = BASE / "datasets" / name

    if already_downloaded(manifest, name):
        print(f"  ✓ {name} — already downloaded, skipping")
        return

    print(f"\n  ↓ {name} ({hf_id})\n    → {dest}")

    try:
        snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        manifest[name] = {
            "hf_id": hf_id,
            "status": "complete",
            "downloaded_at": datetime.now().isoformat(),
            "path": str(dest),
        }
        save_manifest(manifest)
        print(f"    ✓ complete")

    except KeyboardInterrupt:
        print(f"\n  ⚠ interrupted — run again to resume {name}")
        manifest[name] = {"hf_id": hf_id, "status": "interrupted"}
        save_manifest(manifest)
        raise

    except Exception as e:
        print(f"    ✗ failed: {e}")
        manifest[name] = {"hf_id": hf_id, "status": "failed", "error": str(e)}
        save_manifest(manifest)

def main():
    if not Path("/Volumes/1TB Drive").exists():
        print("✗ Drive not found — plug it in and try again.")
        return

    setup_dirs()
    manifest = load_manifest()
    queue = sorted(DATASETS, key=lambda x: x["priority"])

    print(f"\nAnthos Intelligence — Dataset Acquisition")
    print(f"Target: {BASE}")
    print(f"Queue:  {len(queue)} datasets")
    print("Ctrl+C any time — resumes on next run.\n")

    for entry in queue:
        download_dataset(entry, manifest)

    print("\n── Done ──────────────────────────────────────────")
    for name, entry in manifest.items():
        icon = "✓" if entry.get("status") == "complete" else "✗"
        print(f"  {icon} {name}: {entry.get('status')}")
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"PDFs:     {BASE}/pdfs/")

if __name__ == "__main__":
    main()
