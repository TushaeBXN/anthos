"""
ingest/github_ingest.py
Pulls all non-fork repos for GITHUB_USERNAMES, extracts code files,
and writes ShareGPT-format JSONL to output/raw/github_raw.jsonl.

Each example = one code file turned into an instruction/response pair.
Instruction: "Here is [filename] from [repo]. Explain and show the code."
Response: the file content in a fenced code block.
"""

import json
import time
import base64
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GITHUB_TOKEN, GITHUB_USERNAMES, GITHUB_SKIP_FORKS,
    GITHUB_CODE_EXTENSIONS, GITHUB_SKIP_DIRS, GITHUB_MAX_FILE_BYTES,
    GITHUB_RAW_OUT, ensure_output_dirs,
)

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


def gh_get(url, params=None):
    """GET with basic rate-limit retry."""
    while True:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 5)
            print(f"  Rate limited — waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()


def list_repos(username):
    repos = []
    page = 1
    while True:
        batch = gh_get(
            f"https://api.github.com/users/{username}/repos",
            params={"per_page": 100, "page": page, "type": "owner"},
        )
        if not batch:
            break
        repos.extend(batch)
        page += 1
    return repos


def list_tree(owner, repo, branch):
    data = gh_get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    return data.get("tree", [])


def fetch_file(owner, repo, path):
    data = gh_get(f"https://api.github.com/repos/{owner}/{repo}/contents/{path}")
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return None


def should_skip_path(path_str):
    parts = Path(path_str).parts
    return any(skip in parts for skip in GITHUB_SKIP_DIRS)


def ext_ok(path_str):
    return Path(path_str).suffix in GITHUB_CODE_EXTENSIONS


def file_to_example(owner, repo, path, content):
    ext = Path(path).suffix.lstrip(".")
    lang = ext if ext else "text"
    instruction = (
        f"This is `{path}` from the `{owner}/{repo}` repository. "
        f"Show the full implementation."
    )
    response = f"```{lang}\n{content.strip()}\n```"
    return {
        "conversations": [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ],
        "source": "github",
        "repo": f"{owner}/{repo}",
        "file": path,
    }


def run():
    ensure_output_dirs()
    count = 0

    with GITHUB_RAW_OUT.open("w") as out:
        for username in GITHUB_USERNAMES:
            print(f"\n── {username} ──")
            try:
                repos = list_repos(username)
            except Exception as e:
                print(f"  Could not list repos: {e}")
                continue

            for repo in repos:
                if GITHUB_SKIP_FORKS and repo.get("fork"):
                    continue
                rname = repo["name"]
                branch = repo.get("default_branch", "main")
                print(f"  {rname} ({branch})")

                try:
                    tree = list_tree(username, rname, branch)
                except Exception as e:
                    print(f"    skip — {e}")
                    continue

                for item in tree:
                    if item["type"] != "blob":
                        continue
                    fpath = item["path"]
                    fsize = item.get("size", 0)
                    if not ext_ok(fpath):
                        continue
                    if should_skip_path(fpath):
                        continue
                    if fsize > GITHUB_MAX_FILE_BYTES:
                        continue

                    try:
                        content = fetch_file(username, rname, fpath)
                    except Exception:
                        continue

                    if not content or not content.strip():
                        continue

                    example = file_to_example(username, rname, fpath, content)
                    out.write(json.dumps(example) + "\n")
                    count += 1

    print(f"\n✓ GitHub ingest complete — {count} examples → {GITHUB_RAW_OUT}")


if __name__ == "__main__":
    run()
