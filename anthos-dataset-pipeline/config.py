"""
config.py — Central config for Anthos Dataset Pipeline
All secrets via environment variables. Never hardcode.
"""

import os
from pathlib import Path

# ── API Keys ──────────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── GitHub ────────────────────────────────────────────────────────────────────
GITHUB_USERNAMES = ["TushaeBXN", "TushaeThomas"]
GITHUB_SKIP_FORKS = True
GITHUB_CODE_EXTENSIONS = [".py", ".js", ".jsx", ".ts", ".tsx", ".sh", ".ipynb", ".yaml", ".yml", ".md"]
GITHUB_SKIP_DIRS = ["node_modules", ".git", "__pycache__", "dist", "build", ".venv"]
GITHUB_MAX_FILE_BYTES = 100_000  # skip files over 100KB

# ── Claude.ai Export ──────────────────────────────────────────────────────────
# Path to the conversations.json from Settings → Account → Export Data
CLAUDE_EXPORT_PATH = os.environ.get(
    "CLAUDE_EXPORT_PATH",
    str(Path.home() / "Downloads" / "claude_export" / "conversations.json")
)

# ── Claude Code Logs ──────────────────────────────────────────────────────────
CLAUDE_CODE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
RAW_DIR = OUTPUT_DIR / "raw"
MERGED_DIR = OUTPUT_DIR / "merged"

GITHUB_RAW_OUT = RAW_DIR / "github_raw.jsonl"
CLAUDE_AI_RAW_OUT = RAW_DIR / "claude_ai_raw.jsonl"
CLAUDE_CODE_RAW_OUT = RAW_DIR / "claude_code_raw.jsonl"
MERGED_OUT = MERGED_DIR / "anthos-code-training-v1.jsonl"
THOUGHT_OUT = MERGED_DIR / "anthos-code-training-v1-thought.jsonl"

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_REPO_ID = "machomenc/anthos-code-training-v1"
HF_TRAIN_SPLIT = 0.95

# ── Quality Filter ────────────────────────────────────────────────────────────
MIN_CODE_LINES = 10          # drop assistant turns with fewer lines of code
REQUIRE_CODE_BLOCK = True    # drop examples with no ``` block at all

# ── Thought-Token Augmentation ────────────────────────────────────────────────
THOUGHT_MODEL = "claude-sonnet-4-6"
THOUGHT_MAX_TOKENS = 512
THOUGHT_SYSTEM_PROMPT = """\
You are helping build training data for a custom AI architecture called Anthos
(Thought-Token Bifurcated Recurrent Transformer). For each coding response,
prepend a <|thought|> block that shows the internal reasoning — what design
tradeoffs were considered, why this approach was chosen, what was rejected.
Keep it concise (3-6 sentences). Output ONLY the <|thought|>...</|thought|>
block followed by the original response. Do not change the code itself.\
"""

def ensure_output_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
