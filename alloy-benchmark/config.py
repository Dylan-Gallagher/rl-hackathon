"""Central configuration for the alloy-agents CTF benchmark.

All credentials are read from the environment. Set these before running:
    DAYTONA_API_KEY     Daytona sandbox API key
    ANTHROPIC_API_KEY   Anthropic API key (Model A)
    GLM_API_KEY         z.ai / GLM key (Model B) — a Coding Plan key
    QWEN_BASE_URL       optional; OpenAI-compatible URL of a self-hosted Qwen
"""
import os

# --- API credentials (env only) -------------------------------------------
DAYTONA_API_KEY = os.environ.get("DAYTONA_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")

# --- Model identities ------------------------------------------------------
# Two comparably-capable models from different lineages (good for the overlap
# analysis). Reasoning/thinking is DISABLED on both so no cross-model thinking
# signature can ever be carried across a turn boundary.
# NOTE: a GLM Coding Plan key must use the /coding/ endpoint (quota-based); the
# pay-as-you-go /paas/v4 endpoint reports "insufficient balance" for such keys.
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")

MODEL_A = {"key": "A", "provider": "anthropic", "model": "claude-sonnet-5", "label": "claude-sonnet-5"}
MODEL_B = {"key": "B", "provider": "glm", "model": "glm-4.7", "label": "glm-4.7"}

# Self-hosted Qwen3-8B baseline (served with vLLM, reached over an SSH tunnel).
QWEN_MODEL = "Qwen/Qwen3-8B"
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")

# Public pricing (USD per 1M tokens) for cost projection.
PRICING = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "glm-4.7": {"input": 0.60, "output": 2.20},
    "Qwen/Qwen3-8B": {"input": 0.0, "output": 0.0},  # self-hosted; GPU billed separately
}

# --- Experiment parameters -------------------------------------------------
TURN_CAP = 40
ATTEMPTS_PER_CHALLENGE = 4          # in the requested 3-5 range
SAMPLE_SIZE = 30
MASTER_SEED = 20260829

# Full-pool category proportions (from ctf_archive.json, 658 challenges) —
# the sample is stratified to match these even though we draw from the
# sha256-verifiable subset.
POOL_PROPORTIONS = {
    "crypto": 0.348,
    "pwn": 0.246,
    "rev": 0.187,
    "misc": 0.129,
    "forensics": 0.058,
    "web": 0.032,
}

# --- Sandbox ---------------------------------------------------------------
# The Daytona account caps TOTAL concurrent sandbox resources, so per-sandbox
# size bounds achievable concurrency (workers*mem <= mem cap, workers*disk <= disk cap).
SOLVER_SNAPSHOT = os.environ.get("SOLVER_SNAPSHOT", "alloy-ctf-solver-v3")
SANDBOX_CPU = 1
SANDBOX_MEM = 2      # GiB
SANDBOX_DISK = 5     # GiB
ACCOUNT_MEM_CAP_GIB = 10
ACCOUNT_DISK_CAP_GIB = 30
COMMAND_TIMEOUT = 120   # seconds per shell command

# --- Paths -----------------------------------------------------------------
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CHALLENGES_DIR = ROOT / "challenges"
TRAJ_DIR = ROOT / "trajectories"
ARCHIVE_DIR = ROOT / "ctf-archive"
CTF_ARCHIVE_JSON = ROOT / "CTF-Dojo" / "ctf_archive.json"
