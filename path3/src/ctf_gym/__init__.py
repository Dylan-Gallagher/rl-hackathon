"""ctf_gym: unified CTF task/env abstraction for RL hackathon Path 3.

Core runtime is stdlib-only. Optional extras:
  - daytona: `pip install daytona-sdk==0.207.*` for the Daytona sandbox backend
  - eval:    `pip install httpx` for the OpenAI-compatible policy
  - sft:     `pip install "trl>=0.9" "peft" "transformers" "datasets"` for EI training
"""

__version__ = "0.1.0"

from ctf_gym.contracts import (
    EnvSpec,
    FlagInjection,
    FlagSpec,
    Horizon,
    Obs,
    Task,
    Transcript,
    TranscriptMessage,
    ValidationError,
    load_task,
)

__all__ = [
    "EnvSpec",
    "FlagInjection",
    "FlagSpec",
    "Horizon",
    "Obs",
    "Task",
    "Transcript",
    "TranscriptMessage",
    "ValidationError",
    "load_task",
    "__version__",
]
