"""Typed, validated contracts for tasks, observations and transcripts (README section 3).

Security model:
  - Public task JSON never contains plaintext static flags. Static flags are
    represented by their SHA-256 digest; the exact flag lives in an external
    *verifier record* (sidecar JSON, chmod 0600) that stays on the host.
  - Verifier execution happens outside sandboxes (see ctf_gym.verifier).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

CATEGORIES = ("pwn", "rev", "crypto", "web", "forensics", "misc")
SOURCES = ("nyuctf", "random-crypto", "custom")
SPLITS = ("train", "eval")
LAUNCH_MODES = ("supervisor", "compose", "repl", "none")
FLAG_MODES = ("generated", "static")
VERIFY_MODES = ("regex", "exact", "script")

TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class ValidationError(ValueError):
    """Raised when a contract object fails validation."""


@dataclass
class Horizon:
    max_steps: int = 40
    timeout_s: int = 1800

    def validate(self) -> None:
        if not isinstance(self.max_steps, int) or not (1 <= self.max_steps <= 1000):
            raise ValidationError(f"horizon.max_steps must be int in [1,1000], got {self.max_steps!r}")
        if not isinstance(self.timeout_s, int) or not (1 <= self.timeout_s <= 86400):
            raise ValidationError(f"horizon.timeout_s must be int in [1,86400], got {self.timeout_s!r}")


@dataclass
class EnvSpec:
    image: str
    launch: str = "none"  # supervisor | compose | repl | none

    def validate(self) -> None:
        if not self.image or not isinstance(self.image, str):
            raise ValidationError("env.image must be a non-empty string")
        if self.launch not in LAUNCH_MODES:
            raise ValidationError(f"env.launch must be one of {LAUNCH_MODES}, got {self.launch!r}")


@dataclass
class FlagInjection:
    """How the per-episode flag is placed inside the sandbox.

    mode "file":  task assets containing ``placeholder`` are uploaded with the
                  placeholder replaced by the real flag (safe default).
    mode "env":   flag passed as a container environment variable. This is
                  explicitly UNSAFE for strict anti-cheat evaluation (an agent
                  can simply run `env`) and must be opted into.
    """

    mode: str = "file"  # file | env
    placeholder: str = "FLAG_PLACEHOLDER"
    path: Optional[str] = None  # relative path inside sandbox (mode=file)
    var: str = "CTF_FLAG"  # env var name (mode=env)

    def validate(self) -> None:
        if self.mode not in ("file", "env", "none"):
            raise ValidationError(f"flag.injection.mode must be file|env|none, got {self.mode!r}")
        if self.mode == "file":
            if not self.path:
                raise ValidationError("flag.injection.path is required for mode=file")
            _check_relpath(self.path)
        elif self.mode == "env" and not re.match(r"^[A-Z_][A-Z0-9_]*$", self.var or ""):
            raise ValidationError(f"flag.injection.var invalid: {self.var!r}")


@dataclass
class FlagSpec:
    mode: str = "generated"  # generated | static
    verify: str = "exact"  # regex | exact | script
    format: str = "flag{uuid4}"
    sha256: Optional[str] = None  # digest of the static flag (never the flag itself)
    injection: FlagInjection = field(default_factory=FlagInjection)

    def validate(self) -> None:
        if self.mode not in FLAG_MODES:
            raise ValidationError(f"flag.mode must be one of {FLAG_MODES}, got {self.mode!r}")
        if self.verify not in VERIFY_MODES:
            raise ValidationError(f"flag.verify must be one of {VERIFY_MODES}, got {self.verify!r}")
        if not self.format or not isinstance(self.format, str):
            raise ValidationError("flag.format must be a non-empty string")
        if self.mode == "static":
            if not self.sha256 or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
                raise ValidationError("flag.sha256 must be a 64-hex digest for static flags")
        self.injection.validate()


@dataclass
class Task:
    task_id: str
    source: str  # nyuctf | random-crypto | custom
    category: str
    env: EnvSpec
    flag: FlagSpec
    prompt: str
    horizon: Horizon = field(default_factory=Horizon)
    split: str = "train"
    # relative asset paths (inside the task bundle dir) uploaded to the sandbox
    assets: tuple[str, ...] = ()
    compose_file: Optional[str] = None  # relative path, launch=compose only
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not TASK_ID_RE.match(self.task_id or ""):
            raise ValidationError(f"task_id invalid: {self.task_id!r}")
        if self.source not in SOURCES:
            raise ValidationError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.category not in CATEGORIES:
            raise ValidationError(f"category must be one of {CATEGORIES}, got {self.category!r}")
        if self.split not in SPLITS:
            raise ValidationError(f"split must be one of {SPLITS}, got {self.split!r}")
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValidationError("prompt must be a non-empty string")
        self.env.validate()
        self.flag.validate()
        self.horizon.validate()
        for a in self.assets:
            _check_relpath(a)
        if self.env.launch == "compose":
            if not self.compose_file:
                raise ValidationError("compose_file required when env.launch == compose")
            _check_relpath(self.compose_file)
            # compose/DinD is not suitable for strict anti-cheat locked eval
            self.metadata.setdefault("anti_cheat_note", "compose/DinD launch is not strict-egress-safe unless flattened")

    def to_public_dict(self) -> dict[str, Any]:
        """Public JSON view. Never contains plaintext flags."""
        self.validate()
        return {
            "task_id": self.task_id,
            "source": self.source,
            "category": self.category,
            "env": asdict(self.env),
            "flag": {
                "mode": self.flag.mode,
                "verify": self.flag.verify,
                "format": self.flag.format,
                "sha256": self.flag.sha256,
                "injection": asdict(self.flag.injection),
            },
            "prompt": self.prompt,
            "horizon": asdict(self.horizon),
            "split": self.split,
            "assets": list(self.assets),
            "compose_file": self.compose_file,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        try:
            env = EnvSpec(**d["env"])
            flag_d = dict(d["flag"])
            flag = FlagSpec(
                mode=flag_d["mode"],
                verify=flag_d["verify"],
                format=flag_d["format"],
                sha256=flag_d.get("sha256"),
                injection=FlagInjection(**flag_d.get("injection", {})),
            )
            return cls(
                task_id=d["task_id"],
                source=d["source"],
                category=d["category"],
                env=env,
                flag=flag,
                prompt=d["prompt"],
                horizon=Horizon(**d.get("horizon", {})),
                split=d.get("split", "train"),
                assets=tuple(d.get("assets", ())),
                compose_file=d.get("compose_file"),
                metadata=dict(d.get("metadata", {})),
            )
        except KeyError as e:
            raise ValidationError(f"missing required field: {e}") from e
        except TypeError as e:
            raise ValidationError(f"malformed task dict: {e}") from e


@dataclass
class Obs:
    step: int
    content: str
    exit_code: Optional[int] = None
    truncated: bool = False
    sandbox_id: Optional[str] = None
    done: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptMessage:
    turn: int
    role: str  # assistant | tool
    content: str
    model: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn, "role": self.role, "content": self.content, "model": self.model}


@dataclass
class Transcript:
    task_id: str
    episode_id: str
    policy: str
    split: str
    messages: list[TranscriptMessage] = field(default_factory=list)
    solved: bool = False
    steps: int = 0
    flags_found: list[str] = field(default_factory=list)
    sandbox_id: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None

    def validate(self) -> None:
        if not self.task_id or not self.episode_id or not self.policy:
            raise ValidationError("transcript requires task_id, episode_id, policy")
        if self.split not in SPLITS:
            raise ValidationError(f"transcript.split invalid: {self.split!r}")
        for m in self.messages:
            if m.role not in ("assistant", "tool", "user", "system"):
                raise ValidationError(f"message.role invalid: {m.role!r}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "policy": self.policy,
            "split": self.split,
            "messages": [m.to_dict() for m in self.messages],
            "solved": self.solved,
            "steps": self.steps,
            "flags_found": list(self.flags_found),
            "sandbox_id": self.sandbox_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        return cls(
            task_id=d["task_id"],
            episode_id=d["episode_id"],
            policy=d["policy"],
            split=d.get("split", "train"),
            messages=[TranscriptMessage(**m) for m in d.get("messages", [])],
            solved=bool(d.get("solved", False)),
            steps=int(d.get("steps", 0)),
            flags_found=list(d.get("flags_found", [])),
            sandbox_id=d.get("sandbox_id"),
            tokens_in=int(d.get("tokens_in", 0)),
            tokens_out=int(d.get("tokens_out", 0)),
            error=d.get("error"),
        )


def load_task(path: str) -> Task:
    with open(path, "r", encoding="utf-8") as f:
        task = Task.from_dict(json.load(f))
    task.validate()
    return task


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _check_relpath(p: str) -> None:
    """Reject absolute paths and traversal outside the task bundle."""
    if not isinstance(p, str) or not p:
        raise ValidationError(f"path must be a non-empty relative string, got {p!r}")
    if p.startswith("/") or p.startswith("\\") or (len(p) > 1 and p[1] == ":"):
        raise ValidationError(f"absolute path not allowed: {p!r}")
    parts = p.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        raise ValidationError(f"path traversal not allowed: {p!r}")


def task_json_schema() -> dict[str, Any]:
    """JSON Schema for the public task JSON (also written to schemas/)."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ctf_gym public task",
        "type": "object",
        "required": ["task_id", "source", "category", "env", "flag", "prompt", "horizon", "split"],
        "additionalProperties": False,
        "properties": {
            "task_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,127}$"},
            "source": {"enum": list(SOURCES)},
            "category": {"enum": list(CATEGORIES)},
            "env": {
                "type": "object",
                "required": ["image", "launch"],
                "additionalProperties": False,
                "properties": {
                    "image": {"type": "string", "minLength": 1},
                    "launch": {"enum": list(LAUNCH_MODES)},
                },
            },
            "flag": {
                "type": "object",
                "required": ["mode", "verify", "format", "injection"],
                "additionalProperties": False,
                "properties": {
                    "mode": {"enum": list(FLAG_MODES)},
                    "verify": {"enum": list(VERIFY_MODES)},
                    "format": {"type": "string"},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "injection": {
                        "type": "object",
                        "required": ["mode"],
                        "properties": {
                            "mode": {"enum": ["file", "env", "none"]},
                            "placeholder": {"type": "string"},
                            "path": {"type": ["string", "null"]},
                            "var": {"type": "string"},
                        },
                    },
                },
            },
            "prompt": {"type": "string", "minLength": 1},
            "horizon": {
                "type": "object",
                "required": ["max_steps", "timeout_s"],
                "properties": {
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "timeout_s": {"type": "integer", "minimum": 1, "maximum": 86400},
                },
            },
            "split": {"enum": list(SPLITS)},
            "assets": {"type": "array", "items": {"type": "string"}},
            "compose_file": {"type": ["string", "null"]},
            "metadata": {"type": "object"},
        },
    }


def transcript_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ctf_gym transcript record",
        "type": "object",
        "required": ["task_id", "episode_id", "policy", "split", "messages", "solved", "steps"],
        "properties": {
            "task_id": {"type": "string"},
            "episode_id": {"type": "string"},
            "policy": {"type": "string"},
            "split": {"enum": list(SPLITS)},
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["turn", "role", "content"],
                    "properties": {
                        "turn": {"type": "integer"},
                        "role": {"enum": ["assistant", "tool", "user", "system"]},
                        "content": {"type": "string"},
                        "model": {"type": ["string", "null"]},
                    },
                },
            },
            "solved": {"type": "boolean"},
            "steps": {"type": "integer"},
            "flags_found": {"type": "array", "items": {"type": "string"}},
            "sandbox_id": {"type": ["string", "null"]},
            "tokens_in": {"type": "integer"},
            "tokens_out": {"type": "integer"},
            "error": {"type": ["string", "null"]},
        },
    }


def secure_verifier_write(path: str, payload: dict[str, Any]) -> None:
    """Write a verifier record sidecar with 0600 permissions where supported."""
    data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # e.g. some filesystems
