"""Flag generation, external verification, output capping, and flag scanning.

All verification runs on the host, outside any sandbox. Public task JSON only
ever sees digests; exact static flags live in verifier records (chmod 0600).
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from ctf_gym.contracts import FlagSpec, secure_verifier_write, sha256_hex

# Generous universal flag-ish pattern; exact/regex verification is stricter.
FLAG_SCAN_RE = re.compile(r"\b(?:flag|ctf|picoCTF)\{[^}\s]{1,256}\}", re.IGNORECASE)

DEFAULT_HEAD_BYTES = 4096
DEFAULT_TAIL_BYTES = 4096
TRUNCATION_MARKER = "\n...[ctf_gym: output truncated head=%d tail=%d bytes]...\n"


def generate_flag(fmt: str = "flag{uuid4}") -> str:
    """Generate a fresh flag. Supported formats: flag{uuid4} and any template
    containing the literal token ``uuid4`` (replaced by a random uuid4)."""
    if "uuid4" not in fmt:
        raise ValueError(f"unsupported generated flag format: {fmt!r}")
    return fmt.replace("uuid4", str(uuid.uuid4()))


@dataclass
class VerifierRecord:
    """External (host-side) verifier material for a task."""

    task_id: str
    mode: str  # generated | static
    verify: str  # exact | regex | script
    sha256: str
    static_flag: Optional[str] = None  # present only for static flags
    pattern: Optional[str] = None  # for verify == regex
    script: Optional[str] = None  # host-side path for verify == script

    def to_dict(self) -> dict:
        d = {"task_id": self.task_id, "mode": self.mode, "verify": self.verify, "sha256": self.sha256}
        if self.static_flag is not None:
            d["static_flag"] = self.static_flag
        if self.pattern is not None:
            d["pattern"] = self.pattern
        if self.script is not None:
            d["script"] = self.script
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VerifierRecord":
        return cls(
            task_id=d["task_id"],
            mode=d["mode"],
            verify=d["verify"],
            sha256=d["sha256"],
            static_flag=d.get("static_flag"),
            pattern=d.get("pattern"),
            script=d.get("script"),
        )

    def write(self, path: str) -> None:
        secure_verifier_write(path, self.to_dict())

    def expected_flag(self, generated: Optional[str] = None) -> str:
        if self.mode == "static":
            if self.static_flag is None:
                raise ValueError("static verifier record missing static_flag")
            return self.static_flag
        if generated is None:
            raise ValueError("generated-mode verification needs the episode flag")
        return generated


def make_verifier_record(task_id: str, spec: FlagSpec, static_flag: Optional[str] = None,
                         pattern: Optional[str] = None, script: Optional[str] = None) -> VerifierRecord:
    if spec.mode == "static":
        if not static_flag:
            raise ValueError("static flag tasks require the exact flag to build a verifier record")
        digest = sha256_hex(static_flag)
        if spec.sha256 and spec.sha256 != digest:
            raise ValueError("provided flag does not match task digest")
    else:
        digest = spec.sha256 or ""
    return VerifierRecord(task_id=task_id, mode=spec.mode, verify=spec.verify,
                          sha256=digest, static_flag=static_flag, pattern=pattern, script=script)


def verify_flag(candidate: str, record: VerifierRecord, generated: Optional[str] = None) -> bool:
    """Constant-time-ish exact digest comparison; regex mode for regex verify."""
    candidate = candidate.strip()
    if not candidate:
        return False
    if record.verify == "regex":
        if not record.pattern:
            return False
        return re.fullmatch(record.pattern, candidate) is not None
    expected = record.expected_flag(generated)
    return _digest_eq(candidate, expected)


def _digest_eq(a: str, b: str) -> bool:
    da, db = sha256_hex(a), sha256_hex(b)
    if len(da) != len(db):
        return False
    diff = 0
    for x, y in zip(da, db):
        diff |= ord(x) ^ ord(y)
    return diff == 0


def scan_for_flags(text: str) -> list[str]:
    """Scan an observation for flag-shaped strings (auto-solve detection)."""
    return list(dict.fromkeys(FLAG_SCAN_RE.findall(text)))


def solved_by_scan(observed: Sequence[str], record: VerifierRecord, generated: Optional[str] = None) -> Optional[str]:
    """Return the verified flag if any scanned candidate verifies, else None."""
    for text in observed:
        for cand in scan_for_flags(text):
            if verify_flag(cand, record, generated):
                return cand
    return None


def cap_output(text: str, head_bytes: int = DEFAULT_HEAD_BYTES, tail_bytes: int = DEFAULT_TAIL_BYTES) -> tuple[str, bool]:
    """Cap tool output with a head+tail policy, preserving both ends.

    Returns (capped_text, truncated_flag).
    """
    if head_bytes < 0 or tail_bytes < 0:
        raise ValueError("head/tail byte caps must be >= 0")
    data = text.encode("utf-8", errors="replace")
    if len(data) <= head_bytes + tail_bytes:
        return text, False
    head = data[:head_bytes].decode("utf-8", errors="replace")
    tail = data[-tail_bytes:].decode("utf-8", errors="replace") if tail_bytes else ""
    marker = TRUNCATION_MARKER % (head_bytes, tail_bytes)
    return head + marker + tail, True


def read_verifier_record(path: str) -> VerifierRecord:
    import json

    if not os.path.exists(path):
        raise FileNotFoundError(f"verifier record not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return VerifierRecord.from_dict(json.load(f))
