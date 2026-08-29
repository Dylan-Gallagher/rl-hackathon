"""Deterministic offline demo corpus for the scoreboard.

Seeds a fake-but-plausible transcript corpus: 2 solo frontier policies, 1
alloy, and a cold-start student (weak) plus its post-GRPO self (stronger),
~10 tasks across pwn/rev/crypto/web/forensics/misc, k episodes each, plus
ensemble race ``summary.json`` files under ``runs/``.

Usage:
    python -m path4.scoreboard.demo.seed_demo --out runs/demo
    python -m path4.scoreboard.demo.seed_demo            # small committed corpus

The default output (``path4/scoreboard/demo/data``) is capped at 40 episodes
so it can be committed; ``--out`` anywhere else gets the full corpus.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import count
from pathlib import Path

from contracts import Transcript, TranscriptMessage, episode_id_for, write_transcript

TASKS: list[tuple[str, str, str]] = [
    # (task_id, category, prompt)
    ("pwn-stack0", "pwn", "ret2win: overflow the buffer at 0x4040 and get the flag."),
    ("pwn-heap1", "pwn", "tcache poisoning on glibc 2.35, leak libc and pop a shell."),
    ("rev-crackme", "rev", "static crackme: recover the key transform and read the flag."),
    ("rev-vm", "rev", "bytecode VM challenge; decode the instruction set first."),
    ("crypto-lcg", "crypto", "LCG with truncated output — recover the state, decrypt the flag."),
    ("crypto-rsa-copper", "crypto", "RSA with small private exponent: Wiener attack."),
    ("web-sqli", "web", "login bypass via SQLi, then read the flag from /admin."),
    ("web-ssti", "web", "Jinja2 SSTI in the profile page — RCE to the flag file."),
    ("forensics-pcap", "forensics", "extract the transferred file from the pcap and unzip it."),
    ("misc-github", "misc", "OSINT: find the leaked token in the repo history."),
]

# (policy, per-task base solve probability, model label for assistant turns)
POLICIES = [
    ("solo:anthropic/claude-opus-4", 0.62, "anthropic/claude-opus-4"),
    ("solo:openai/gpt-5", 0.55, "openai/gpt-5"),
    ("alloy:opus,sonnet", 0.72, "alloy:opus,sonnet"),
    ("student-sft-cold", 0.18, "student-sft-cold"),
    ("student-grpo", 0.40, "student-grpo"),
]

# per-task difficulty modifier (multiplier on base prob, clamped)
TASK_MOD = {
    "pwn-heap1": 0.55,
    "rev-vm": 0.5,
    "crypto-rsa-copper": 0.8,
    "misc-github": 1.3,
    "web-sqli": 1.25,
}

_ACTIONS = [
    "ls -la /challenge",
    "file ./chal && checksec --file=./chal",
    "cat README.md",
    "python3 solve.py",
    "gdb ./chal -ex run",
    "strings ./chal | grep -i flag",
    "curl http://localhost:1337/login",
    "tshark -r capture.pcap",
]


def _episode(rng: random.Random, policy: str, model: str, task_id: str, category: str, idx: int,
             solved: bool) -> Transcript:
    steps = rng.randint(6, 30) if solved else rng.randint(10, 40)
    messages: list[TranscriptMessage] = []
    turn = 0
    n_turns = rng.randint(3, 8)
    for i in range(n_turns):
        act = rng.choice(_ACTIONS)
        messages.append(TranscriptMessage(
            turn=turn, role="assistant",
            content=f"Step {i + 1}: run `{act}` and reason about the output. "
                    f"Target: {task_id}.",
            model=model,
        ))
        messages.append(TranscriptMessage(
            turn=turn, role="tool",
            content=f"$ {act}\n[truncated output {rng.randint(20, 900)} bytes]\nok",
        ))
        turn += 1
    if solved:
        messages.append(TranscriptMessage(
            turn=turn, role="assistant",
            content=f"Got it. Flag for {task_id}: flag{{demo_{task_id.replace('-', '_')}}}",
            model=model,
        ))
    else:
        messages.append(TranscriptMessage(
            turn=turn, role="assistant",
            content="Out of ideas for now — the mitigation blocks the overwrite path.",
            model=model,
        ))
    return Transcript(
        task_id=task_id,
        episode_id=episode_id_for(policy, task_id, idx),
        policy=policy,
        split="eval",
        messages=messages,
        solved=solved,
        steps=steps,
        flags_found=[f"flag{{demo_{task_id.replace('-', '_')}}}"] if solved else [],
        tokens_in=rng.randint(2000, 20000),
        tokens_out=rng.randint(500, 8000),
        category=category,
    )


def seed(out: Path, episodes_per_task: int = 8, max_episodes: int | None = None,
         seed_value: int = 1337) -> dict:
    """Write the demo corpus under ``out``; returns a small stats dict."""
    rng = random.Random(seed_value)
    episodes = 0
    per_policy: dict[str, list[bool]] = {}
    # spread a cap across all policies (committed small-corpus story)
    per_policy_cap = (
        max(1, max_episodes // (len(POLICIES) * len(TASKS))) if max_episodes else episodes_per_task
    )
    for policy, base_p, model in POLICIES:
        pbar = count()
        for task_id, category, _prompt in TASKS:
            p = min(0.97, max(0.02, base_p * TASK_MOD.get(task_id, 1.0)))
            for _ in range(min(episodes_per_task, per_policy_cap)):
                solved = rng.random() < p
                t = _episode(rng, policy, model, task_id, category, next(pbar), solved)
                write_transcript(t, out / f"{policy.replace('/', '_')}.jsonl")
                per_policy.setdefault(policy, []).append(solved)
                episodes += 1

    # ensemble races: winner ~ best solve probability among participants
    race_dir = out / "runs"
    for i, (task_id, _c, _p) in enumerate(TASKS):
        weights = {
            pol: min(0.97, max(0.02, base * TASK_MOD.get(task_id, 1.0)))
            for pol, base, _m in POLICIES
        }
        winner = rng.choices(list(weights), weights=list(weights.values()))[0]
        summary = {
            "race_id": f"demo-race-{i:03d}",
            "task_id": task_id,
            "winner": winner,
            "per_policy": {pol: {"solved": pol == winner} for pol in weights},
        }
        race_dir.mkdir(parents=True, exist_ok=True)
        rdir = race_dir / f"race-{i:03d}"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return {
        "out": str(out),
        "episodes": episodes,
        "policies": {p: len(v) for p, v in per_policy.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed a deterministic demo transcript corpus.")
    default_out = Path(__file__).parent / "data"
    ap.add_argument("--out", type=Path, default=default_out,
                    help=f"output dir (default: {default_out}, capped at 40 episodes)")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per (policy, task)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    cap = 40 if args.out.resolve() == default_out.resolve() else None
    if args.out.resolve() == default_out.resolve() and args.out.exists():
        # keep the committed corpus reproducible: wipe and regenerate
        import shutil

        shutil.rmtree(args.out)
    stats = seed(args.out, episodes_per_task=args.episodes, max_episodes=cap,
                 seed_value=args.seed)
    cap_note = f" (capped at {cap})" if cap else ""
    print(f"seeded {stats['episodes']} episodes{cap_note} -> {stats['out']}")
    for pol, n in stats["policies"].items():
        print(f"  {pol}: {n} episodes")


if __name__ == "__main__":
    main()
