"""Async episode runner: bounded concurrency, deterministic ids/seeds,
append-only canonical JSONL transcripts with resumability and error records."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from ctf_gym.contracts import Obs, Task, Transcript, TranscriptMessage, sha256_hex
from ctf_gym.env.base import BaseCTFEnv
from ctf_gym.eval.parser import extract_declared_flags, parse_action
from ctf_gym.eval.policy import Policy, PolicyError
from ctf_gym.verifier import VerifierRecord


def deterministic_episode_id(run_id: str, task_id: str, k: int) -> str:
    digest = sha256_hex(f"{run_id}|{task_id}|{k}")
    return f"{task_id}-e{k:03d}-{digest[:10]}"


def deterministic_seed(run_id: str, episode_id: str) -> int:
    return int(sha256_hex(f"seed|{run_id}|{episode_id}")[:8], 16)


@dataclass
class EpisodeResult:
    transcript: Transcript
    exit_code: int = 0  # 0 ok, 1 episode error (recorded as error transcript)


class TranscriptWriter:
    """Append-only JSONL writer with resume support."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._seen: set[str] = set()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "episode_id" in rec:
                        self._seen.add(rec["episode_id"])

    def has(self, episode_id: str) -> bool:
        return episode_id in self._seen

    def append(self, t: Transcript) -> None:
        rec = t.to_dict()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seen.add(t.episode_id)


def build_env_factory():
    """Default env factory: picks backend by launch mode / env prefix."""
    from ctf_gym.env.docker_env import DockerEnv
    from ctf_gym.env.repl_env import ReplEnv

    def factory(task: Task, verifier: VerifierRecord, **kwargs: Any) -> BaseCTFEnv:
        backend = kwargs.pop("backend", None)
        if backend == "repl" or (backend is None and task.env.launch == "repl"):
            return ReplEnv(task, verifier, **kwargs)
        if backend == "daytona":
            from ctf_gym.env.daytona_env import DaytonaEnv

            return DaytonaEnv(task, verifier, **kwargs)
        return DockerEnv(task, verifier, **kwargs)

    return factory


default_env_factory = build_env_factory()


async def run_episode(
    task: Task,
    verifier: VerifierRecord,
    policy: Policy,
    *,
    run_id: str,
    k: int,
    env_factory: Callable[..., BaseCTFEnv] = default_env_factory,
    assets_dir: Optional[str] = None,
    env_kwargs: Optional[dict[str, Any]] = None,
) -> EpisodeResult:
    """Run one episode for the k-th independent sample of a task."""
    episode_id = deterministic_episode_id(run_id, task.task_id, k)
    seed = deterministic_seed(run_id, episode_id)
    transcript = Transcript(task_id=task.task_id, episode_id=episode_id,
                            policy=policy.name, split=task.split)
    env: Optional[BaseCTFEnv] = None
    try:
        env = env_factory(
            task, verifier,
            run_id=run_id, episode_id=episode_id, seed=seed,
            assets_dir=assets_dir, **(env_kwargs or {}),
        )
        obs: Obs = await env.reset(seed=seed)
        transcript.sandbox_id = env.sandbox_id
        history: list[dict[str, Any]] = [
            {"role": "user", "content": f"[sandbox] {obs.content}"}
        ]
        done = False
        while not done:
            resp = await policy.act(task.prompt, history)
            transcript.messages.append(TranscriptMessage(
                turn=transcript.steps, role="assistant", content=resp.text, model=resp.model))
            transcript.tokens_in += resp.tokens_in
            transcript.tokens_out += resp.tokens_out
            # never trust self-report: declared flags still get verified
            for declared in extract_declared_flags(resp.text):
                if env.submit_flag(declared):
                    transcript.flags_found.append(declared)
                    break
            parsed = parse_action(resp.text)
            if parsed.command is None:
                tool_out = "[no command parsed from model output — reply with a ```bash block]"
                done = False
            else:
                obs = await env.step(parsed.command)
                tool_out = obs.content
            transcript.messages.append(TranscriptMessage(
                turn=transcript.steps, role="tool", content=tool_out, model=None))
            history.append({"role": "assistant", "content": resp.text})
            history.append({"role": "user", "content": f"[tool output]\n{tool_out}"})
            done = obs.done
        transcript.solved = env.solved()
        if env.verified_flag and env.verified_flag not in transcript.flags_found:
            transcript.flags_found.append(env.verified_flag)
        transcript.steps = len([m for m in transcript.messages if m.role == "assistant"])
        return EpisodeResult(transcript=transcript)
    except Exception as e:  # error transcripts, never crash the batch
        transcript.error = f"{type(e).__name__}: {e}"
        transcript.solved = False
        return EpisodeResult(transcript=transcript, exit_code=1)
    finally:
        if env is not None:
            await env.close()


async def run_eval(
    tasks: Sequence[Task],
    verifiers: dict[str, VerifierRecord],
    policy: Policy,
    *,
    run_id: str,
    k: int = 1,
    concurrency: int = 4,
    out_path: str,
    env_factory: Callable[..., BaseCTFEnv] = default_env_factory,
    assets_dir: Optional[str] = None,
    env_kwargs: Optional[dict[str, Any]] = None,
) -> list[Transcript]:
    writer = TranscriptWriter(out_path)
    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[Transcript] = []

    async def one(task: Task, idx: int) -> None:
        episode_id = deterministic_episode_id(run_id, task.task_id, idx)
        if writer.has(episode_id):
            return
        async with sem:
            res = await run_episode(
                task, verifiers[task.task_id], policy, run_id=run_id, k=idx,
                env_factory=env_factory, assets_dir=assets_dir, env_kwargs=env_kwargs,
            )
        writer.append(res.transcript)
        results.append(res.transcript)

    await asyncio.gather(*(one(t, i) for t in tasks for i in range(k)))
    return results
