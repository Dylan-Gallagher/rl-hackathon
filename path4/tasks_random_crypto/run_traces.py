"""Runner: real GLM episodes (train traces + eval baselines) via path4/ensemble.

Usage:
  ZAI_API_KEY=... python -m path4.tasks_random_crypto.run_traces \
      --mode eval --policy solo:glm-4.6 --out runs/eval_baselines/solo_glm-4.6.jsonl \
      --k 4 --concurrency 4

  ... --mode train --policy alloy:glm-4.6:0.5,glm-4.5-air:0.5 \
      --out runs/traces/train/alloy.jsonl --target-solves 120 --max-rounds 4

Transcripts are the canonical §3.2 JSONL (one episode per line), appended
incrementally so a killed run keeps whatever finished (salvage semantics).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from path4.tasks_random_crypto.env_hard import HardReplCTFEnv as ReplCTFEnv
from contracts.task import Task, load_tasks
from contracts.transcript import write_transcript

from path4.ensemble.agent import run_episode
from path4.tasks_random_crypto.teachers import make_teacher_client
from path4.ensemble.policies import parse_policy

EPISODE_TIMEOUT_S = 330
MAX_STEPS = 12


async def one_episode(
    task: Task, pol, client: ChatClient, sem: asyncio.Semaphore, race_id: str
):
    async with sem:
        env = ReplCTFEnv(task)
        try:
            return await run_episode(
                task, pol, client, env, max_steps=MAX_STEPS,
                race_id=race_id, timeout_s=EPISODE_TIMEOUT_S,
                seed=abs(hash(race_id + task.task_id)) % (2**31),
            )
        except Exception as e:  # salvage: log and move on
            return {"error": f"{type(e).__name__}: {e}", "task_id": task.task_id}


async def run(
    mode: str, tasks_dir: Path, policy_spec: str, out: Path, k: int,
    concurrency: int, target_solves: int, max_rounds: int,
) -> None:
    tasks = load_tasks(tasks_dir)
    pol = parse_policy(policy_spec)
    client = make_teacher_client()
    sem = asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"attempts": 0, "solved": 0, "errors": 0}
    stats_path = out.parent / (out.name + ".stats.json")

    def flush() -> None:
        stats_path.write_text(json.dumps(
            {**stats, "policy": pol.name(), "out": str(out), "ts": time.time()}, indent=2))

    round_i = 0
    while True:
        round_i += 1
        pending = list(tasks) if mode == "train" and round_i > 1 else list(tasks)
        if mode == "train" and round_i == 1:
            pending = list(tasks)
        jobs = []
        for t in pending:
            for a in range(k if mode == "eval" else 1):
                jobs.append(one_episode(t, pol, client, sem, f"{mode}-r{round_i}-a{a}"))
        random.Random(round_i).shuffle(jobs)
        batch = asyncio.as_completed(jobs)
        n = len(jobs)
        for i, fut in enumerate(batch, 1):
            res = await fut
            if isinstance(res, dict) and "error" in res:
                stats["errors"] += 1
                print(f"[{pol.name()} r{round_i} {i}/{n}] ERROR {res['error']}", flush=True)
                continue
            write_transcript(res, out, mode="a")
            stats["attempts"] += 1
            stats["solved"] += int(bool(res.solved))
            flush()
            if i % 5 == 0 or res.solved:
                print(f"[{pol.name()} r{round_i} {i}/{n}] solved={res.solved} "
                      f"steps={res.steps} total={stats['solved']}/{stats['attempts']}",
                      flush=True)
        flush()
        if mode != "train" or stats["solved"] >= target_solves or round_i >= max_rounds:
            break
        print(f"[{pol.name()}] round {round_i} done: {stats['solved']}/{target_solves}; "
              f"re-attempting all tasks", flush=True)
    print(f"[{pol.name()}] DONE solved={stats['solved']}/{stats['attempts']} "
          f"errors={stats['errors']} -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=1, help="episodes per task per round")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--target-solves", type=int, default=120)
    ap.add_argument("--max-rounds", type=int, default=4)
    a = ap.parse_args()
    asyncio.run(run(a.mode, a.tasks_dir, a.policy, a.out, a.k,
                    a.concurrency, a.target_solves, a.max_rounds))


if __name__ == "__main__":
    main()
