"""Orchestrator: run the alloy replication across configs, in parallel.

For each challenge in benchmark/subset.json we run 3 configs:
  * single:<opus>     — baseline A
  * single:<glm>      — baseline B
  * alloy:<opus+glm>  — the paper's method (random per-step model swap)
each repeated --attempts times. Every (challenge, config, attempt) gets its own
fresh sandbox with a freshly planted flag. Results stream to results/runs.jsonl
(resumable: completed jobs are skipped on re-run).
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
from alloy.providers import AnthropicAdapter, OpenAICompatAdapter
from alloy.agent import run_episode, SingleModel, Alloy
from alloy.sandbox import client, Sandbox
from benchmark.rehost import load_challenge, rehost, challenge_prompt

REPO = os.path.join(ROOT, "benchmark_repo")
OPUS = "claude-opus-4-8"
SECOND = "claude-sonnet-4-5-20250929"
_write_lock = threading.Lock()


def make_adapters():
    opus = AnthropicAdapter(os.environ["ANTHROPIC_API_KEY"], OPUS, max_tokens=4096)
    second = AnthropicAdapter(os.environ["ANTHROPIC_API_KEY"], SECOND, max_tokens=4096)
    return opus, second


def make_policy(config: str, seed: int):
    opus, second = make_adapters()
    if config == "single_opus":
        return SingleModel(opus)
    if config == "single_sonnet":
        return SingleModel(second)
    if config == "alloy":
        return Alloy([opus, second], seed=seed)
    raise ValueError(config)


def job_key(cid, config, attempt):
    return f"{cid}|{config}|{attempt}"


def load_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done.add(job_key(r["cid"], r["config"], r["attempt"]))
            except Exception:  # noqa: BLE001
                pass
    return done


def run_job(rec, max_iters, out_path, deadline_ts=None, tdir=None):
    cid, config, attempt = rec["cid"], rec["config"], rec["attempt"]
    # skip cleanly if we're past the wall-clock deadline (don't even spin a box)
    if deadline_ts and time.time() > deadline_ts:
        rec_out = {"cid": cid, "config": config, "attempt": attempt,
                   "solved": False, "iters": 0, "error": "skipped (deadline)",
                   "model_sequence": [], "submitted": None, "seconds": 0}
        with _write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec_out) + "\n")
        print(f"  [skip  ] {config:12} {cid} (deadline)", flush=True)
        return rec_out
    seed = (abs(hash((cid, attempt))) % 100000)
    d = client()
    sb = None
    t0 = time.time()
    result = {"cid": cid, "config": config, "attempt": attempt,
              "solved": False, "iters": 0, "error": None,
              "model_sequence": [], "submitted": None}
    ep = None
    try:
        ch = load_challenge(cid, rec["name"], rec["ctype"],
                            os.path.join(REPO, rec["path"]))
        sb = Sandbox(d, ttl_minutes=20)
        flag, info = rehost(sb, ch)
        if not info["direct_read_blocked"]:
            result["error"] = "flag readable directly (rehost invalid)"
        else:
            policy = make_policy(config, seed)
            ep = run_episode(policy, challenge_prompt=challenge_prompt(ch),
                             planted_flag=flag, bash_exec=sb.agent_exec,
                             max_iters=max_iters)
            result.update(solved=ep.solved, iters=ep.iters,
                          error=ep.error, submitted=ep.submitted,
                          model_sequence=ep.model_sequence)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"job error: {e}"
    finally:
        if sb:
            sb.destroy()
    result["seconds"] = round(time.time() - t0, 1)
    # save full transcript for inspection / the deliverable
    if tdir and ep is not None:
        try:
            os.makedirs(tdir, exist_ok=True)
            tf = os.path.join(tdir, f"{cid}__{config}__{attempt}.json")
            json.dump({"system": ep.transcript.system,
                       "messages": ep.transcript.messages,
                       "model_sequence": ep.model_sequence,
                       "solved": ep.solved, "submitted": ep.submitted},
                      open(tf, "w"), indent=1)
        except Exception:  # noqa: BLE001
            pass
    with _write_lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")
    tag = "SOLVED" if result["solved"] else ("ERR" if result["error"] else "miss")
    print(f"  [{tag:6}] {config:12} {cid} (attempt {attempt}) "
          f"{result['iters']}it {result['seconds']}s", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=os.path.join(ROOT, "benchmark", "subset.json"))
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=80)
    ap.add_argument("--parallel", type=int, default=30)
    ap.add_argument("--configs", default="single_opus,single_sonnet,alloy")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="cap #challenges (0=all)")
    ap.add_argument("--deadline-min", type=float, default=0,
                    help="stop launching new jobs after this many minutes (0=off)")
    ap.add_argument("--transcripts", default=os.path.join(ROOT, "results", "transcripts"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    subset = json.load(open(args.subset))
    if args.limit:
        subset = subset[:args.limit]
    configs = args.configs.split(",")
    done = load_done(args.out)

    jobs = []
    for ch in subset:
        for config in configs:
            for attempt in range(args.attempts):
                if job_key(ch["cid"], config, attempt) in done:
                    continue
                jobs.append({**ch, "config": config, "attempt": attempt})

    t0 = time.time()
    deadline_ts = (t0 + args.deadline_min * 60) if args.deadline_min else None
    print(f"{len(subset)} challenges x {len(configs)} configs x {args.attempts} "
          f"attempts = {len(subset)*len(configs)*args.attempts} total; "
          f"{len(done)} already done, {len(jobs)} to run; parallel={args.parallel}; "
          f"deadline={args.deadline_min}min", flush=True)
    solved = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_job, j, args.max_iters, args.out,
                          deadline_ts, args.transcripts) for j in jobs]
        for fut in as_completed(futs):
            if fut.result()["solved"]:
                solved += 1
    print(f"\nDONE: ran {len(jobs)} jobs in {time.time()-t0:.0f}s; "
          f"{solved} newly solved. Results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
