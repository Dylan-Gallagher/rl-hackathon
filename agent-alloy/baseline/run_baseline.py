"""Baseline orchestrator: small open-weight model through the IDENTICAL harness.

Reuses alloy.agent (same loop, tools, turn cap) + alloy.providers.OpenAICompatAdapter
pointed at the self-hosted vLLM endpoint (baseline/results/serving.json). Runs a
difficulty SPREAD to locate where an 8B shows signal (above floor, below ceiling):
  * intercode      -> InterCode-CTF (picoCTF-derived; the ~46% 'signal' band)
  * archive_pwn    -> pwn.college ctf-archive PWN subset (real CTF pwn; near-floor)
Per-episode we record solved / iters / error AND a mechanical-failure
classification (format_failures.classify) so format failures are separated from
genuine task failures. Results -> baseline/results/<benchmark>.jsonl (per-challenge).
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
from alloy.providers import OpenAICompatAdapter
from alloy.agent import run_episode, SingleModel
from alloy.sandbox import client, Sandbox
from baseline import intercode
from baseline.format_failures import classify
from benchmark.rehost import load_challenge, rehost as pwn_rehost, challenge_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
_lock = threading.Lock()


def load_serving():
    p = os.path.join(HERE, "results", "serving.json")
    if not os.path.exists(p):
        sys.exit("no serving.json — run `python baseline/serve_vllm.py` first "
                 "(requires Daytona GPU credits).")
    return json.load(open(p))


def make_adapter(serving):
    return OpenAICompatAdapter(api_key="EMPTY", model=serving["model"],
                               base_url=serving["base_url"],
                               max_tokens=serving["sampling"].get("max_tokens", 4096),
                               sampling=serving["sampling"])


def build_jobs(benchmark, limit):
    if benchmark == "intercode":
        intercode.prepare()
        tasks = intercode.load_tasks()
        if limit:
            tasks = tasks[:limit]
        return [{"kind": "intercode", "id": f"ic_{t['task_id']:03d}",
                 "tags": t.get("tags", []), "task": t} for t in tasks]
    if benchmark == "archive_pwn":
        subset = json.load(open(os.path.join(ROOT, "benchmark", "subset.json")))
        if limit:
            subset = subset[:limit]
        return [{"kind": "archive_pwn", "id": r["cid"], "tags": ["PWN"],
                 "rec": r} for r in subset]
    sys.exit(f"unknown benchmark {benchmark}")


def run_job(job, serving, max_iters, out_path, tdir):
    d = client(); sb = None; t0 = time.time()
    res = {"id": job["id"], "benchmark": job["kind"], "tags": job["tags"],
           "solved": False, "iters": 0, "error": None, "outcome": None,
           "format": {}}
    ep = None
    try:
        sb = Sandbox(d, ttl_minutes=20)
        if job["kind"] == "intercode":
            flag, _info = intercode.rehost(sb, job["task"])
            prompt = intercode.prompt(job["task"])
        else:
            ch = load_challenge(job["rec"]["cid"], job["rec"]["name"],
                                job["rec"]["ctype"], os.path.join(ROOT, "benchmark_repo", job["rec"]["path"]))
            flag, info = pwn_rehost(sb, ch)
            if not info["direct_read_blocked"]:
                res["error"] = "rehost invalid (flag readable)"; raise RuntimeError(res["error"])
            prompt = challenge_prompt(ch)
        adapter = make_adapter(serving)
        ep = run_episode(SingleModel(adapter), challenge_prompt=prompt,
                         planted_flag=flag, bash_exec=sb.agent_exec,
                         max_iters=max_iters)
        res.update(solved=ep.solved, iters=ep.iters, error=ep.error)
    except Exception as e:  # noqa: BLE001
        res["error"] = res["error"] or f"job error: {e}"
    finally:
        if sb:
            sb.destroy()
    res["seconds"] = round(time.time() - t0, 1)
    if ep is not None:
        tdict = {"messages": ep.transcript.messages, "solved": ep.solved,
                 "submitted": ep.submitted}
        res["format"] = classify(tdict)
        res["outcome"] = res["format"]["outcome"]
        if tdir:
            os.makedirs(tdir, exist_ok=True)
            json.dump(tdict, open(os.path.join(tdir, f"{job['id']}.json"), "w"))
    elif res["error"]:
        res["outcome"] = "infra_error"
    with _lock:
        open(out_path, "a").write(json.dumps(res) + "\n")
    print(f"  [{(res['outcome'] or '?'):14}] {job['id']:10} solved={res['solved']} "
          f"{res['iters']}it {res['seconds']}s {(res['error'] or '')[:40]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["intercode", "archive_pwn"], required=True)
    ap.add_argument("--max-iters", type=int, default=45)   # same cap as frontier slice
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    serving = load_serving()
    out = os.path.join(HERE, "results", f"{args.benchmark}.jsonl")
    tdir = os.path.join(HERE, "results", "transcripts", args.benchmark)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    jobs = [j for j in build_jobs(args.benchmark, args.limit) if j["id"] not in done]
    print(f"baseline {serving['model']} on {args.benchmark}: {len(jobs)} jobs "
          f"(max_iters={args.max_iters}, parallel={args.parallel})", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_job, j, serving, args.max_iters, out, tdir) for j in jobs]
        for f in as_completed(futs):
            f.result()
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
