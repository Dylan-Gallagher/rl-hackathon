"""Full sweep: 3 conditions x 30 challenges x N attempts, concurrent & resumable.

Each trajectory is independent (its own sandbox + its own conversation), so we
fan out with a thread pool. Existing trajectory files are skipped, so the sweep
can be re-run to fill gaps or after interruption.
"""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import config
from agent import run_trajectory, save_trajectory

CONDITIONS = ["A", "B", "alloy"]


def already_done(condition: str, cid: str, attempt: int) -> bool:
    p = config.TRAJ_DIR / condition / f"{cid}__attempt{attempt}.json"
    if not p.exists():
        return False
    try:
        t = json.load(open(p))
        # Re-run only hard infrastructure errors, keep legitimate unsolved runs.
        return t.get("error") in (None, "no_tool_streak")
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=config.ATTEMPTS_PER_CHALLENGE)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--challenges", nargs="+", default=None, help="subset of challenge ids")
    args = ap.parse_args()

    # Respect the account-wide concurrent caps: workers must satisfy BOTH
    # workers*mem <= mem_cap AND workers*disk <= disk_cap. Keep one slot of
    # headroom so we never sit exactly on a cap.
    mem_cap_workers = config.ACCOUNT_MEM_CAP_GIB // config.SANDBOX_MEM
    disk_cap_workers = config.ACCOUNT_DISK_CAP_GIB // config.SANDBOX_DISK
    hard_cap = max(1, min(mem_cap_workers, disk_cap_workers) - 1)
    if args.workers > hard_cap:
        print(f"Capping workers {args.workers} -> {hard_cap} "
              f"(mem {config.ACCOUNT_MEM_CAP_GIB}/{config.SANDBOX_MEM}={mem_cap_workers}, "
              f"disk {config.ACCOUNT_DISK_CAP_GIB}/{config.SANDBOX_DISK}={disk_cap_workers}).")
        args.workers = hard_cap

    manifest = json.load(open(config.CHALLENGES_DIR / "manifest.json"))
    cids = args.challenges or list(manifest)

    jobs = []
    for cond in args.conditions:
        for cid in cids:
            for attempt in range(args.attempts):
                if not already_done(cond, cid, attempt):
                    jobs.append((cond, cid, attempt))
    print(f"{len(jobs)} trajectories to run "
          f"({len(args.conditions)} conditions x {len(cids)} challenges x {args.attempts} attempts, "
          f"minus completed).")

    done = 0
    solved = 0
    t0 = time.time()

    def _run(job):
        cond, cid, attempt = job
        traj = run_trajectory(manifest[cid], cond, attempt)
        save_trajectory(traj)
        return job, traj

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run, j): j for j in jobs}
        for fut in as_completed(futs):
            job, traj = fut.result()
            done += 1
            solved += int(traj["solved"])
            elapsed = time.time() - t0
            rate = done / elapsed * 60
            eta = (len(jobs) - done) / max(rate, 1e-9)
            print(f"[{done}/{len(jobs)}] {traj['condition']:5s} {traj['challenge_id']:40s} "
                  f"solved={traj['solved']} turns={traj['n_turns']} ${traj['cost_usd']:.3f} "
                  f"| {rate:.1f}/min ETA~{eta:.0f}min")

    print(f"\nDone: {done} trajectories, {solved} solved, "
          f"{time.time()-t0:.0f}s total.")


if __name__ == "__main__":
    main()
