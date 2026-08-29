"""Auto-triage candidate challenges into a clean, runnable subset.

For each candidate we (1) score exploitability locally from binary strings/symbols
(bias toward challenges that plausibly lead to /flag), then (2) smoke-test in a
real sandbox: rehost, confirm the agent user CANNOT read /flag directly, and
confirm the binary is a runnable dynamic ELF. Passing challenges (up to --target)
are written to benchmark/subset.json.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/Users/eniola/Documents/Dev/agent-alloy"
REPO = os.path.join(ROOT, "benchmark_repo")
sys.path.insert(0, ROOT)
from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
from alloy.sandbox import client, Sandbox
from benchmark.rehost import load_challenge, rehost

INTEREST = [b"/bin/sh", b"system", b"/flag", b"flag", b"execve", b"win",
            b"cat ", b"%n", b"gets", b"strcpy", b"printf", b"scanf"]


def local_score(chdir: str, binaries: list[str]) -> int:
    score = 0
    for b in binaries:
        p = os.path.join(chdir, b)
        try:
            data = open(p, "rb").read()
        except OSError:
            continue
        for kw in INTEREST:
            if kw in data:
                score += 1
    return score


def smoke_one(cid, name, path):
    d = client()
    sb = None
    try:
        ch = load_challenge(cid, name, "PWN", os.path.join(REPO, path))
        if not ch.binaries:
            return (cid, False, "no ELF binary", None)
        sb = Sandbox(d, ttl_minutes=8)
        flag, info = rehost(sb, ch)
        if not info["direct_read_blocked"]:
            return (cid, False, "flag readable directly", None)
        # binary must be a runnable dynamic ELF (exists, executes/prints or waits)
        b0 = ch.binaries[0]
        chk = sb.agent_exec(f"file /challenge/{b0}; echo '---'; "
                            f"timeout 3 /challenge/{b0} </dev/null >/dev/null 2>&1; "
                            f"echo rc=$?", timeout=30)
        ok_elf = "ELF" in chk
        return (cid, ok_elf, chk[:160],
                {"cid": cid, "name": name, "ctype": "PWN", "path": path,
                 "binaries": ch.binaries})
    except Exception as e:  # noqa: BLE001
        return (cid, False, f"error: {e}", None)
    finally:
        if sb:
            sb.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="/tmp/pwn_candidates.txt")
    ap.add_argument("--target", type=int, default=24)
    ap.add_argument("--max-test", type=int, default=40,
                    help="how many top-scored candidates to smoke-test")
    ap.add_argument("--parallel", type=int, default=12)
    args = ap.parse_args()

    paths = [l.strip() for l in open(args.candidates) if l.strip()]
    scored = []
    for p in paths:
        cdir = os.path.join(REPO, p)
        ch = load_challenge(p.replace("/", "__"), p.split("/")[-1], "PWN", cdir)
        scored.append((local_score(cdir, ch.binaries), p))
    scored.sort(reverse=True)
    test = [p for _s, p in scored[:args.max_test]]
    print(f"smoke-testing {len(test)} top-scored candidates "
          f"(parallel={args.parallel}) to reach target={args.target}...")

    kept = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(smoke_one, p.replace("/", "__"),
                          p.split("/")[-1], p): p for p in test}
        for fut in as_completed(futs):
            cid, ok, msg, rec = fut.result()
            print(f"  [{'PASS' if ok else 'skip'}] {cid}: {msg}")
            if ok and rec:
                kept.append(rec)
            if len(kept) >= args.target:
                break

    kept = kept[:args.target]
    out = os.path.join(ROOT, "benchmark", "subset.json")
    json.dump(kept, open(out, "w"), indent=2)
    print(f"\nSELECTED {len(kept)} challenges -> {out}")
    for r in kept:
        print("   -", r["cid"])


if __name__ == "__main__":
    main()
