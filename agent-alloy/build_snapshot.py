"""Build a reusable Daytona snapshot with a CTF toolchain baked in.

All challenge sandboxes are created from this snapshot so they launch fast and
already contain the tools an autonomous solver needs (python3, pwntools,
gcc/gdb, binutils, common crypto libs). Run once; safe to re-run (idempotent by
name — will skip/replace).
"""
import os, sys, time
from dotenv import load_dotenv; load_dotenv(".env")
from daytona import Daytona, DaytonaConfig, Image, CreateSnapshotParams, Resources

SNAPSHOT = "alloy-ctf-toolchain-v1"

APT = ("python3 python3-pip python3-dev python-is-python3 gcc g++ make gdb "
       "file binutils xxd ltrace strace netcat-openbsd socat curl wget unzip git "
       "ruby perl bsdmainutils libc6-i386 libc6-dbg gcc-multilib "
       "libgmp-dev libmpfr-dev libssl-dev patchelf")
PIP = ["pwntools", "pycryptodome", "sympy", "gmpy2", "z3-solver", "requests",
       "ropgadget", "capstone", "unicorn", "keystone-engine"]

def main():
    d = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    existing = [s.name for s in d.snapshot.list().items]
    if SNAPSHOT in existing and "--force" not in sys.argv:
        print(f"snapshot '{SNAPSHOT}' already exists — skipping (use --force to rebuild)")
        return
    if SNAPSHOT in existing:
        print(f"deleting existing '{SNAPSHOT}'...")
        d.snapshot.delete(d.snapshot.get(SNAPSHOT))
        for _ in range(30):
            time.sleep(2)
            if SNAPSHOT not in [x.name for x in d.snapshot.list().items]:
                break
        print("  delete propagated")
    img = (Image.base("ubuntu:22.04")
           .env({"DEBIAN_FRONTEND": "noninteractive"})
           .run_commands(
               "apt-get update",
               f"apt-get install -y --no-install-recommends {APT}",
               "rm -rf /var/lib/apt/lists/*",
               # a non-root 'hacker' user the agent will run as
               "useradd -m -s /bin/bash hacker",
               "ln -sf /usr/bin/python3 /usr/bin/python",
               "python3 -m pip install --upgrade pip",
               "mkdir -p /challenge && chown root:root /challenge")
           .pip_install(*PIP))
    print("building snapshot (first build compiles the toolchain, be patient)...")
    t0 = time.time()
    d.snapshot.create(
        CreateSnapshotParams(name=SNAPSHOT, image=img,
                             resources=Resources(cpu=1, memory=1, disk=4)),
        on_logs=lambda m: print("  ", m[:160]), timeout=1800)
    print(f"snapshot '{SNAPSHOT}' built in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
