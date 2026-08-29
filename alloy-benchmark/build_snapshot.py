"""Build the one-time Daytona 'solver' snapshot: Ubuntu 22.04 + a broad CTF
toolset. All attempts boot from this snapshot; challenge files are uploaded at
boot time. x86-64 so the challenges' flagCheck ELF binaries run natively.
"""
import config
from daytona import Daytona, DaytonaConfig, CreateSnapshotParams, Image, Resources

APT = [
    "python3", "python3-pip", "python3-dev",
    "build-essential", "gdb", "file", "binutils", "xxd", "bsdmainutils",
    "foremost", "binwalk", "exiftool", "steghide", "tshark", "tcpdump",
    "netcat-openbsd", "unzip", "p7zip-full", "openssl", "ltrace", "strace",
    "libc6-i386", "lib32z1", "libssl-dev", "libgmp-dev", "git", "curl",
]

PIP = [
    "pwntools", "pycryptodome", "sympy", "gmpy2", "requests", "numpy",
    "z3-solver", "primefac", "factordb-pycli",
    "scapy", "Pillow", "capstone", "ropgadget",
]


def main():
    d = Daytona(DaytonaConfig(api_key=config.DAYTONA_API_KEY))
    img = (
        Image.base("ubuntu:22.04")
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
        .run_commands(
            "apt-get update",
            "apt-get install -y --no-install-recommends " + " ".join(APT),
            "ln -sf /usr/bin/python3 /usr/bin/python || true",
            "python3 -m pip install --no-cache-dir --upgrade pip",
        )
        .pip_install(PIP)
        .run_commands("mkdir -p /challenge")
        .workdir("/challenge")
        .entrypoint(["/bin/sh", "-c", "sleep infinity"])
    )
    print(f"Building snapshot '{config.SOLVER_SNAPSHOT}' (this can take several minutes)...")
    d.snapshot.create(
        CreateSnapshotParams(
            name=config.SOLVER_SNAPSHOT,
            image=img,
            resources=Resources(cpu=config.SANDBOX_CPU, memory=config.SANDBOX_MEM, disk=config.SANDBOX_DISK),
        ),
        on_logs=print,
    )
    print("Snapshot build complete.")


if __name__ == "__main__":
    main()
