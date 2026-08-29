"""NYU CTF Bench converter.

Reads the upstream repo's ``development_dataset.json`` / ``test_dataset.json``
(together with each challenge's ``challenge.json``) directly — no vendoring of
challenge content; you must clone https://github.com/NYU-LLM-CTF/NYU_CTF_Bench
(GPL-2.0) locally.

Maps the 6 NYU categories (crypto / pwn / rev / web / forensics / misc),
copies the portable assets needed by each task into the task bundle, formats
local host/port info, chooses launch mode compose|none, and writes external
SHA-256 verifier records. Upstream dataset sizes at time of writing:
development 57, test 200.

Expected upstream challenge.json fields (tolerant to extras):
  name, category, description, flag, files (list), box (host), type
  ("default" | "compose"), plus optional composefile path under the
  challenge directory.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Optional

from ctf_gym.contracts import (
    EnvSpec,
    FlagInjection,
    FlagSpec,
    Horizon,
    Task,
    sha256_hex,
)
from ctf_gym.converters.random_crypto import ConvertedTask, ConverterError
from ctf_gym.verifier import make_verifier_record

CATEGORY_MAP = {
    "crypto": "crypto",
    "cryptography": "crypto",
    "pwn": "pwn",
    "binary_exploitation": "pwn",
    "rev": "rev",
    "reverse_engineering": "rev",
    "web": "web",
    "web_exploitation": "web",
    "forensics": "forensics",
    "misc": "misc",
    "warmups": "misc",
}

SPLIT_FILES = {"train": "development_dataset.json", "eval": "test_dataset.json"}


def map_category(raw: str) -> str:
    key = (raw or "").strip().lower().replace(" ", "_")
    if key in CATEGORY_MAP:
        return CATEGORY_MAP[key]
    if key in CATEGORY_MAP.values():
        return key
    return "misc"


def _slug(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_." and (not out or out[-1] != "-"):
            out.append("-")
    return "".join(out).strip("-")[:80] or "task"


def _find_challenge_dir(repo: str, name: str, year_hint: Optional[str]) -> Optional[str]:
    """Locate <repo>/<year>/<category-ish>/<name>/challenge.json by name scan."""
    if year_hint:
        cand = os.path.join(repo, year_hint, name)
        if os.path.isfile(os.path.join(cand, "challenge.json")):
            return cand
    for root, dirs, files in os.walk(repo):
        if "challenge.json" in files and os.path.basename(root) == name:
            return root
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules")]
    return None


def _format_box(description: str, box: Optional[str]) -> str:
    if box:
        return description.rstrip() + f"\n\nThe challenge is running at: {box} (host reachable inside the sandbox)."
    return description


def convert_challenge(repo: str, name: str, year_hint: Optional[str], split: str,
                      image: str = "ghcr.io/ctf-gym/ctf-tools:latest",
                      index: int = 0, max_steps: int = 40, timeout_s: int = 1800) -> Optional[ConvertedTask]:
    chdir = _find_challenge_dir(repo, name, year_hint)
    if chdir is None:
        return None
    with open(os.path.join(chdir, "challenge.json"), encoding="utf-8") as f:
        chal: dict[str, Any] = json.load(f)

    raw_cat = chal.get("category", "misc")
    category = map_category(raw_cat)
    task_id = f"nyuctf-{_slug(chal.get('name') or name)}"
    files = list(chal.get("files") or [])
    is_compose = (chal.get("type") or "default").lower() == "compose"
    compose_rel = None
    if is_compose:
        # upstream stores compose files inside the challenge dir
        for cand in ("docker-compose.yml", "docker-compose.yaml", "compose.yml"):
            if os.path.isfile(os.path.join(chdir, cand)):
                compose_rel = f"assets/{task_id}/{cand}"
                break
        if compose_rel is None:
            raise ConverterError(f"compose challenge {name} has no compose file in {chdir}")

    flag = str(chal.get("flag") or "").strip()
    if not flag:
        raise ConverterError(f"challenge {name} has no flag")

    task = Task(
        task_id=task_id,
        source="nyuctf",
        category=category,
        env=EnvSpec(image=image, launch="compose" if is_compose else "none"),
        flag=FlagSpec(
            mode="static",
            verify="exact",
            format=flag.split("{", 1)[0] + "{...}" if "{" in flag else "flag{...}",
            sha256=sha256_hex(flag),
            # The flag is baked into the upstream challenge distribution
            # (server side / compose); we never inject it into the sandbox.
            # Tasks whose flag only exists in plaintext client files are
            # inherently leak-prone and are flagged in _conversion.json.
            injection=FlagInjection(mode="none"),
        ),
        prompt=_format_box(str(chal.get("description") or name), chal.get("box")),
        horizon=Horizon(max_steps=max_steps, timeout_s=timeout_s),
        split=split,
        assets=tuple(f"assets/{task_id}/{fn}" for fn in files),
        compose_file=compose_rel,
        metadata={
            "nyuctf_name": chal.get("name") or name,
            "nyuctf_category": raw_cat,
            "year": year_hint or "",
            "challenge_dir": os.path.relpath(chdir, repo),
        },
    )
    task.validate()
    ct = ConvertedTask(task=task, static_flag=flag)
    ct.challenge_dir = chdir  # type: ignore[attr-defined]
    ct.files = files  # type: ignore[attr-defined]
    return ct


def convert_dataset(repo: str, split: str, out_dir: str, verifiers_dir: Optional[str] = None,
                    assets_root: Optional[str] = None, image: str = "ghcr.io/ctf-gym/ctf-tools:latest",
                    limit: Optional[int] = None) -> list[ConvertedTask]:
    """Convert an NYU dataset split into public tasks + verifier records + assets.

    split: train -> development_dataset.json (57 challenges upstream)
           eval  -> test_dataset.json (200 challenges upstream)
    """
    if split not in SPLIT_FILES:
        raise ConverterError(f"split must be train|eval, got {split!r}")
    dataset_path = os.path.join(repo, SPLIT_FILES[split])
    if not os.path.isfile(dataset_path):
        raise ConverterError(
            f"{dataset_path} not found — clone the NYU_CTF_Bench repo (GPL-2.0) first"
        )
    with open(dataset_path, encoding="utf-8") as f:
        entries = json.load(f)
    # entries: list of challenge names or dicts {name/category/...}
    names: list[tuple[str, Optional[str]]] = []
    for e in entries:
        if isinstance(e, str):
            names.append((e, None))
        elif isinstance(e, dict):
            names.append((str(e.get("name") or ""), e.get("year")))
    if assets_root is None:
        assets_root = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "assets")
    converted: list[ConvertedTask] = []
    skipped: list[str] = []
    for i, (name, year) in enumerate(names):
        if not name:
            continue
        if limit is not None and len(converted) >= limit:
            break
        try:
            ct = convert_challenge(repo, name, year, split=split, image=image, index=i)
        except ConverterError as e:
            skipped.append(f"{name}: {e}")
            continue
        if ct is None:
            skipped.append(f"{name}: challenge dir not found")
            continue
        converted.append(ct)
    # write outputs
    os.makedirs(out_dir, exist_ok=True)
    if verifiers_dir is None:
        verifiers_dir = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "verifiers")
    os.makedirs(verifiers_dir, exist_ok=True)
    os.makedirs(assets_root, exist_ok=True)
    for ct in converted:
        task = ct.task
        public_path = os.path.join(out_dir, f"{task.task_id}.json")
        with open(public_path, "w", encoding="utf-8") as f:
            json.dump(task.to_public_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        make_verifier_record(task.task_id, task.flag, static_flag=ct.static_flag).write(
            os.path.join(verifiers_dir, f"{task.task_id}.json"))
        # copy portable assets (challenge distribution files), replacing nothing
        # here: placeholder substitution happens at episode time in the env.
        for rel in task.assets:
            dst = os.path.join(assets_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src_name = rel.split("/", 2)[2] if rel.count("/") >= 2 else os.path.basename(rel)
            src = os.path.join(ct.challenge_dir, src_name)  # type: ignore[attr-defined]
            if os.path.isfile(src):
                shutil.copyfile(src, dst)
            else:
                skipped.append(f"{task.task_id}: asset missing upstream: {src_name}")
        if task.compose_file:
            dst = os.path.join(assets_root, task.compose_file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = os.path.join(ct.challenge_dir, os.path.basename(task.compose_file))  # type: ignore[attr-defined]
            shutil.copyfile(src, dst)
    meta_path = os.path.join(out_dir, "_conversion.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"split": split, "converted": len(converted), "skipped": skipped,
                   "source_repo": os.path.abspath(repo)}, f, indent=2)
    return converted
