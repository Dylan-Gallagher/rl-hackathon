"""Tests for the hardened repl env (no flag.txt shortcut)."""

from __future__ import annotations

import asyncio

from contracts.task import Task

from path4.tasks_random_crypto.env_hard import HardReplCTFEnv


def _task() -> Task:
    return Task.from_dict({
        "task_id": "rc-test-000-classical_cipher-caesar",
        "source": "random-crypto",
        "category": "crypto",
        "env": {"launch": "repl"},
        "flag": {"mode": "static", "verify": "exact", "format": "flag{secret1}"},
        "prompt": "Decipher: foo. The flag is flag{...}.",
        "horizon": {"max_steps": 12, "timeout_s": 60},
        "split": "train",
    })


def run(coro):
    return asyncio.run(coro)


def test_no_flag_file_and_cheat_blocked() -> None:
    env = HardReplCTFEnv(_task())
    run(env.reset())
    obs = run(env.step("cat flag.txt"))
    assert "No such file" in obs.output
    assert not env.solved()
    assert run(env.step("ls")).output == "challenge.py"


def test_submit_and_stdout_solve() -> None:
    env = HardReplCTFEnv(_task())
    run(env.reset())
    assert "Correct!" in run(env.step("submit flag{secret1}")).output
    assert env.solved()

    env2 = HardReplCTFEnv(_task())
    run(env2.reset())
    obs = run(env2.step("python print('flag{secret1}')"))
    assert env2.solved()
    run(env2.close())


def test_wrong_flag_not_solved() -> None:
    env = HardReplCTFEnv(_task())
    run(env.reset())
    run(env.step("submit flag{nope}"))
    assert not env.solved()
    run(env.close())


def test_prep_python_code() -> None:
    from path4.tasks_random_crypto.env_hard import prep_python_code
    assert prep_python_code('-c "print(1)"') == "print(1)"
    assert prep_python_code("<< 'EOF'\nprint(1)\nEOF") == "print(1)"
    assert prep_python_code("'print(1)'") == "print(1)"
    assert prep_python_code("print(1)") == "print(1)"
    assert prep_python_code("-c \"x = 'a b'\nprint(x)\"") == "x = 'a b'\nprint(x)"


def test_c_style_and_heredoc_snippets_run() -> None:
    env = HardReplCTFEnv(_task())
    run(env.reset())
    obs = run(env.step('python3 -c "print(\'flag{secret1}\')"'))
    assert env.solved() and "flag{secret1}" in obs.output
    run(env.close())
