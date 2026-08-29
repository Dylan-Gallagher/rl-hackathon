"""Env end-to-end tests via asyncio.run (no pytest-asyncio)."""

import asyncio
import hashlib

import pytest

from contracts.env.daytona import DaytonaCTFEnv
from contracts.env.mock import MockCTFEnv
from contracts.env.repl import ReplCTFEnv
from contracts.task import Task


def test_daytona_import_guard():
    try:
        import daytona  # noqa: F401
        pytest.skip("daytona installed; guard not exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="rl-hackathon\\[daytona\\]"):
        DaytonaCTFEnv(Task(task_id="t", source="custom", category="misc"))


def test_mock_env_end_to_end():
    task = Task(task_id="mock-1", source="custom", category="misc",
                flag={"mode": "generated", "verify": "exact"})
    env = MockCTFEnv(task)

    async def run():
        obs = await env.reset(seed=42)
        assert "flag.txt" in obs.output and not obs.done
        obs = await env.step("ls")
        assert "flag.txt" in obs.output
        obs = await env.step("cat flag.txt")
        assert obs.output.startswith("flag{")
        assert env.solved()
        await env.step("bogus-cmd")
        await env.close()

    asyncio.run(run())


def test_mock_env_deterministic_under_seed():
    task = Task(task_id="mock-2", source="custom", category="misc",
                flag={"mode": "generated"})

    async def flag_for(seed):
        env = MockCTFEnv(task)
        obs = await env.reset(seed=seed)
        obs = await env.step("cat flag.txt")
        await env.close()
        return obs.output

    assert asyncio.run(flag_for(7)) == asyncio.run(flag_for(7))
    assert asyncio.run(flag_for(7)) != asyncio.run(flag_for(8))


def test_mock_env_static_flag():
    task = Task(task_id="mock-3", source="nyuctf", category="rev",
                flag={"mode": "static", "verify": "exact", "format": "flag{st4t1c}"})
    env = MockCTFEnv(task)

    async def run():
        await env.reset(seed=1)
        await env.step("cat flag.txt")
        assert env.solved()
        await env.close()

    asyncio.run(run())


def test_repl_env_solves_toy_crypto():
    digest = hashlib.md5(b"alloy-demo").hexdigest()
    task = Task(task_id="repl-1", source="random-crypto", category="crypto",
                env={"image": "", "launch": "repl"},
                flag={"mode": "static", "verify": "exact", "format": digest},
                prompt=f"The flag is the md5 hex digest of 'alloy-demo'.")
    env = ReplCTFEnv(task)

    async def run():
        obs = await env.reset(seed=42)
        assert "challenge.py" in obs.output
        obs = await env.step(
            "python import hashlib; print(hashlib.md5(b'alloy-demo').hexdigest())")
        assert digest in obs.output
        assert env.solved()
        obs = await env.step("ls")
        assert "flag.txt" in obs.output
        await env.close()

    asyncio.run(run())


def test_repl_env_error_and_timeout_paths():
    task = Task(task_id="repl-2", source="random-crypto", category="crypto",
                env={"launch": "repl"}, flag={"mode": "generated"})
    env = ReplCTFEnv(task, snippet_timeout_s=2)

    async def run():
        await env.reset(seed=0)
        obs = await env.step("python raise ValueError('boom')")
        assert "ValueError" in obs.output and "boom" in obs.output
        obs = await env.step("badcmd --x")
        assert "command not found" in obs.output
        obs = await env.step("python import time; time.sleep(60)")
        assert "TimeoutError" in obs.output
        await env.close()

    asyncio.run(run())


def test_repl_env_generated_flag_seeded():
    task = Task(task_id="repl-3", source="random-crypto", category="crypto",
                env={"launch": "repl"}, flag={"mode": "generated"})

    async def flags():
        out = []
        for _ in range(2):
            env = ReplCTFEnv(task)
            await env.reset(seed=11)
            o = await env.step("cat flag.txt")
            out.append(o.output.strip())
            await env.close()
        return out

    f1, f2 = asyncio.run(flags())
    assert f1 == f2 and f1.startswith("flag{")


def test_mock_env_same_seed_different_tasks_gives_different_flags():
    async def flag_for(task_id):
        task = Task(task_id=task_id, source="custom", category="misc",
                    flag={"mode": "generated"})
        env = MockCTFEnv(task)
        await env.reset(seed=42)
        obs = await env.step("cat flag.txt")
        await env.close()
        return obs.output

    assert asyncio.run(flag_for("task-a")) != asyncio.run(flag_for("task-b"))
    assert asyncio.run(flag_for("task-a")) == asyncio.run(flag_for("task-a"))
