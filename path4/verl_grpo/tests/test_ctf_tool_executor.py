from __future__ import annotations

import asyncio
import json

import pytest
from contracts import MockCTFEnv, Task

from path4.verl_grpo.ctf_tool_executor import (
    CTFToolExecutor,
    default_env_factory,
    make_verl_tool_executor,
    run_episode,
)


def _task() -> Task:
    return Task(task_id="mock-crypto-1", source="custom", category="crypto",
                prompt="Solve the mock CTF.", horizon={"max_steps": 40})


def test_default_env_factory_mock(monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    assert isinstance(default_env_factory(_task()), MockCTFEnv)


async def _solving_policy(messages) -> str:
    """Scripted policy: read the flag from the tool output, then submit it."""
    last = messages[-1]["content"]
    if "flag{" in last:
        flag = last[last.index("flag{"):].split()[0]
        return f"submit {flag}"
    return "cat flag.txt"


def test_run_episode_solves_mock_end_to_end():
    result = asyncio.run(run_episode(_solving_policy, _task(), max_steps=40, seed=0))
    assert result["solved"] is True
    assert result["reward"] == 1.0
    assert result["steps"] == 1  # 'cat flag.txt' output scanned -> auto-solve (no submit needed)


def test_run_episode_failing_policy_binary_reward():
    async def bad_policy(messages):
        return "ls"

    result = asyncio.run(run_episode(bad_policy, _task(), max_steps=3, seed=0))
    assert result["solved"] is False and result["reward"] == 0.0
    assert result["steps"] == 3


def test_executor_execute_shape_and_capping():
    ex = CTFToolExecutor()
    state = {}
    obs = asyncio.run(ex.reset(state, _task(), seed=0))
    assert set(obs) == {"output", "done", "reward"} and obs["reward"] == 0.0
    # huge tool output gets capped before entering the rollout context
    big = "A" * 10000
    state["ctf_env"]._flag = big  # mock will print the "flag" back to us
    res = asyncio.run(ex.execute("cat flag.txt", state))
    assert len(res["output"]) < 6000 and "[truncated" in res["output"]
    asyncio.run(ex.close(state))
    assert "ctf_env" not in state


def test_executor_binary_reward_on_solve():
    ex = CTFToolExecutor()
    state = {}
    asyncio.run(ex.reset(state, _task(), seed=0))
    env = state["ctf_env"]
    flag = env._flag
    res = asyncio.run(ex.execute(f"submit {flag}", state))
    assert res["reward"] == 1.0 and res["done"] is True
    asyncio.run(ex.close(state))


def test_verl_wrapper_guarded():
    try:
        import verl  # noqa: F401
        pytest.skip("verl installed; guard path not reachable")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="pip install verl"):
        make_verl_tool_executor()


def test_no_verl_at_import():
    import sys
    assert "verl" not in sys.modules
    import importlib
    importlib.import_module("path4.verl_grpo.ctf_tool_executor")
    assert "verl" not in sys.modules
