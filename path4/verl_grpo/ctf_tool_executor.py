"""CTFToolExecutor — veRL-shaped async tool-executor adapter over CTFEnv.

Maps our §3.3 env protocol onto veRL's agentic tool-calling hook so Path 4's
GRPO rollouts (README Path 4 step 2: "Wire DaytonaEnv as veRL's async tool
executor") can step sandboxes as async network calls.

Mapping to veRL's API (see https://verl.readthedocs.io/en/latest/start/agentic_rl.html):
- veRL's ``ToolExecutor.execute(action, **kwargs)`` (async) -> :meth:`CTFToolExecutor.execute`.
- veRL passes an opaque per-request ``state`` dict; we keep the env, step
  count, and accumulated transcript under well-known keys (see ``STATE_*``),
  so veRL's generic tool loop can hold the state for us.
- veRL's ``AsyncToolCallingLLM`` invokes the executor per generated tool call;
  ``done=True`` ends the rollout turn-sequence, and the final reward is fed
  to the reward manager — we return ``reward`` alongside so the trainer can
  use flag_reward directly (binary, README §1.2).

The actual ``import verl`` happens ONLY inside :func:`make_verl_tool_executor`,
with a clean error; this module itself stays dependency-light.
"""

from __future__ import annotations

from typing import Any, Callable

from contracts import CTFEnv, MockCTFEnv, capped_output
from contracts.task import Task

from path4.verl_grpo.reward import flag_reward

# Well-known keys in the opaque state dict veRL hands back to us.
STATE_ENV = "ctf_env"          # the CTFEnv instance (per rollout)
STATE_STEPS = "ctf_steps"      # actions taken this episode
STATE_SOLVED = "ctf_solved"
STATE_HISTORY = "ctf_history"  # [(action, output), ...]

EnvFactory = Callable[[Task], CTFEnv]


def default_env_factory(task: Task) -> CTFEnv:
    """MockCTFEnv by default; DaytonaCTFEnv when DAYTONA env vars are set."""
    import os
    if os.environ.get("DAYTONA_API_KEY"):
        from contracts import DaytonaCTFEnv
        return DaytonaCTFEnv(task)
    return MockCTFEnv(task)


class CTFToolExecutor:
    """Async tool executor over a CTFEnv factory (veRL hook shape).

    State is carried in the caller's dict (``state``), mirroring how veRL's
    tool-calling loop threads per-request context. Output is capped
    (head/tail) to keep long tool dumps out of the 8k training context.
    """

    def __init__(self, env_factory: EnvFactory = default_env_factory,
                 max_output_chars: int = 4000):
        self.env_factory = env_factory
        self.max_output_chars = max_output_chars

    async def reset(self, state: dict, task: Task, seed: int | None = None) -> dict[str, Any]:
        """Fresh sandbox per rollout (README §1.5); returns the first obs."""
        env = self.env_factory(task)
        state[STATE_ENV] = env
        state[STATE_STEPS] = 0
        state[STATE_SOLVED] = False
        state[STATE_HISTORY] = []
        obs = await env.reset(seed=seed)
        return {"output": capped_output(obs.output, self.max_output_chars),
                "done": False, "reward": 0.0}

    async def execute(self, action: str, state: dict) -> dict[str, Any]:
        """veRL ToolExecutor.execute shape: ``{output, done, reward}``.

        reward is binary (flag_reward) and only nonzero when the env's
        out-of-sandbox verifier reports a solve on this step.
        """
        env: CTFEnv = state[STATE_ENV]
        obs = await env.step(action)
        state[STATE_STEPS] = state.get(STATE_STEPS, 0) + 1
        state[STATE_HISTORY].append((action, obs.output))
        solved = env.solved()
        state[STATE_SOLVED] = solved
        return {"output": capped_output(obs.output, self.max_output_chars),
                "done": obs.done or solved,
                "reward": flag_reward(solved)}

    async def close(self, state: dict) -> None:
        env: CTFEnv | None = state.pop(STATE_ENV, None)
        if env is not None:
            await env.close()


def make_verl_tool_executor(**executor_kwargs) -> "CTFToolExecutor":  # pragma: no cover
    """Wrap CTFToolExecutor for a verl runtime; requires verl installed.

    Called from inside a veRL training script (agent loop config points its
    tool manager at ``path4.verl_grpo.ctf_tool_executor``). Kept separate so
    importing THIS module never requires veRL.
    """
    try:
        import verl  # noqa: F401  (presence check; adapter API used by caller)
    except ImportError as e:
        raise ImportError(
            "veRL is not installed. It is only needed for the actual GRPO run:\n"
            "    pip install verl   (in the training environment; see run_grpo.sh)\n"
            "Everything else (tests, dry-run, mock rollouts) works without it."
        ) from e
    return CTFToolExecutor(**executor_kwargs)


async def run_episode(policy_fn: Callable, task: Task, executor: CTFToolExecutor | None = None,
                      max_steps: int = 40, seed: int | None = None) -> dict[str, Any]:
    """Full generate→step→reward loop with a pluggable policy.

    ``policy_fn``: async ``(messages) -> action``. This is exactly the loop
    veRL's async rollout runs for us at training time; this function proves
    the contract end-to-end (used by tests with a scripted policy and by the
    demo without any RL infra).
    """
    ex = executor or CTFToolExecutor()
    state: dict[str, Any] = {}
    messages: list[dict] = []
    obs = await ex.reset(state, task, seed=seed)
    messages.append({"role": "tool", "content": obs["output"]})
    total_reward = 0.0
    for _ in range(max_steps):
        action = await policy_fn(messages)
        messages.append({"role": "assistant", "content": action})
        result = await ex.execute(action, state)
        messages.append({"role": "tool", "content": result["output"]})
        total_reward = max(total_reward, result["reward"])
        if result["done"]:
            break
    await ex.close(state)
    return {"solved": bool(state.get(STATE_SOLVED)), "reward": total_reward,
            "steps": state.get(STATE_STEPS, 0), "messages": messages}
