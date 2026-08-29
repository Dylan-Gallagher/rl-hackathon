"""Agent loop for the Path 4 ensemble layer.

ONE agent loop, ONE transcript (§3.2); each model call routes per-turn via a
Policy (XBow pattern — models never learn about each other). The caller owns
env lifecycle: pass a FRESH env instance per episode (fresh sandbox per
rollout, README §1.5); this function only resets/uses it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from typing import Any, Awaitable, Callable, Protocol

from contracts.env.base import CTFEnv, capped_output
from contracts.flag import DEFAULT_FLAG_PATTERN, scan_for_flags
from contracts.task import Task
from contracts.transcript import Transcript, TranscriptMessage, episode_id_for

from path4.ensemble.llm import MockChatClient
from path4.ensemble.policies import Policy

logger = logging.getLogger(__name__)

#: Observed outputs are capped before going back into the model context.
OBS_CONTEXT_CAP = 4000

SYSTEM_TEMPLATE = """\
You are an elite CTF operator working in a sandboxed shell. Solve the challenge.

CHALLENGE:
{prompt}

RULES:
- Reply with exactly ONE shell command per turn, wrapped in a fenced block:
  ```bash
  <command>
  ```
- The command runs in the challenge sandbox; you then see its output.
- Do not assume network egress is available.
- If you want to share a non-flag hint with racing teammates, add a line
  starting with `FINDING:` followed by a short note (no flags, ever).
- When you have the flag, stop issuing commands and print a final line:
  FLAG: flag{{...}}\
"""

_FINDING_RE = re.compile(r"^FINDING:\s*(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


def extract_command(content: str) -> tuple[str | None, str]:
    """Robustly extract the shell command from a model reply.

    Returns ``(command, nudge)`` — exactly one is non-empty:
    - first fenced block wins (strip ``$`` prompts / leading commentary);
    - no fence: accept the whole content only if it is a single line;
    - otherwise return a nudge message asking for a fenced block.
    """
    m = _FENCE_RE.search(content)
    if m:
        cmd = m.group(1).strip()
        # strip common shell-prompt prefixes inside the block
        cmd = re.sub(r"^\$\s+", "", cmd, flags=re.MULTILINE).strip()
        if cmd:
            return cmd, ""
    stripped = content.strip()
    if stripped and "\n" not in stripped and "```" not in stripped:
        return stripped, ""
    return None, (
        "Your reply did not contain a runnable command. Respond with exactly ONE "
        "shell command wrapped in a fenced code block:\n```bash\n<command>\n```"
    )


def extract_findings(content: str) -> list[str]:
    """Pull ``FINDING:`` notes out of an assistant reply (hints only, no flags)."""
    return [n.strip() for n in _FINDING_RE.findall(content) if n.strip()]


class ChatLike(Protocol):
    async def chat(self, messages: list[dict[str, str]], model: str | None = None) -> dict[str, Any]:
        ...


def default_chat_client() -> ChatLike:
    """Pick the chat client when none is passed: CTF_MOCK_LLM=1 -> MockChatClient."""
    if os.environ.get("CTF_MOCK_LLM") == "1":
        return MockChatClient()
    from path4.ensemble.llm import ChatClient

    return ChatClient()


async def run_episode(
    task: Task,
    policy: Policy,
    chat_client: ChatLike | None,
    env: CTFEnv,
    max_steps: int = 40,
    race_id: str | None = None,
    on_finding: Callable[[str, str], Awaitable[None]] | None = None,
    findings_inbox: "Inbox | None" = None,
    stop_event: asyncio.Event | None = None,
    timeout_s: float | None = None,
    seed: int | None = None,
) -> Transcript:
    """Run one episode; return the canonical Transcript (§3.2).

    ``env`` must be a fresh instance owned by this episode (caller's duty).
    ``on_finding(policy_name, note)`` fires for each ``FINDING:`` note the
    model posts; ``findings_inbox`` (see racer.FindingsBus) is drained each
    turn and delivered as user-role 'shared finding from teammate'.
    ``stop_event`` lets a racing winner cancel the remaining episodes.
    """
    client = chat_client or default_chat_client()
    rng = random.Random(seed if seed is not None else (race_id, policy.name(), task.task_id).__hash__())
    timeout_s = timeout_s if timeout_s is not None else task.horizon.timeout_s
    started = time.monotonic()

    episode_id = episode_id_for(f"{race_id + ':' if race_id else ''}{policy.name()}", task.task_id, 0)
    transcript = Transcript(
        task_id=task.task_id,
        episode_id=episode_id,
        policy=policy.name(),
        split=task.split,
    )

    init = await env.reset(seed=seed)
    transcript.sandbox_id = str((init.metadata or {}).get("sandbox_id", "")) or None
    transcript.messages.append(
        TranscriptMessage(turn=0, role="tool", content=capped_output(init.output, OBS_CONTEXT_CAP))
    )
    llm_messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(prompt=task.prompt or task.task_id)},
        {"role": "user", "content": f"Initial sandbox state:\n{capped_output(init.output, OBS_CONTEXT_CAP)}"},
    ]

    cancelled = False
    for turn in range(max_steps):
        if stop_event is not None and stop_event.is_set():
            cancelled = True
            break
        if time.monotonic() - started > timeout_s:
            cancelled = True
            break
        if findings_inbox is not None:
            for note in findings_inbox.drain():
                llm_messages.append(
                    {"role": "user", "content": f"shared finding from teammate: {note}"}
                )

        model = await policy.pick_model(turn, rng)
        reply = await client.chat(llm_messages, model=model)
        transcript.tokens_in += int(reply.get("tokens_in", 0))
        transcript.tokens_out += int(reply.get("tokens_out", 0))
        content = reply.get("content", "")
        served = reply.get("model") or model
        transcript.messages.append(
            TranscriptMessage(turn=turn, role="assistant", content=content, model=served)
        )
        llm_messages.append({"role": "assistant", "content": content})

        for note in extract_findings(content):
            if on_finding is not None:
                await on_finding(policy.name(), note)

        command, nudge = extract_command(content)
        if command is None:
            llm_messages.append({"role": "user", "content": nudge})
            continue

        obs = await env.step(command)
        transcript.steps += 1
        capped = capped_output(obs.output, OBS_CONTEXT_CAP)
        transcript.messages.append(TranscriptMessage(turn=turn, role="tool", content=capped))
        llm_messages.append({"role": "user", "content": f"$ {command}\n{capped}"})

        flags = scan_for_flags(obs.output) if task.flag.mode == "generated" else []
        for f in flags:
            if f not in transcript.flags_found:
                transcript.flags_found.append(f)
        if env.solved():
            transcript.solved = True
            break

    if env.solved():
        transcript.solved = True
    transcript.race_id = race_id
    transcript.cancelled = cancelled
    transcript.wall_time_ext = round(time.monotonic() - started, 3)
    return transcript
