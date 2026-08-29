"""Shared contracts for the CTF Alloy Hackathon (README §3)."""

# contracts — shared schemas & env protocol

Every path builds against this package. Do not fork these schemas; propose
changes in the group channel (README §5).

## What's here (§3 mapping)

| File | README § | Contents |
|---|---|---|
| `task.py` | §3.1 | `Task`, `TaskEnv`, `TaskFlag`, `Horizon`, `Task.load/from_dict`, `load_tasks` |
| `transcript.py` | §3.2 | `Transcript`, `TranscriptMessage`, `write_transcript` (JSONL), `iter_transcripts`, `episode_id_for` |
| `flag.py` | §1.5, §3.3 | `new_flag`, `verify_flag`, `scan_for_flags` |
| `env/base.py` | §3.3 | `Obs`, abstract `CTFEnv`, `capped_output` |
| `env/mock.py` | §3.3 | `MockCTFEnv` — in-process fake shell (tests/demo) |
| `env/repl.py` | §3.3 | `ReplCTFEnv` — local python subprocess (random-crypto; no docker) |
| `env/docker.py` | §3.3 | `DockerCTFEnv` — local docker fallback |
| `env/daytona.py` | §3.3 | `DaytonaCTFEnv` — scale-out backend (`pip install 'rl-hackathon[daytona]'`) |
| `models.yaml` | §3.4 | LiteLLM proxy config (`alloy` = simple-shuffle group) |
| `tasks/examples/` | §3.1 | 3 example task JSONs (repl / static-exact / compose-regex) |
| `tests/` | — | pytest suite (`python -m pytest contracts -q`) |

## Usage

```python
from contracts import Task, MockCTFEnv, Transcript, TranscriptMessage, write_transcript, new_flag

task = Task.load("contracts/tasks/examples/nyuctf-rev-maze.json")
env = MockCTFEnv(task)
obs = await env.reset(seed=0)          # fresh flag injected per episode
obs = await env.step("cat flag.txt")   # capped output
if env.solved(): ...                   # verifier runs OUTSIDE the sandbox
await env.close()

t = Transcript(task_id=task.task_id, episode_id="solo-m:task:0", policy="solo:m",
               messages=[TranscriptMessage(turn=0, role="assistant",
                                           content="...", model="alloy")],
               solved=env.solved(), steps=n, flags_found=[...])
write_transcript(t, "runs/episodes.jsonl")   # one JSON line per episode
```

Model serving: `litellm --config contracts/models.yaml` → one OpenAI-compatible
endpoint; `model="alloy"` randomly routes per request (Sonnet/Gemini 2.5 Pro).
Policy strings are just names: `solo:MODEL`, `alloy:w1,w2`, `race:k`.

## Anti-cheat (§1.5) — where each rule lives

- **Fresh flag per episode**: `new_flag()` + each env's `reset()` injects a
  fresh `flag{uuid4}` only where the challenge needs it.
- **Verifier outside the sandbox**: `verify_flag` / `env.solved()` run in the
  orchestrator process, never inside agent-executed code. `script`-mode
  verification is likewise executed by the env backend outside the sandbox.
- **Egress lock**: lives in `DaytonaCTFEnv` (`egress_locked=True` → network
  limits). Mock/Repl/Docker envs provide NO egress lock — dev convenience only.
- **Fresh sandbox per rollout**: Daytona/Docker envs create a new sandbox on
  every `reset()`; auto-delete TTL ensures no leftovers.

## Notes

- Transcript extra fields (e.g. `category`, `race_id`) are allowed
  (`extra='allow'`) but are optional extensions — readers must not require them.
- `ReplCTFEnv` is NOT a security boundary; §1.5 rules still apply at infra level.
