# path4/ensemble — Racing Alloy Ensemble (Path 4, step 3)

The inference layer of the flagship demo (**README §4 Path 4 step 3**): k
policies — frontier models plus the RL'd student — **race per challenge** on
isolated envs; **first verified flag wins**. Also supports the **per-turn
alloy** pattern (XBow): one agent loop, one transcript, each model call routed
to a different provider. *"Our RL'd model competing with frontier models on
the scoreboard"* is the money shot.

## Quickstart

```bash
# 1) offline demo — no keys, no network, uses MockChatClient + MockCTFEnv
python -m path4.ensemble.cli race \
    --tasks contracts/tasks/examples \
    --policies solo:mock-a,solo:mock-b \
    --mock --env mock --out /tmp/race_demo --max-steps 10
# -> rich table per task + per-episode transcripts + summary.json in /tmp/race_demo/<task_id>/

# 2) single-policy episode
python -m path4.ensemble.cli episode --tasks contracts/tasks/examples \
    --policy alloy:claude-sonnet-4:0.6,gemini-2.5-pro:0.4 --out runs/ep.jsonl

# 3) real run — LiteLLM proxy first:
litellm --config contracts/models.yaml
python -m path4.ensemble.cli race --tasks contracts/tasks/examples \
    --policies solo:gpt-5.4,alloy:claude-sonnet-4:0.6,gemini-2.5-pro:0.4,solo:student-qwen \
    --env daytona --findings-bus --out runs/race1
```

Client env vars: `OPENAI_BASE_URL` (default `http://127.0.0.1:4000/v1`),
`OPENAI_API_KEY`. `CTF_MOCK_LLM=1` makes the agent loop fall back to
`MockChatClient` when no client is passed.

## Policy grammar

```
solo:MODEL                      # one model every turn, e.g. solo:gpt-5.4
alloy:M1:w1,M2:w2               # per-turn weighted-random routing (XBow pattern)
```

Weights are optional (`alloy:a,b` = 50/50). Policies are routing rules only —
models never learn about each other; each thinks it wrote all prior assistant
turns. Seeded rng per episode → reproducible transcripts.

## Components

| File | What |
|---|---|
| `llm.py` | `ChatClient` (raw httpx → any OpenAI-compatible `/v1/chat/completions`; retries 429/5xx with backoff; returns the *served* model name when the provider echoes one) and `MockChatClient` (deterministic scripts keyed by model + call index; last entry repeats). |
| `policies.py` | `Policy` protocol, `SoloPolicy`, `AlloyPolicy`, `parse_policy`. |
| `agent.py` | `run_episode(task, policy, chat_client, env, ...)`: system prompt = CTF operator persona + task.prompt + rules (one command per turn in a fenced block; stop with `FLAG: flag{...}`); robust command extraction; `scan_for_flags` on observations; token accounting; per-message `model` in the transcript (§3.2). Caller must pass a **fresh env per episode**. |
| `racer.py` | `race(task, policies, env_factory, chat_client, ...)`: concurrent episodes via `asyncio.gather`; the first episode whose env reports `solved()` sets a shared `asyncio.Event` that cancels the others at their next turn. Writes one transcript JSONL per episode + `summary.json`. `FindingsBus` = minimal pub/sub for teammate hints. |
| `cli.py` | `race` and `episode` typer commands; `--env mock|repl|docker|daytona` maps to the `contracts` env backends (daytona/docker guarded imports). |

## Transcript layout

Canonical `contracts` §3.2 transcripts: `policy` = the policy string,
assistant messages carry the per-turn `model` that produced them, observations
are `tool`-role, `race_id` / `cancelled` / `wall_time` ride along as optional
extension fields. One JSONL line per episode; episode ids embed
`race_id:policy:task`.

## Anti-cheat notes (README §1.5)

- **Fresh sandbox per episode** — `race` builds a new env per policy via
  `env_factory(task)`; `run_episode` requires a caller-owned fresh env.
- **Flag verification outside the sandbox** — `env.solved()` runs the verifier
  in the backend, never inside the sandbox.
- **Egress lock** — enforced by `DaytonaCTFEnv` network limits (see
  `contracts/env/daytona.py`); the system prompt tells models not to assume
  egress.
- **Bus never carries flags** — `FindingsBus.publish` scrubs any `flag{...}`
  substring before delivery (test-enforced).

## Tests

```bash
python -m pytest path4/ensemble -q   # fully offline; asyncio.run in sync tests
```
