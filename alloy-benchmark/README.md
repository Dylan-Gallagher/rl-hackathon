# Alloy Agents — CTF benchmark

Tests XBOW's [Alloy Agents](https://xbow.com/blog/alloy-agents) claim: that
alternating between two different LLMs turn-by-turn inside a single agentic loop
solves more problems than either model alone.

## Design

Three conditions over the **same** stratified sample of 30 CTF-Dojo challenges,
`ATTEMPTS_PER_CHALLENGE` attempts each, 40-turn cap:

- **A-solo** — Claude (`claude-sonnet-5`) every turn
- **B-solo** — GLM (`glm-4.7`) every turn
- **alloy** — random per-turn choice between A and B (seeded, logged)

Only the model selection differs; the thread, tools, sandbox and verification
are identical across conditions.

### Models
`claude-sonnet-5` and `glm-4.7` — comparably capable, different lineages (a good
test for the overlap analysis). **Reasoning/thinking is disabled on both**, which
removes the cross-model thinking-signature problem entirely.

### The alloy mechanism (`conversation.py`)
The conversation lives ONCE in a provider-neutral canonical form. Each turn it is
rendered fresh into the target provider's wire format; the reply is parsed
straight back to neutral parts. The canonical form has no slot for provider name,
model name, response id, stop reason, thinking/signature, reasoning content or
usage, so none of those can cross a turn boundary. Tool-call ids are our own
synthetic ids; provider ids are discarded at the door. Every model sees the whole
thread as if it wrote every turn.

Verified by `verify_mechanism.py` (run it; it fails loud):
1. **Normalization equivalence** — a Claude-shaped reply and a GLM-shaped reply
   with identical content but different provider junk normalize to *byte-identical*
   canonical turns.
2. **Provenance scan** — no foreign model/provider name, provider-id pattern, or
   reasoning/signature field survives into either rendered payload.
3. **Live differential** — a real alternating conversation; every outgoing
   payload is scanned for the other provider's fingerprints; GLM is asserted to
   return zero reasoning content.

### Challenge substrate
Sampling frame = the 252 CTF-Dojo challenges that have both a stored
`.flag.sha256` (sha256 of the canonical `pwn.college{...}` flag) and a `flagCheck`
oracle. These are self-contained and locally solvable, giving deterministic
verification with no live-service dependency. The sample is stratified to match
the **full-pool** category proportions (crypto ~35%, pwn ~25%, rev ~19%,
misc ~13%, forensics ~6%, web ~3%).

Each attempt boots the `alloy-ctf-solver-v1` Daytona snapshot (Ubuntu 22.04 +
pwntools/pycryptodome/sympy/gmpy2/z3/... and standard CTF CLI tools), uploads the
challenge files to `/challenge`, and gives the agent two tools: `run_command` and
`submit_flag`. A flag is correct iff `sha256(submitted) == stored`.

## Files
- `config.py` — models, keys, parameters
- `conversation.py` — the alloy mechanism (canonical state + render/parse adapters)
- `llm_clients.py` — Claude + GLM callers, reasoning disabled
- `verify_mechanism.py` — mechanism leak tests (**must pass**)
- `build_snapshot.py` — one-time Daytona solver snapshot
- `prepare_challenges.py` — stratified sampling + sparse checkout + manifest
- `sandbox.py` — Daytona per-attempt wrapper + flag verification
- `agent.py` — the agentic loop (3 conditions) + trajectory logging
- `run_experiment.py` — concurrent, resumable full sweep
- `analyze.py` — solve rates + A/B overlap + alloy-vs-solo
- `project_cost.py` — cost projection from completed trajectories

## Run
```bash
python3 build_snapshot.py         # once
python3 prepare_challenges.py     # once -> challenges/manifest.json
python3 verify_mechanism.py       # must print PASSED
python3 agent.py <challenge_id> alloy   # one trajectory
python3 project_cost.py           # cost projection
python3 run_experiment.py --workers 10  # full sweep
python3 analyze.py                # results
```
