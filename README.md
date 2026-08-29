# CTF Alloy Hackathon — Findings & 4 Parallel Paths

> **Goal:** Build an "alloy" of LLMs that solves CTF challenges, then use it to make a single open-weights model better at CTF (SFT + RL with flag-as-reward). Demo at the hackathon.
>
> **How to use this doc:** 4 paths, ordered simplest → coolest. Paths are designed to run **in parallel by separate agent groups**. Paths 1–2 are near-guaranteed backup results; Paths 3–4 are the win-the-hackathon material. Shared contracts (task schema, transcript format, env protocol) are in §3 — every group builds against them so work composes instead of colliding.

---

## 1. Research Findings (what we know works)

### 1.1 The "alloy" concept has three flavors; we use #3
1. **Weight merge** (mergekit / model soups) — merge model weights into one. Fragile, research-grade. Not us.
2. **Mixture-of-Agents (MoA)** ([arXiv 2406.04692](https://arxiv.org/abs/2406.04692)) — parallel proposers + aggregator per question. Single-turn, not agentic. Not us directly.
3. **Orchestration alloy** — different models combined inside an agentic loop. Two proven sub-patterns:
   - **Per-turn alloy (XBow style)** — [xbow.com/blog/alloy-agents](https://xbow.com/blog/alloy-agents): ONE agent loop, ONE transcript; each model call randomly routes to a different provider (e.g. Sonnet/Gemini, sometimes weighted 60/40). Models don't know about each other — each thinks it wrote all prior assistant turns. Same # of calls, compounded strengths. **Result: 25% → 40% → 55% solve rate on their vuln benchmark; alloy of Sonnet 4.0 + Gemini 2.5 Pro = 68.8% vs 57.5% best solo, and beat running two racing agents (57.2%).** Rules learned:
     - Alloy models from **different providers** (same-family alloys ≈ average, no boost).
     - Pick the pair with the **lowest Spearman correlation** of per-challenge solve rates (XBow best: ρ=0.46), among the strongest models.
     - Bias sampling toward the stronger model; cap iterations (~80) then restart fresh.
     - Works because CTF-like tasks = "a few great ideas among many dead ends" — exactly the structure where per-turn alloying shines.
   - **Racing alloy** — [verialabs/ctf-agent](https://github.com/verialabs/ctf-agent): coordinator LLM spawns a solver swarm per challenge; each swarm races Opus(med/max) + GPT-5.4 + GPT-5.4-mini + codex in parallel Docker sandboxes; first verified flag wins; partial findings shared via message bus. **Won 1st place at BSidesSF 2026 CTF, 52/52 challenges.** MIT license.

### 1.2 RL on CTF is proven (GRPO with flag-as-reward)
- **Random-Crypto benchmark** ([site](https://aielte-research.github.io/Random-Crypto/), [paper arXiv 2506.02048](https://arxiv.org/abs/2506.02048)): 5,000 procedurally generated crypto CTF challenges **designed for RL** (+50 human-verified eval set + generator for infinite fresh tasks).
- **HackSynth-GRPO** ([repo](https://github.com/aielte-research/HackSynth-GRPO)): reference implementation — Unsloth GRPO + function calling + agentic loop + sandboxed Python REPL. **Llama-3.1-8B-Instruct: Pass@8 0.10 → 0.90 after GRPO.** (Maj@8 only 0.02 → 0.14 → RL sharpens a mode; since flags self-verify, Pass@k is the metric that matters.)

### 1.3 SFT vs GRPO (strategy)
- GRPO needs **within-group reward variance**: all-fail groups → zero gradient. A weak student on broad CTF starves. → **Cold-start SFT first is effectively mandatory.**
- SFT on alloy traces: ceiling = alloy capability; suffers exposure bias (student drifts off teacher manifold on long horizons). GRPO trains on the student's own states → fixes exactly that.
- Evidence: "SFT memorizes, RL generalizes" (Chu et al. 2025); DeepSeek-R1 used cold-start SFT before RL.
- **Expert Iteration (ReST-EM / rejection-sampling SFT)** is the cheap middle: student attempts k=8–16 times → keep flag-verified solves → SFT on own successes → repeat. ~80% of RL's benefit, ~20% of the machinery. No PPO infra, just inference + SFT.
- **Recommended sequence: Alloy-SFT → EI → GRPO-on-the-learnable-band** (tasks with ~5–40% student pass rate).

### 1.4 Liftable assets (don't rebuild these)
| Asset | What | Use |
|---|---|---|
| [Random-Crypto](https://github.com/aielte-research/HackSynth-GRPO) | 5k RL challenges + generator | RL train pool |
| HackSynth-GRPO | Working GRPO-on-CTF trainer (Unsloth) | Reference implementation, Path 1 |
| [NYU CTF Bench](https://github.com/NYU-LLM-CTF/NYU_CTF_Bench) | 200 test + 55 dev challenges, 6 categories, dockerized, `pip install nyuctf` | Multi-category eval + dev train pool |
| [ctf-agent sandbox](https://github.com/verialabs/ctf-agent) | Dockerfile with radare2, gdb, pwntools, angr, SageMath, z3, volatility3, steghide… | Base toolchain image |
| [LiteLLM](https://docs.litellm.ai/docs/routing) | `simple-shuffle` router = random model per request behind one OpenAI-compatible endpoint | Per-turn alloy, config-only |
| [veRL](https://verl.readthedocs.io/en/latest/start/agentic_rl.html) / SkyRL-Agent | Async multi-turn agentic RL (rollouts = network calls) | Full GRPO, Path 4 |
| CTFusion ([arXiv 2605.11504](https://arxiv.org/abs/2605.11504)) | Live-CTF eval via MCP-on-CTFd; showed static CTF benchmarks are cheat-prone (agents web-search writeups) | Final arbiter + anti-cheat design |

### 1.5 Anti-cheat / reward-hacking (design it in from day 1)
- **No egress from sandboxes during training/eval** (kills writeup-search cheating, per CTFusion).
- **Fresh `flag{uuid4}` per episode**, injected only where the challenge needs it, verifier runs outside the sandbox.
- **Fresh sandbox per rollout** (no cross-rollout contamination).
- Rename task ids / strip challenge names from prompts (pretraining contamination).

### 1.6 Infrastructure: Daytona
Daytona sandboxes: sub-second builds from Docker images, **warm pools** (pre-provisioned sandboxes), snapshots, labels, auto-delete TTL, **network limits** (egress lock), fork/pause, Docker-in-Docker support, and GPU sandboxes (H100/H200, spot). Maps perfectly to: per-episode isolation, GRPO group fan-out, and vLLM student inference.

---

## 2. The four paths at a glance

| # | Name | One-liner | Risk | Hackathon value |
|---|---|---|---|---|
| 1 | **Proven GRPO** | Lift HackSynth-GRPO, train Llama-3.1-8B on Random-Crypto, reproduce 0.1→0.9 | Very low | Solid backup: "we RL'd a model on CTFs, here are the curves" |
| 2 | **Alloy Distillation** | XBow per-turn alloy (LiteLLM) generates traces → SFT a student | Low | "Our alloy beats every model in it, and we distilled it" |
| 3 | **Unified CTF Gym + Expert Iteration** | Task/Env abstraction over Daytona + NYU CTF Bench; student self-improves via EI | Medium | "Multi-category CTF gym; student teaches itself, per-category gains" |
| 4 | **Full Agentic GRPO + Alloy Ensemble** | veRL multi-turn GRPO on the gym + alloy/racing ensemble at inference + live demo | Higher | The winner: end-to-end system, one model RL'd, alloy out front, live scoreboard |

**Progression logic:** 1 and 2 are independently demo-able backups (finish day 1–2). 3 builds shared infra that 4 consumes. 4 is the flagship demo. Every path ships *something* even if it stalls.

---

## 3. Shared contracts (ALL groups build against these)

> These live in `rl_hackathon/contracts/`. Do not fork them per-path; propose changes in the group channel instead.

### 3.1 Task schema (`tasks/*.json`)
```json
{
  "task_id": "nyuctf-2021f-rev-maze",
  "source": "nyuctf | random-crypto | custom",
  "category": "pwn | rev | crypto | web | forensics | misc",
  "env": {"image": "ghcr.io/<org>/ctf-tasks/rev-maze:abc123", "launch": "supervisor | compose | repl | none"},
  "flag": {"mode": "generated | static", "verify": "regex | exact | script", "format": "flag{uuid4}"},
  "prompt": "challenge description text",
  "horizon": {"max_steps": 40, "timeout_s": 1800},
  "split": "train | eval"
}
```

### 3.2 Transcript format (one JSONL line per message; the universal artifact)
```json
{"task_id": "...", "episode_id": "...", "policy": "solo:MODEL | alloy:w1,w2 | race:k", "split": "train",
 "messages": [
   {"turn": 0, "role": "assistant", "content": "...", "model": "anthropic/claude-opus-4"},
   {"turn": 0, "role": "tool", "content": "...", "model": null}
 ],
 "solved": true, "steps": 23, "flags_found": ["flag{...}"], "sandbox_id": "daytona-...", "tokens_in": 0, "tokens_out": 0}
```
Everything downstream (SFT datasets, EI filtering, eval metrics, GRPO reward attribution) reads this format.

### 3.3 Env protocol (`env/base.py`)
```python
class CTFEnv:
    async def reset(self, seed=None) -> Obs   # fresh sandbox, inject flag, return initial state
    async def step(self, action: str) -> Obs  # exec command; capped output (head/tail policy)
    def solved(self) -> bool                  # flag verifier + regex scan of observations
    async def close(self)                     # destroy sandbox
```
Backends: `DockerEnv` (local) and `DaytonaEnv` (scale). Same protocol.

### 3.4 Model config (`models.yaml`)
One OpenAI-compatible endpoint for everything. Alloy = LiteLLM `simple-shuffle` model group; solo = direct model name. Policy objects only emit model *names*.

---

## 4. The Paths

---

## 🟢 PATH 1 — "Proven GRPO" (backup #1)

**Owner group: RL-Basics** · **Est: 1–2 days** · **Risk: very low** (someone already did it; we lift and reproduce)

**Goal:** Train an open-weights model with GRPO on Random-Crypto and show big capability gains. This is the guaranteed-result anchor of the hackathon.

**Lift inventory:**
- [HackSynth-GRPO](https://github.com/aielte-research/HackSynth-GRPO) (Unsloth GRPO + agentic loop + sandboxed REPL)
- Random-Crypto 5k train CSVs + 50-challenge verified eval set
- Target model: `meta-llama/Llama-3.1-8B-Instruct` (match the paper) and/or `Qwen/Qwen2.5-Coder-7B-Instruct` (fresh angle)

**Steps:**
1. Clone repo, both conda + venv envs (their GRPO env and vLLM env differ — documented in README).
2. Eval baseline: `eval_agent.py` on 50-challenge verified set, Pass@1/8 before training.
3. Run `train_agent.py` on the 5k set (start `--difficulties easy`, then add medium). Single H100 (Daytona GPU sandbox or local).
4. Re-eval checkpoint. Plot Pass@1/Pass@8 curves + a few trajectory diffs (before/after examples are great demo material).
5. Bonus if time: vary GRPO group size G (8 vs 16) and note sample-efficiency effect.

**Deliverables:** trained checkpoint (HF), eval table (before/after), training curves, 3 before/after trajectory examples.

**Success criteria:** ≥2× baseline Pass@8 on verified eval set (paper got 0.10→0.90; anything ≥0.3 is a good demo).

**Risks & mitigations:**
- Env/dependency hell (their repo is AGPL research code) → use their exact conda env; don't upgrade anything.
- Training instability on non-Llama models → fall back to Llama-3.1-8B which has precedent.
- GPU time budget → easy difficulty only still shows the effect.

---

## 🟡 PATH 2 — "Alloy Distillation" (backup #2, and the fun story)

**Owner group: Alloy** · **Est: 2 days** · **Risk: low** (components proven separately; our novelty is the combination + measurement)

**Goal:** Build the XBow-style per-turn alloy, prove it beats every constituent on our CTF set, then distill it into a student via SFT.

**Steps:**
1. **LiteLLM alloy endpoint** (config-only):
   ```yaml
   model_list:
     - model_name: alloy
       litellm_params: {model: anthropic/claude-sonnet-4, api_key: os.environ/ANTHROPIC_API_KEY}
     - model_name: alloy
       litellm_params: {model: google/gemini-2.5-pro, api_key: os.environ/GEMINI_API_KEY}
   router_settings: {routing_strategy: simple-shuffle}
   ```
   (Add a third deployment if we have keys; weights biased toward the stronger model once measured.)
2. **Episode runner** (uses Path 3's contracts; a minimal local fallback: Random-Crypto rows + sandboxed Python REPL, no docker needed). Terminal-command actions, text in/out — no native function-calling across providers (XBow pattern).
3. **Baseline table** (the key science artifact): on a fixed set (~100 Random-Crypto eval + easy NYU dev):
   - solo A, solo B, alloy 50/50, alloy 60/40, race(A,B) — Pass@1/4/8 each.
   - Compute Spearman ρ between per-challenge outcomes of A and B (XBow's model-selection method, done empirically).
4. **Trace generation:** run the alloy over the train pool, keep `solved==true` transcripts.
5. **SFT:** mask non-assistant turns → TRL SFTTrainer, LoRA on the student (Qwen2.5-Coder-7B or Path 1's model). Eval student on the same fixed set: vs. each solo model vs. alloy.

**Deliverables:** alloy proxy config + episode runner; baseline table + correlation analysis; 500–2k success traces; distilled student checkpoint + eval deltas.

**Success criteria:** (a) alloy > best solo on our set (XBow says ~+10pp); (b) student > its own base by a clear margin. Either alone is a demo; both is the story: *"we built a model alloy, measured it, and distilled it into one open model."*

**Risks & mitigations:**
- API cost/spend → cap iterations (~40 steps), downselect challenges, 2 models not 3.
- Cross-provider transcript quirks → strip provider-specific artifacts; one chat template.
- Alloy doesn't beat solo on our set → that's still a finding; pivot to race-alloy data-gen (transcripts are the same format either way).

---

## 🟠 PATH 3 — "Unified CTF Gym on Daytona + Expert Iteration" (shared infra + self-improving student)

**Owner group: Gym** · **Est: 2–3 days** · **Risk: medium** (most integration work; unblocks Path 4)

**Goal:** The `Task`/`CTFEnv` abstraction running on Daytona with NYU CTF Bench + Random-Crypto registered; a student model that measurably teaches itself via expert iteration.

**Steps:**
1. **Base toolchain image**: lift [ctf-agent's `sandbox/Dockerfile.sandbox`](https://github.com/verialabs/ctf-agent) → build → GHCR → Daytona snapshot.
2. **`DaytonaEnv`**: implement `CTFEnv` on the Daytona Python SDK — sandbox per episode (labels: `run_id/task_id/episode_id`), auto-delete TTL, **egress locked** via network limits, capped tool output. `DockerEnv` local fallback for dev.
3. **Converters**:
   - `nyuctf.py`: `CTFDataset` dev(55)/test(200) → task registry. Compose challenges via Daytona Docker-in-Docker snapshot (compat path); flatten high-value ones later.
   - `random_crypto.py`: CSV rows → `repl`-type tasks (no docker needed) + wire the generator for infinite fresh tasks.
   - `picoctf.py` (stretch): pull a year of archive challenges as `files`-type tasks.
4. **Flag injection & verification**: per-episode `flag{uuid4}` where mode allows; verifier outside sandbox; regex scan of observations for auto-solve detection.
5. **Eval harness**: any model/policy over any task split → per-category Pass@1/8, Maj@8; writes canonical transcripts.
6. **Expert Iteration loop** (the self-improvement demo):
   - vLLM serving the student on a Daytona GPU sandbox.
   - k=8 samples per task on the train pool → keep solves → SFT (LoRA) → repeat 2–3 rounds.
   - Track Pass@1/8 by round on the locked eval split. Difficulty bands (pass-rate buckets) fall out of this for free → hand to Path 4.

**Deliverables:** working gym (`tasks/`, `env/`, `converters/`), eval harness with baseline table across ≥2 frontier models + student, EI round-over-round gains chart, difficulty-band task lists.

**Success criteria:** (a) NYU dev split solvable end-to-end in Daytona sandboxes; (b) student Pass@8 improves ≥1.5× over base after EI rounds; (c) eval harness produces per-category breakdowns for the demo.

**Risks & mitigations:**
- NYU compose challenges flaky in DinD → start with `files`/`repl` tasks + flatten; keep a curated subset of ~30 compose challenges.
- Daytona API rate limits during group fan-out → warm pools + retry/backoff; concurrency cap.
- Sandbox boot latency → pre-warm pool sized to `group_size × parallel_tasks`.

---

## 🔴 PATH 4 — "Full Agentic GRPO + Alloy Ensemble" (the flagship)

**Owner group: Flagship** · **Est: 3+ days, overlaps Path 3's infra** · **Risk: higher** (novel integration; capped by fallbacks below)

**Goal:** The end-to-end system demo: **one open model RL'd with multi-turn GRPO inside the CTF gym (Daytona rollouts), bootstrapped by alloy traces, deployed inside a racing alloy ensemble — scored live.**

**Architecture:**
```
                ┌── Path 2: per-turn alloy (frontier APIs) ──> success traces ─┐
                │                                                             ▼
 student ── cold-start SFT ──> EI rounds (Path 3) ──> GRPO (veRL, multi-turn) ──> student'
                                                                                 │
 live demo: student' + alloy members racing per challenge (Daytona sandboxes) ───┘
 scoreboard: per-model & ensemble Pass@k, per-category, live trajectory viewer
```

**Steps:**
1. **Cold start** (needs Path 2 traces + Path 3 gym): SFT student on alloy successes (mask non-assistant turns).
2. **GRPO via veRL agentic RL** (needs Path 3 `CTFEnv` + EI difficulty bands):
   - Wire `DaytonaEnv` as veRL's async tool executor (rollout steps = network calls → 8–16 sandboxes in flight per group; veRL's server-based async rollout is built for this).
   - Reward = flag verifier (binary). Mask observation/tool tokens in the loss (veRL multi-turn supports this).
   - DAPO-style dynamic sampling: drop all-zero/all-one groups. Curriculum = Path 3's 5–40% pass-rate band. Start with crypto+web (short horizons), add pwn/rev last.
   - Note: Path 1's Unsloth result is the fallback — "GRPO on CTF" is already proven if veRL integration slips.
3. **Ensemble inference layer**: `race(k models + student)` on Daytona — first verified flag wins; optional shared-findings bus (ctf-agent pattern). Include the **student as a racing member** — "our RL'd model competing with frontier models on the scoreboard" is the money shot.
4. **Live demo / scoreboard**:
   - Self-hosted CTFd (or static pack) as the challenge board; agents poll and solve; live Pass@k + trajectory viewer streaming canonical transcripts.
   - Stretch: run the final eval through a CTFusion-style live-CTF setup for an uncontaminated number.

**Deliverables:** GRPO-trained student (multi-category), racing ensemble harness, live scoreboard UI, full story: *alloy → traces → SFT → EI → GRPO → ensemble*.

**Success criteria (any one is demo-worthy; aim for all):**
- GRPO student beats its own cold-start checkpoint on locked eval (Pass@1 and/or Pass@8).
- Ensemble (with student in it) ≥ best solo frontier model on the scoreboard.
- A clean live demo: watchers see challenges get solved in real time, models labeled.

**Risks & mitigations:**
- veRL agentic integration is the riskiest piece → **fallback ladder:** Unsloth single-turn GRPO (Path 1) → EI-only (Path 3) → alloy-only results (Path 2). The demo degrades gracefully.
- Long-horizon rollouts blow context/time → train on short-horizon categories first; cap steps at 40 for training vs 80 for eval.
- Reward hacking → per §1.5: egress lock, fresh flags, monitor "solves without plausible exploit step" on a canary set.
- GPU budget → LoRA everywhere; one H100-class node equivalent (Daytona GPU sandbox) is enough for 7–8B.

---

## 5. Parallelization map & dependencies

```
Day 0-1:   P1 ──────────────> result         (independent)
           P2: proxy+runner -> baselines     (independent; uses P3 contracts, local fallback OK)
           P3: image+DaytonaEnv+converters   (independent)
Day 1-2:   P2: traces -> SFT -> student      (needs P3 gym OR local fallback)
           P3: eval harness -> EI rounds     (independent)
Day 2-3+:  P4: cold-start (P2 traces + P3 gym) -> GRPO -> ensemble -> scoreboard
```

- **P1 never blocks anyone.** Run it first, bank the result.
- **P2 and P3 share only the contracts (§3).** P2 can start on Random-Crypto/local REPL without Daytona.
- **P4 consumes P2 + P3 outputs** but has explicit fallbacks at every stage, so a P4 stall never takes down the demo.
- Contracts changes are proposed in a shared channel; P3 owns the contract files (others file "PRs").

## 6. Resource checklist

- **API keys**: Anthropic, Google, OpenAI (≥2 providers required for a real alloy; 3 is better).
- **Daytona**: org + API key; build/push `ctf-tools` image; confirm snapshot + warm pool + network limits on our tier; GPU sandbox (H100, spot) for student inference / Unsloth GRPO.
- **GPU**: Path 1 & 2 SFT fit on 1×H100 (LoRA). Path 4 GRPO: 1 node (8×GPU ideal, 1×H100 + LoRA possible for 7–8B with reduced group size).
- **HF**: write access for checkpoints/datasets; student base: `Qwen/Qwen2.5-Coder-7B-Instruct` and/or `meta-llama/Llama-3.1-8B-Instruct`.
- **License note**: HackSynth-GRPO is AGPL — keep our wrapper code separate if we plan to open-source; check provider ToS for training on API outputs (SFT-on-API-traces is standard practice but flag it in the writeup).

## 7. Immediate next actions (per path)

| Path | First concrete task |
|---|---|
| 1 | Clone HackSynth-GRPO, reproduce their eval on the 50-challenge verified set, record baseline numbers |
| 2 | Stand up LiteLLM alloy endpoint + minimal episode runner on 20 Random-Crypto rows; verify per-turn model switching works (log `model` per turn in transcripts) |
| 3 | Build `ctf-tools` image from ctf-agent's Dockerfile, push to GHCR, create Daytona snapshot, spawn first sandbox, run `echo` + `radare2 -v` |
| 4 | Write the cold-start SFT config skeleton + veRL `DaytonaEnv` adapter stub against §3.3 while waiting on P2/P3 outputs; define the scoreboard data model (it's just transcripts + a viewer) |

---

*Sources: XBow alloy agents blog · verialabs/ctf-agent · aielte-research Random-Crypto + HackSynth-GRPO · NYU CTF Bench · veRL agentic RL docs · CTFusion (arXiv 2605.11504) · MoA (arXiv 2406.04692) · Chu et al. 2025 "SFT Memorizes, RL Generalizes" · Daytona docs.*
