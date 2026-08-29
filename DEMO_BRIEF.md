# Path 4 demo brief — CTF alloy → traces → SFT student

Speaker notes + numbers for the live demo. Pair with `demo/presentation.html`.

**One-liner:** We mixed frontier models into an *alloy*, collected **1,200+ verified CTF solves**, and LoRA-trained an open **Qwen2.5-Coder-7B**. On a locked 20-task crypto eval, SFT **almost doubled Pass@4** (26% → 42%).

---

## 1. What to say (6–8 min)

1. **Problem (30s).** CTF is “a few good ideas among many dead ends.” One model’s first idea is often wrong. XBow-style **per-turn alloy** (different model each step, one transcript) plus a **racing ensemble** (first verified flag wins).
2. **System (45s).** Shared contracts: task JSON, transcript JSONL, `CTFEnv`. Hardened REPL (no `cat flag.txt` cheat). Flag check **outside** the sandbox.
3. **Teachers (45s).** Real APIs, this machine: GLM-4.6 + 4.5-air alloy, then GLM-5.3-flash (coding plan), Grok 4.6 / Grok Build (pi OAuth), Claude Sonnet 4.5 (Anthropic). Not Cursor — no API.
4. **Student (60s).** Qwen2.5-Coder-7B-Instruct, LoRA r=16, trained on vast.ai **RTX 5090**. v1 = 117 alloy traces in 65s. v2 = **698** success traces in ~5 min (in progress for re-eval).
5. **Numbers (90s).** Table below. Explain Pass@1 / Pass@4 / Maj@4 in one sentence each. Lead with Pass@4. Caveat Pass@1.
6. **Live (90s).** Scoreboard + trajectory: per-turn `model` labels, FLAG line. Optional: student vs teacher race if vLLM tunnel is up.
7. **Close (30s).** Alloy → traces → SFT is real. GRPO/veRL is wired (flag reward, DAPO filter, curriculum band) — next if we had GPU hours.

---

## 2. Headline results (locked eval, k=4, 20 crypto tasks)

Same 20 tasks for base and SFT. 79 episodes each (1 HTTP 400 dropped, both sides).

| Policy | Episodes | Solve rate | **Pass@1** | **Pass@4** | Maj@4 |
|---|---|---|---|---|---|
| 7B base (`Qwen2.5-Coder-7B-Instruct`) | 79 | 20.3% | **20.3%** | 26.3% | 21.1% |
| 7B SFT v1 (117 alloy traces, 65s) | 79 | 17.7% | 17.7% | **42.1%** | 10.5% |

**Read this out loud:** SFT is **−2.6 pt Pass@1**, **+15.8 pt Pass@4**. The student solves **more tasks if you sample a few times**, but is less consistent on the first try. Thin-data SFT: coverage up, mode not sharpened.

SFT v2 (698 traces, 88 steps, ~5 min, loss 0.97 → 0.32) was trained for the demo; re-eval may still be running — don’t quote v2 Pass@k unless `runs/eval_student/sft_v2` exists.

---

## 3. What Pass@k / Maj@k mean

- **Pass@1** — one attempt: did we get the flag?
- **Pass@4** — four independent attempts: did *at least one* get the flag? (unbiased combinatorial estimator over tasks with ≥4 episodes)
- **Maj@4** — majority of the four attempts solved (need 3/4). Punishes one-lucky-sample solvers.

We care about Pass@k because flags **self-verify**. Maj@k is the “is it reliable?” check.

---

## 4. Training data (harvest, still growing)

| Teacher | Solved / attempts | Notes |
|---|---|---|
| Claude Sonnet 4.5 | 703 / 827 | Anthropic Messages API, conc 16, 1000-task pool |
| Grok 4.6 | 178 / 181 | pi OAuth `xai` |
| Alloy GLM-4.6 + GLM-4.5-air (per-turn 50/50) | 126 / 178 | original SFT v1 data |
| GLM-5.3-flash | 115 / 116 | Z.ai **coding** plan (`zai-coding-cn`) |
| Grok Build 0.1 | 94 / 95 | pi OAuth, volume Grok |
| **Total** | **~1,216 / 1,397** | **88% solve rate** on attempted episodes |

Unique tasks with ≥1 solve: hundreds of Random-Crypto *easy* classical/hash/prng/rsa. **Eval 20 tasks never in train.**

**Dataset:** Random-Crypto 5,000 + 50 verified. We only auto-run stdlib-solvable easy rows (~1,400 of 5,000). 20 verified held out.

SFT v1 used **117** of the alloy solves. SFT v2 used **698** train records (735 after filters, 5% val).

---

## 5. Architecture (what we built)

```
Teachers (GLM / Grok / Claude)
    → agent loop (one shell command / turn, HardReplCTFEnv)
    → §3.2 JSONL transcripts (solved=true only)
    → build_dataset (assistant-only mask)
    → LoRA SFT on 7B (5090)
    → vLLM serves base + adapter
    → eval / race / scoreboard
```

**Per-turn alloy (XBow):** each assistant turn samples a teacher (`alloy:glm-4.6:0.5,glm-4.5-air:0.5`). One transcript; models don’t know they’re sharing.

**Racing ensemble:** k policies, isolated envs, first verified flag wins, others cancel. Findings bus scrubs `flag{...}`.

**Contracts (`contracts/`):** Task schema, transcript JSONL, `CTFEnv` (Mock / Repl / Docker / Daytona stub), flag gen/verify.

**Anti-cheat:** no `flag.txt` in the hard REPL; `submit flag{...}` or flag in Python stdout; verifier **outside** sandbox; fresh env per episode.

**Not finished (honest):** veRL multi-turn GRPO launcher is a skeleton (reward, DAPO filter, 5–40% curriculum, `CTFToolExecutor` are real). Daytona **GPU credits = 0**; training ran on vast.ai 5090 instead.

---

## 6. Models & compute

| Role | Model | Where |
|---|---|---|
| Student | Qwen2.5-Coder-7B-Instruct + LoRA r=16 α=32 | vast.ai RTX 5090 32GB |
| Teachers | glm-4.6, glm-4.5-air, glm-5.3-flash, grok-4.6, grok-build-0.1, claude-sonnet-4-5 | this machine, APIs |
| Serving | vLLM `--enable-lora` models `base` / `sft` | 5090, tunnel `127.0.0.1:8000` |

LoRA: ~40M trainable / 7.66B (v2). v1 adapter ~8.8M reported on 0.5B-path confusion — **v1/v2 adapters are 155MB on 7B**. Training: bf16, grad checkpoint, max_len 1536–2048, accum 16.

Blackwell quirk: `VLLM_USE_FLASHINFER_SAMPLER=0` required on sm_120.

---

## 7. Live demo checklist

```bash
# scoreboard over real eval + traces
python -m path4.scoreboard.cli serve --transcripts runs/eval_student --port 8099

# optional live race (if tunnel up)
curl -s http://127.0.0.1:8000/v1/models | head
```

- Browser: `http://localhost:8099` — click **solo:sft**, then an episode with `solved`.
- Show per-turn `model` field on an **alloy** transcript (`runs/traces/train/alloy.jsonl`).
- Slides: open `demo/presentation.html`.

If vLLM is down: slides + scoreboard + JSONL trajectories still demo. Don’t SSH live unless rehearsed.

---

## 8. Caveats (say them)

- SFT v1 data was **117 short crypto traces** — Pass@1 dip is expected.
- Eval is **crypto-only**, 20 tasks, k=4 — not NYU full bench.
- Teachers are **subscription APIs**; student is the open-weights artifact.
- GLM 5.x is on the **coding pack**, not the pay-as-you-go key.
- No GRPO numbers this round (compute: Daytona GPU wallet empty).

---

## 9. File map

| Path | What |
|---|---|
| `demo/presentation.html` | Slide deck |
| `runs/eval_student/base/base.jsonl` | Base 7B eval |
| `runs/eval_student/sft/sft.jsonl` | SFT v1 eval |
| `runs/traces/` | All teacher transcripts |
| `runs/sft/student_v2.jsonl` | 698-record SFT v2 dataset |
| `runs/student_sft_adapter/` | LoRA adapter backup |
| `path4/` | Ensemble, scoreboard, SFT, GRPO stubs |
| `contracts/` | Shared §3 schemas |
