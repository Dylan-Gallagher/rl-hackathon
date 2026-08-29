# DEMO PLAN — Path 4 (2h runway, v1 — numbers filled at T-20)

## Timeline (T = demo start)

| Clock | Milestone | Owner |
|---|---|---|
| T-110 | GPU worker + trace worker launched | done |
| T-65  | GPU sandbox + stack ready; ≥120 alloy traces + baseline evals done | G, H |
| T-65→T-35 | build_dataset → LoRA SFT on sandbox → vLLM serves base & SFT student | I |
| T-35→T-20 | student eval k=4 (base vs SFT) on locked 20-task set → transcripts → scoreboard | I |
| T-20→T-5 | assemble scoreboard from ALL real runs; rehearse live race; freeze | J |
| T-5   | Scoreboard serving on :8099, terminal ready, backup corpus ready | all |

## The demo (10-min slot)

1. **Story (60s)** — one diagram: `alloy traces → SFT student → (EI band → GRPO) → student races frontier models → live scoreboard`. "One open model, fine-tuned in this room today, racing frontier models."
2. **Live race (2–3 min)** — start a real race NOW on fresh Random-Crypto tasks:
   `student-sft vs glm-4.6 vs alloy(glm-4.6, glm-4.5-air)` — scoreboard auto-refreshes,
   episodes stream in, click one trajectory: per-turn model labels show the alloy switching models mid-solve.
3. **Numbers (2 min)** — scoreboard table (all REAL runs from today):
   - baselines: solo glm-4.6 / solo glm-4.5-air / alloy (Pass@1, Pass@4, per-category)
   - **the money row: student-SFT vs student-base on the same locked eval** — Pass@1/Pass@4 delta
   - race-win column: did the ensemble + student win races
4. **Trajectory contrast (2 min)** — same task: base student flails → SFT student solves
   (crisp python, FLAG: line, clean finish).
5. **Training story (1 min)** — `build_dataset` stats (traces in, records out), EI stats →
   curriculum 5–40% band picker output (real files), veRL GRPO config wired to the same
   `CTFEnv` + flag reward; fallback ladder named aloud.
6. **Close (30s)** — anti-cheat: fresh flags, verifier outside sandbox, egress lock; all §3 contracts shared with Paths 1–3.

## Fallbacks (decide point in parentheses)

- **No GPU by T-65** → student rows become "pipeline proof": SFT dry-run live + Path 1's
  HackSynth-GRPO reference curve (0.10→0.90 Pass@8) as the RL evidence; race = glm-4.6 vs alloy.
- **Trace yield < 60** → SFT anyway on what exists (report count honestly); baselines unaffected.
- **ZAI API dies live** → scoreboard already holds full results (warm state); live race optional.
- **vLLM serving trouble** → eval student via transformers generate (slower, pre-run before demo).

## Pre-demo checklist (J)

- [ ] scoreboard `serve --transcripts runs/demo_final --port 8099` up + auto-refresh verified
- [ ] live-race command tested once end-to-end at T-15 (then killed; restart at showtime)
- [ ] 2 showcase trajectories bookmarked (episode ids)
- [ ] backup: `runs/demo` seeded corpus ready to point at if anything breaks
- [ ] terminal font large; browser zoom 125%; no stray processes on :8099
