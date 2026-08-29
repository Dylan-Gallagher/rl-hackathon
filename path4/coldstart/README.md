# path4/coldstart — Cold-start SFT over alloy success traces

Pipeline position: **alloy → traces → SFT (here) → EI (Path 3) → GRPO (`path4/verl_grpo`)**.
GRPO needs within-group reward variance; a weak student on broad CTF starves,
so cold-start SFT on alloy successes is effectively mandatory (README §1.3).

## What's here

| File | What |
|---|---|
| `build_dataset.py` | CLI: transcripts dir → filtered/cleaned/deduped SFT JSONL + val split + stats |
| `train_sft.py` | TRL SFTTrainer + PEFT LoRA wiring, assistant-only loss via our `mask` field; `--dry-run` mode with zero heavy deps |
| `configs/sft_lora.yaml` | TRL/LoRA config skeleton, every knob commented with hackathon rationale |
| `tests/` | pytest suite (fixture transcripts in `tests/fixtures/`) |

## Quickstart

```bash
# 1) dataset from Path 2's success traces (README §3.2 transcripts)
python -m path4.coldstart.build_dataset runs/episodes.jsonl -o path4/coldstart/data/sft_train.jsonl --val-frac 0.05

# 2) CI-friendly plumbing check (no torch/trl needed)
python -m path4.coldstart.train_sft --dataset path4/coldstart/data/sft_train.jsonl --dry-run

# 3) real training (needs [train] extras)
python -m path4.coldstart.train_sft --dataset path4/coldstart/data/sft_train.jsonl \
    --config path4/coldstart/configs/sft_lora.yaml --out runs/sft_coldstart
```

## Dataset record format

```json
{"task_id": "...", "episode_id": "...",
 "messages": [{"role": "assistant", "content": "..."}, {"role": "tool", "content": "..."}],
 "mask": [true, false]}
```

`mask[i]` is True only for assistant messages → only those tokens contribute
to the loss. Filters: solved-only, `steps <= max_steps` (40, §3.1 horizon),
provider-artifact strip, per-message char cap (`contracts.capped_output`),
dedup on `(task_id, sha256(messages))`.

## Mocked vs real

- **Real**: dataset building, cleaning, masking logic, config, collator label logic (pure-python core unit-tested with a fake tokenizer).
- **Mocked/stubbed**: `train()` requires torch/transformers/trl/peft/datasets (optional `[train]` extras — clean error if missing, never at import). `--dry-run` proves everything else.

## Env vars

None required. `--dry-run` works with the base install (`pydantic, pyyaml, typer, rich`).
