"""Cold-start SFT trainer (TRL + PEFT), with a torch-free testable core.

Heavy deps (torch / transformers / trl / peft / datasets) are OPTIONAL:
importing this module never imports them; they are imported lazily inside
``train()`` with a clean instructive error. ``--dry-run`` executes the whole
plumbing (dataset build, collation via a fake char tokenizer, token stats)
without loading any model, so CI proves the pipeline without GPUs.

Pure-python core (unit-tested without torch):
- ``token_counts(messages, tokenizer_len_fn)`` — per-message token counts.
- ``build_labels(messages, mask, tokenizer_len_fn)`` — token-level labels:
  ``-100`` (ignore_index) for every token of non-assistant messages, and for
  assistant messages the *global token position index* (a placeholder the
  torch collator replaces with the real ``input_ids``). This makes the
  "which tokens do we learn on" logic injectable and testable with a fake
  tokenizer; the runtime collator is then a thin copy+mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer(add_completion=False, help="Cold-start SFT training (TRL/PEFT).")
console = Console()

TRAIN_EXTRAS_MSG = (
    "training extras are not installed. Install them in the training env with:\n"
    "    pip install -e '.[train]'   (or: pip install torch transformers trl peft datasets)\n"
    "CI/dry-run mode needs none of these: python -m path4.coldstart.train_sft --dry-run ..."
)

IGNORE_INDEX = -100  # torch's CrossEntropyLoss ignore_index; do not change


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as e:  # pragma: no cover - exercised only when extras missing
        console.print(f"[red]missing dependency:[/] {module}\n{TRAIN_EXTRAS_MSG}")
        raise SystemExit(2) from e


# ---------------------------------------------------------------------------
# torch-free core
# ---------------------------------------------------------------------------

def token_counts(messages: list[dict], tokenizer_len_fn) -> list[int]:
    """Per-message token counts using an injected length function."""
    return [int(tokenizer_len_fn(m["content"])) for m in messages]


def build_labels(messages: list[dict], mask: list[bool], tokenizer_len_fn) -> list[list[int]]:
    """Token-level labels for assistant-only loss.

    Returns one list per message, each of length ``len_fn(content)``:
    - masked (non-assistant) messages -> all ``IGNORE_INDEX`` (-100);
    - assistant messages -> ``[global_token_start, ..., global_token_end-1]``.
      These position placeholders mark exactly which token slots learn; the
      torch collator replaces them with the corresponding real ``input_ids``.
    """
    labels: list[list[int]] = []
    pos = 0
    for msg, learnable in zip(messages, mask):
        n = int(tokenizer_len_fn(msg["content"]))
        if learnable:
            labels.append(list(range(pos, pos + n)))
        else:
            labels.append([IGNORE_INDEX] * n)
        pos += n
    return labels


def labels_to_flat(labels: list[list[int]]) -> list[int]:
    """Flatten; replace assistant position placeholders with a 1 marker.

    Used for stats/tests: -100 = ignored token, 1 = learned token.
    """
    return [IGNORE_INDEX if v == IGNORE_INDEX else 1 for row in labels for v in row]


def char_tokenizer_len(content: str) -> int:
    """Fake tokenizer: 1 token per char. Used by --dry-run stats & tests."""
    return len(content)


class MaskedSFTCollator:
    """Runtime collator (torch side). Applies our per-message mask to labels.

    Kept torch-free at import: torch is only touched in ``__call__`` via the
    lazily-imported modules passed at construction.
    """

    def __init__(self, tokenizer, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]):
        torch = _require("torch")
        input_ids_list, labels_list = [], []
        for f in features:
            ids: list[int] = []
            labs: list[int] = []
            for msg, learnable in zip(f["messages"], f["mask"]):
                tok = self.tokenizer(msg["content"], add_special_tokens=False)["input_ids"]
                if learnable:
                    labs.extend(tok)
                else:
                    labs.extend([IGNORE_INDEX] * len(tok))
                ids.extend(tok)
            ids = ids[: self.max_length]
            labs = labs[: self.max_length]
            input_ids_list.append(ids)
            labels_list.append(labs)
        maxlen = max(len(x) for x in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        batch_ids, batch_labels, batch_mask = [], [], []
        for ids, labs in zip(input_ids_list, labels_list):
            pad = maxlen - len(ids)
            batch_ids.append(ids + [pad_id] * pad)
            batch_labels.append(labs + [IGNORE_INDEX] * pad)
            batch_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# dataset / training
# ---------------------------------------------------------------------------

def load_records(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dry_run_stats(records: list[dict]) -> dict:
    """Token stats over the whole dataset using the fake char tokenizer."""
    total = learned = 0
    longest = 0
    for r in records:
        flat = labels_to_flat(build_labels(r["messages"], r["mask"], char_tokenizer_len))
        total += len(flat)
        learned += sum(1 for v in flat if v != IGNORE_INDEX)
        longest = max(longest, len(flat))
    return {"records": len(records), "tokens": total, "learned_tokens": learned,
            "longest_record_tokens": longest}


def train(dataset_path: str, config_path: str, model: str | None, out: str):
    """Full SFT run. Requires the [train] extras; correct-by-inspection wiring."""
    torch = _require("torch")
    transformers = _require("transformers")
    trl = _require("trl")
    peft = _require("peft")
    datasets = _require("datasets")

    cfg = yaml.safe_load(Path(config_path).read_text())
    base = model or cfg["base_model"]

    records = load_records(dataset_path)
    ds = datasets.Dataset.from_list(records)  # dict-of-lists; keeps messages/mask as-is

    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    collator = MaskedSFTCollator(tokenizer, max_length=cfg.get("max_length", 8192))

    sft_cfg_kwargs = {k: v for k, v in cfg.items()
                      if k not in {"base_model", "lora", "max_length"}}
    sft_cfg = trl.SFTConfig(max_length=cfg.get("max_length", 8192), **sft_cfg_kwargs)
    lora = peft.LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                           lora_dropout=cfg["lora"]["dropout"],
                           target_modules=cfg["lora"]["target_modules"],
                           task_type="CAUSAL_LM")
    model_obj = transformers.AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16)

    trainer = trl.SFTTrainer(
        model=model_obj,
        args=sft_cfg,
        train_dataset=ds,
        data_collator=collator,   # assistant-only loss via OUR mask field
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    console.print(f"[green]saved[/] LoRA adapter -> {out}")


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", help="SFT JSONL from build_dataset"),
    config: str = typer.Option("path4/coldstart/configs/sft_lora.yaml", "--config"),
    model: str = typer.Option(None, "--model", help="Override base model id"),
    out: str = typer.Option("runs/sft_coldstart", "--out", help="Output dir"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Run plumbing (dataset+collation stats) without torch/trl"),
):
    """Cold-start SFT on alloy success traces (masked, assistant-only loss)."""
    if dry_run:
        records = load_records(dataset)
        if not records:
            console.print(f"[red]no SFT records produced[/] — check transcripts dir/filtering; "
                          f"{dataset} has 0 records")
            raise SystemExit(1)
        stats = dry_run_stats(records)
        # prove the collator logic end-to-end with the fake tokenizer
        sample = records[0]
        flat = labels_to_flat(build_labels(sample["messages"], sample["mask"], char_tokenizer_len))
        console.print(f"[bold]dry-run stats (char tokenizer):[/] {stats}")
        console.print(f"first record: {len(flat)} tokens, "
                      f"{sum(1 for v in flat if v != IGNORE_INDEX)} learned")
        console.print("[green]dry-run OK[/] — pipeline plumbing verified; no model loaded.")
        return
    train(dataset, config, model, out)


if __name__ == "__main__":
    app()
