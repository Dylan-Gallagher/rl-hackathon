"""CLI for the Path 4 ensemble layer: race policies on tasks, or run one episode.

Offline demo (no keys, no network):
    python -m path4.ensemble.cli race \
        --tasks contracts/tasks/examples --policies solo:mock-a,solo:mock-b \
        --mock --env mock --out /tmp/race_demo --max-steps 10

Real run: start LiteLLM (``litellm --config contracts/models.yaml``), then
    python -m path4.ensemble.cli race --tasks contracts/tasks/examples \
        --policies solo:gpt-5.4,alloy:claude-sonnet-4:0.6,gemini-2.5-pro:0.4,solo:student-qwen \
        --out runs/race1 --env daytona --findings-bus
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from contracts.task import Task, load_tasks
from contracts.transcript import write_transcript

from path4.ensemble.agent import run_episode
from path4.ensemble.llm import ChatClient, MockChatClient
from path4.ensemble.policies import parse_policy
from path4.ensemble import racer as racer_mod
from path4.ensemble.racer import make_env_factory
from path4.ensemble.agent import run_episode

app = typer.Typer(help="Path 4 ensemble: racing policies + per-turn alloy routing.")
console = Console()

MOCK_SCRIPTS = {
    # fast solver
    "mock-a": [
        "```bash\nls\n```",
        "```bash\ncat flag.txt\n```",
        "FLAG: got it",
    ],
    # slow solver: wanders before reading the flag
    "mock-b": [
        "```bash\nls -la\n```",
        "```bash\ncat README\n```",
        "```bash\nls\n```",
        "```bash\ncat flag.txt\n```",
        "FLAG: got it",
    ],
}


def _build_client(mock: bool) -> object:
    if mock:
        return MockChatClient(default=MOCK_SCRIPTS["mock-a"], scripts=dict(MOCK_SCRIPTS))
    return ChatClient()


def _parse_policies(spec: str) -> list:
    policies = [parse_policy(p.strip()) for p in spec.split(",") if p.strip()]
    if not policies:
        raise typer.BadParameter("no policies given")
    return policies


@app.command()
def race(
    tasks: Path = typer.Option(..., "--tasks", help="Directory of task *.json files"),
    policies: str = typer.Option(
        ..., "--policies",
        help="Comma-separated policy strings: solo:MODEL or alloy:M1:0.6,M2:0.4",
    ),
    out: Path = typer.Option(Path("runs/race"), "--out", help="Output directory"),
    mock: bool = typer.Option(False, "--mock", help="Use MockChatClient (offline)"),
    env: str = typer.Option("mock", "--env", help="mock | repl | docker | daytona"),
    max_steps: int = typer.Option(40, "--max-steps"),
    findings_bus: bool = typer.Option(False, "--findings-bus", help="Share non-flag hints"),
) -> None:
    """Race k policies per task; first verified flag wins."""
    task_list = load_tasks(tasks)
    if not task_list:
        raise typer.BadParameter(f"no tasks found in {tasks}")
    pols = _parse_policies(policies)
    client = _build_client(mock)
    env_factory = make_env_factory(env)

    for task in task_list:
        console.print(f"[bold]race[/bold] task=[cyan]{task.task_id}[/cyan] policies={[p.name() for p in pols]}")
        result = asyncio.run(
            racer_mod.race(
                task,
                pols,
                env_factory,
                client,
                max_steps=max_steps,
                findings_bus=findings_bus,
                out_dir=Path(out) / task.task_id,
            )
        )
        table = Table(title=f"{task.task_id} — race {result.race_id}")
        table.add_column("policy")
        table.add_column("solved", justify="center")
        table.add_column("steps", justify="right")
        table.add_column("cancelled", justify="center")
        for t in result.episodes:
            mark = "[green]✔[/green]" if t.solved else "[red]✘[/red]"
            winner_tag = " [bold yellow](WINNER)[/bold yellow]" if result.winner_policy == t.policy else ""
            table.add_row(
                t.policy + winner_tag,
                mark,
                str(t.steps),
                str(getattr(t, "cancelled", False)),
            )
        console.print(table)
        console.print(f"winner={result.winner_policy} wall_time={result.wall_time:.2f}s summary={result.summary_path}")


@app.command()
def episode(
    tasks: Path = typer.Option(..., "--tasks", help="Directory of task *.json files"),
    task_id: str = typer.Option(None, "--task-id", help="Run only this task_id (default: first)"),
    policy: str = typer.Option(..., "--policy", help="solo:MODEL or alloy:M1:0.6,M2:0.4"),
    out: Path = typer.Option(Path("runs/episode"), "--out", help="Transcript JSONL path"),
    mock: bool = typer.Option(False, "--mock", help="Use MockChatClient (offline)"),
    env: str = typer.Option("mock", "--env", help="mock | repl | docker | daytona"),
    max_steps: int = typer.Option(40, "--max-steps"),
) -> None:
    """Run a single-policy episode and write one transcript."""
    task_list = load_tasks(tasks)
    if task_id:
        task = next((t for t in task_list if t.task_id == task_id), None)
        if task is None:
            raise typer.BadParameter(f"task {task_id!r} not found in {tasks}")
    else:
        task = task_list[0]
    client = _build_client(mock)
    pol = parse_policy(policy)

    t0 = time.monotonic()
    t = asyncio.run(run_episode(task, pol, client, make_env_factory(env)(task), max_steps=max_steps))
    write_transcript(t, Path(out))
    console.print(
        f"episode {t.episode_id}: solved={t.solved} steps={t.steps} "
        f"wall={time.monotonic() - t0:.2f}s -> {out}"
    )


if __name__ == "__main__":
    app()
