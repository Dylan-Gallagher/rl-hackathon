"""Typer CLI for the path4 scoreboard.

``python -m path4.scoreboard.cli serve --transcripts runs/demo --port 8080``
``python -m path4.scoreboard.cli table  --transcripts runs/demo``
``python -m path4.scoreboard.cli export --transcripts runs/demo --json out.json``
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from path4.scoreboard.metrics import aggregate, render_table, scan_race_summaries, scan_transcripts

app = typer.Typer(help="Path 4 live scoreboard: Pass@k / Maj@k over canonical transcripts.")
console = Console()


def _load(transcripts_dir: Path, ks: tuple[int, ...] | list[int] | None = None):
    ts = scan_transcripts(transcripts_dir)
    races = scan_race_summaries(transcripts_dir)
    if ks is not None:
        return aggregate(ts, races, ks=ks), len(ts)
    return aggregate(ts, races), len(ts)


@app.command()
def serve(
    transcripts: Path = typer.Option(..., "--transcripts", help="Transcripts dir (scanned recursively)."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    refresh: float = typer.Option(3.0, "--refresh", help="Rescan interval seconds."),
) -> None:
    """Serve the live scoreboard web UI."""
    import uvicorn

    from path4.scoreboard.server import create_app

    uvicorn.run(create_app(transcripts, refresh_s=refresh), host=host, port=port)


@app.command()
def table(
    transcripts: Path = typer.Option(..., "--transcripts"),
    ks: str = typer.Option("1,4,8", "--ks", help="Comma-separated k values."),
) -> None:
    """Print the scoreboard as a rich console table."""
    k_list = [int(k) for k in ks.split(",") if k.strip()]
    summary, n = _load(transcripts, ks=k_list)
    if n == 0:
        console.print(f"[yellow]no transcripts found under {transcripts}[/yellow]")
        raise typer.Exit(1)
    console.print(render_table(summary))


@app.command()
def export(
    transcripts: Path = typer.Option(..., "--transcripts"),
    json_out: Path = typer.Option(..., "--json", help="Output JSON path."),
) -> None:
    """Export the aggregated summary to a JSON file."""
    summary, n = _load(transcripts)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]wrote {json_out} ({n} episodes, {len(summary['policies'])} policies)[/green]")


if __name__ == "__main__":
    app()
