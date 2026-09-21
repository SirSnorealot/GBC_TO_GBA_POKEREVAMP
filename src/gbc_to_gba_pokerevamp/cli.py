"""Entry point: launches the PokeRevamp Assistant GUI (optionally fetching references first)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from gbc_to_gba_pokerevamp import __version__

app = typer.Typer(add_completion=False, help="PokeRevamp Assistant — interactive GB/GBC → GBA sprite revamp editor.")
console = Console()
err_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def main(
    input: Optional[Path] = typer.Argument(None, help="Sprite PNG to open immediately."),
    bootstrap: bool = typer.Option(False, "--bootstrap", help="Download/refresh the reference sprites (needs Git), then exit."),
    update: bool = typer.Option(False, "--update", help="With --bootstrap: git pull existing clones."),
    version: bool = typer.Option(False, "--version", is_eager=True),
) -> None:
    if version:
        console.print(f"PokeRevamp Assistant {__version__}")
        raise typer.Exit()
    if bootstrap:
        from gbc_to_gba_pokerevamp.assets import BootstrapError, bootstrap as run_bootstrap

        try:
            summary = run_bootstrap(update=update)
        except BootstrapError as exc:
            err_console.print(f"[bold red]error:[/] {exc}")
            raise typer.Exit(1)
        console.print_json(json.dumps(summary))
        raise typer.Exit()
    if input is not None and not input.exists():
        err_console.print(f"[bold red]error:[/] input file not found: {input}")
        raise typer.Exit(1)
    from gbc_to_gba_pokerevamp.gui import launch

    launch(input)


if __name__ == "__main__":
    app()
