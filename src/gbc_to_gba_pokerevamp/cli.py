"""Typer command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from gbc_to_gba_pokerevamp import __version__
from gbc_to_gba_pokerevamp.config import RevampConfig

app = typer.Typer(
    name="gbc_to_gba_pokerevamp",
    help="Deterministic, reference-guided GBC-style -> GBA-style Pokémon battle sprite draft converter.",
    no_args_is_help=True,
    add_completion=False,
)
assets_app = typer.Typer(help="Acquire and index reference sprites from pret decompilation repositories.")
references_app = typer.Typer(help="Browse locally indexed Gen III references.")
app.add_typer(assets_app, name="assets")
app.add_typer(references_app, name="references")

console = Console()
err_console = Console(stderr=True)


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit.", is_eager=True),
) -> None:
    if version:
        console.print(f"gbc_to_gba_pokerevamp {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


# --- inspect -------------------------------------------------------------------------------


def inspect_sprite(path: Path) -> dict[str, Any]:
    from gbc_to_gba_pokerevamp.analyze import analyze_colors, connected_component_count
    from gbc_to_gba_pokerevamp.sprite_io import load_sprite, open_image

    img = open_image(path)
    sprite = load_sprite(path)
    infos = analyze_colors(sprite.rgba, sprite.opaque_mask)
    raw = np.array(img.convert("RGBA"))
    unique_rgb = len(np.unique(raw[..., :3].reshape(-1, 3), axis=0))
    palette_used: list[Any] = []
    if img.mode == "P":
        idx = np.array(img)
        pal = img.getpalette() or []
        used = sorted(int(i) for i in np.unique(idx))
        palette_used = [{"index": i, "rgb": tuple(pal[i * 3 : i * 3 + 3])} for i in used if i * 3 + 2 < len(pal)]
    x0, y0, x1, y1 = sprite.bbox
    darkest = min(infos, key=lambda c: c.L) if infos else None
    return {
        "path": str(path),
        "width": sprite.size[0],
        "height": sprite.size[1],
        "mode": img.mode,
        "has_alpha": bool("A" in img.mode or "transparency" in img.info),
        "transparency_info": str(img.info.get("transparency")) if "transparency" in img.info else None,
        "unique_rgb_colors": int(unique_rgb),
        "opaque_colors": len(infos),
        "palette_entries_used": palette_used,
        "bbox": [x0, y0, x1, y1],
        "content_width": x1 - x0,
        "content_height": y1 - y0,
        "connected_components": connected_component_count(sprite.opaque_mask),
        "background_color": sprite.background_color,
        "outline_or_darkest_color": darkest.rgb if darkest else None,
        "colors": [
            {
                "rgb": c.rgb,
                "count": c.count,
                "frequency": round(c.frequency, 4),
                "L": round(c.L, 1),
                "chroma": round(c.chroma, 1),
                "hue": round(c.hue, 1),
                "boundary_share": round(c.boundary_share, 3),
                "role": c.role.name,
            }
            for c in infos
        ],
        "warnings": sprite.warnings,
    }


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="Sprite PNG to inspect."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Print source sprite statistics (dimensions, alpha, colours, roles, bounds)."""
    from gbc_to_gba_pokerevamp.sprite_io import SpriteLoadError

    try:
        info = inspect_sprite(path)
    except SpriteLoadError as exc:
        _fail(str(exc))
        return
    if as_json:
        console.print_json(json.dumps(info, default=str))
        return
    table = Table(title=f"{path}", show_header=False)
    for key in (
        "width", "height", "mode", "has_alpha", "transparency_info", "unique_rgb_colors", "opaque_colors",
        "bbox", "content_width", "content_height", "connected_components", "background_color", "outline_or_darkest_color",
    ):
        table.add_row(key, str(info[key]))
    console.print(table)
    ct = Table(title="Opaque colours")
    for col in ("rgb", "count", "frequency", "L", "chroma", "hue", "boundary_share", "role"):
        ct.add_column(col)
    for c in info["colors"]:
        ct.add_row(*(str(c[k]) for k in ("rgb", "count", "frequency", "L", "chroma", "hue", "boundary_share", "role")))
    console.print(ct)
    if info["palette_entries_used"]:
        console.print("palette entries used:", info["palette_entries_used"])
    for w in info["warnings"]:
        console.print(f"[yellow]warning:[/] {w}")


# --- assets / references -------------------------------------------------------------------


@assets_app.command("bootstrap")
def assets_bootstrap(
    update: bool = typer.Option(False, "--update", help="git pull --ff-only existing clones."),
    skip_clone: bool = typer.Option(False, "--skip-clone", help="Only index repositories already present in vendor/."),
) -> None:
    """Shallow-clone pret/pokecrystal, pokefirered, pokeemerald and copy front sprites into data/."""
    from gbc_to_gba_pokerevamp.assets import BootstrapError, bootstrap

    try:
        summary = bootstrap(update=update, skip_clone=skip_clone)
    except BootstrapError as exc:
        _fail(str(exc))
        return
    console.print_json(json.dumps(summary))


@references_app.command("list")
def references_list(
    kind: Optional[str] = typer.Option(None, "--kind", help="pokemon | trainer"),
    game: Optional[str] = typer.Option(None, "--game", help="rb | yellow | gold | silver | crystal | frlg | emerald"),
    sources: bool = typer.Option(False, "--sources", help="List Gen I/II source sprites instead of Gen III references."),
) -> None:
    """List locally indexed sprites (Gen III references by default, --sources for GB/GBC inputs)."""
    from gbc_to_gba_pokerevamp.references import list_references

    era = "gbc" if sources or (game and game in ("rb", "yellow", "gold", "silver", "crystal")) else "gba"
    entries = list_references(kind=kind, game=game, era=era)
    if not entries:
        _fail("no sprites found. Run `gbc_to_gba_pokerevamp assets bootstrap` first.")
        return
    table = Table(title=f"{len(entries)} sprites")
    table.add_column("kind")
    table.add_column("game")
    table.add_column("name")
    table.add_column("path")
    for e in entries:
        table.add_row(e.kind, e.game, e.name, str(e.path))
    console.print(table)


# --- revamp ---------------------------------------------------------------------------------


def _build_config(config_file: Optional[Path], overrides: dict[str, Any]) -> RevampConfig:
    if config_file is not None:
        if not config_file.exists():
            _fail(f"config file not found: {config_file}")
        return RevampConfig.from_file(config_file, overrides)
    return RevampConfig().with_overrides(overrides)


TUNING = "Tuning (all have sensible defaults)"
ADVANCED = "Advanced"


def _print_result(result) -> None:
    rep = result.report
    console.print(
        f"[green]done[/] {Path(rep.input).name}: {rep.source_opaque_colors} -> {rep.target_opaque_colors} opaque colours, "
        f"geometry={rep.geometry['mode_used']} x{rep.geometry['scale']:.2f}"
    )
    if rep.reference_files:
        console.print("  references: " + ", ".join(Path(p).name for p in rep.reference_files))
    for w in rep.warnings:
        console.print(f"  [yellow]warning:[/] {w}")
    console.print(f"\n[bold]Look here:[/] {result.run_dir}")
    if result.compare_path:
        console.print("  compare.png      source | revamp | reference, magnified  <- start here")
    console.print("  revamped.png     the 64x64 result")
    if result.report_path:
        console.print("  report.json      palettes, roles, geometry, warnings")
    if result.debug_dir:
        console.print("  debug/           one PNG per pipeline stage (01_... to 09_...)")


@app.command()
def revamp(
    input: Path = typer.Argument(..., help="Source sprite PNG."),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="Results go to <output>/<sprite name>/."),
    style: Optional[str] = typer.Option(None, "--style", help="frlg (default) | emerald | gen3-mixed"),
    reference: list[Path] = typer.Option([], "--reference", "-r", help="Gen III reference PNG (repeatable). Default: picked automatically."),
    debug: bool = typer.Option(False, "--debug", help="Also write one PNG per pipeline stage."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="JSON config file; flags override it."),
    # Tuning
    shading_strength: Optional[float] = typer.Option(None, "--shading-strength", min=0.0, max=1.0, rich_help_panel=TUNING, help="0.7"),
    outline_strength: Optional[float] = typer.Option(None, "--outline-strength", min=0.0, max=1.0, rich_help_panel=TUNING, help="0.75"),
    highlight_strength: Optional[float] = typer.Option(None, "--highlight-strength", min=0.0, max=1.0, rich_help_panel=TUNING, help="0.45"),
    reference_weight: Optional[float] = typer.Option(None, "--reference-weight", min=0.0, max=1.0, rich_help_panel=TUNING, help="How strongly reference colours/shading pull the result (0.65)."),
    hue_shift_strength: Optional[float] = typer.Option(None, "--hue-shift-strength", min=0.0, max=1.0, rich_help_panel=TUNING, help="0.6"),
    cleanup_strength: Optional[float] = typer.Option(None, "--cleanup-strength", min=0.0, max=1.0, rich_help_panel=TUNING, help="0.5"),
    palette_mode: Optional[str] = typer.Option(None, "--palette-mode", rich_help_panel=TUNING, help="reference-guided (default) | source-expanded | reference-palette"),
    geometry_mode: Optional[str] = typer.Option(None, "--geometry-mode", rich_help_panel=TUNING, help="none (default: keep pixel size, centre on 64x64) | integer | pixel-aware"),
    # Advanced
    kind: Optional[str] = typer.Option(None, "--kind", rich_help_panel=ADVANCED, help="pokemon | trainer (default: inferred from path)"),
    auto_reference: Optional[bool] = typer.Option(None, "--auto-reference/--no-auto-reference", rich_help_panel=ADVANCED, help="Default on."),
    reference_count: Optional[int] = typer.Option(None, "--reference-count", rich_help_panel=ADVANCED, help="3"),
    reference_mode: Optional[str] = typer.Option(None, "--reference-mode", rich_help_panel=ADVANCED, help="style (default) | palette"),
    max_colors: Optional[int] = typer.Option(None, "--max-colors", rich_help_panel=ADVANCED, help="Palette entries incl. transparency (16)."),
    canvas_size: Optional[int] = typer.Option(None, "--canvas-size", rich_help_panel=ADVANCED, help="64"),
    light_x: Optional[float] = typer.Option(None, "--light-x", rich_help_panel=ADVANCED, help="-1.0"),
    light_y: Optional[float] = typer.Option(None, "--light-y", rich_help_panel=ADVANCED, help="-1.0"),
    anchor_x: Optional[int] = typer.Option(None, "--anchor-x", rich_help_panel=ADVANCED),
    anchor_y: Optional[int] = typer.Option(None, "--anchor-y", rich_help_panel=ADVANCED),
    indexed: bool = typer.Option(False, "--indexed", rich_help_panel=ADVANCED, help="Also write a <=16 entry indexed PNG."),
    compare: bool = typer.Option(True, "--compare/--no-compare", rich_help_panel=ADVANCED, help="Write compare.png (default on)."),
    report: bool = typer.Option(True, "--report/--no-report", rich_help_panel=ADVANCED),
) -> None:
    """Convert one sprite into a 64x64 GBA-style draft.  Minimal use:  revamp sprite.png"""
    from gbc_to_gba_pokerevamp.revamp import RevampError, run_revamp
    from gbc_to_gba_pokerevamp.sprite_io import SpriteLoadError

    overrides = {
        "kind": kind, "style": style, "reference_mode": reference_mode, "auto_reference": auto_reference,
        "reference_count": reference_count, "palette_mode": palette_mode, "geometry_mode": geometry_mode,
        "outline_strength": outline_strength, "shading_strength": shading_strength,
        "highlight_strength": highlight_strength, "hue_shift_strength": hue_shift_strength,
        "cleanup_strength": cleanup_strength, "reference_weight": reference_weight, "max_colors": max_colors,
        "canvas_size": canvas_size, "light_x": light_x, "light_y": light_y, "anchor_x": anchor_x,
        "anchor_y": anchor_y, "indexed_output": indexed or None, "debug": debug or None,
        "report": report, "compare": compare,
    }
    config = _build_config(config_file, overrides)
    for r in reference:
        if not r.exists():
            _fail(f"reference file not found: {r}")
    try:
        result = run_revamp(input, output, config, list(reference))
    except (SpriteLoadError, RevampError) as exc:
        _fail(str(exc))
        return
    _print_result(result)


@app.command()
def compare(
    source: Path = typer.Argument(...),
    result: Path = typer.Argument(...),
    reference: Optional[Path] = typer.Argument(None),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Default: next to RESULT as compare.png"),
    scale: int = typer.Option(4, "--scale", help="Integer magnification for the enlarged row."),
) -> None:
    """Render a comparison sheet for any three PNGs (revamp already writes one automatically)."""
    from gbc_to_gba_pokerevamp.render import compare_sheet
    from gbc_to_gba_pokerevamp.sprite_io import SpriteLoadError, load_sprite

    paths = [p for p in (source, result, reference) if p is not None]
    labels = ["source", "revamp", "reference"][: len(paths)]
    try:
        images = [load_sprite(p).image for p in paths]
    except SpriteLoadError as exc:
        _fail(str(exc))
        return
    sheet = compare_sheet(images, [f"{l}: {p.name}" for l, p in zip(labels, paths)], scale=scale)
    if output is None:
        output = result.with_name("compare.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    console.print(f"[green]wrote[/] {output}")


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory of PNG sprites, or data/sample_set.json / a pairs manifest."),
    output: Path = typer.Option(Path("output") / "batch", "--output", "-o", help="One sub-folder per sprite plus contact_sheet.png."),
    style: Optional[str] = typer.Option(None, "--style", help="frlg (default) | emerald | gen3-mixed"),
    debug: bool = typer.Option(False, "--debug"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="JSON config file for all tuning options."),
) -> None:
    """Convert every sprite in a directory (or manifest) and build one contact sheet."""
    from gbc_to_gba_pokerevamp.contact_sheet import make_contact_sheet
    from gbc_to_gba_pokerevamp.revamp import RevampError, run_revamp
    from gbc_to_gba_pokerevamp.sprite_io import SpriteLoadError

    config = _build_config(config_file, {"style": style, "debug": debug or None, "compare": False})
    jobs: list[tuple[Path, list[Path]]] = []
    if input_dir.is_file() and input_dir.suffix == ".json":
        data = json.loads(input_dir.read_text(encoding="utf-8"))
        entries = data["pokemon"] + data["trainers"] if isinstance(data, dict) and "pokemon" in data else data
        root = Path.cwd()
        for e in entries:
            jobs.append((root / e["source"], [root / t for t in e.get("targets", [])][:1]))
    elif input_dir.is_dir():
        jobs = [(p, []) for p in sorted(input_dir.glob("*.png"))]
    else:
        _fail(f"input must be a directory or JSON manifest: {input_dir}")
    if not jobs:
        _fail(f"no PNG files found in {input_dir}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Path, Path, Path | None]] = []
    failures = 0
    for src, refs in jobs:
        try:
            res = run_revamp(src, output, config, refs)
        except (SpriteLoadError, RevampError) as exc:
            failures += 1
            err_console.print(f"[red]failed[/] {src.name}: {exc}")
            continue
        ref = refs[0] if refs else (Path(res.report.reference_files[0]) if res.report.reference_files else None)
        rows.append((src, res.output_path, ref))
        console.print(f"[green]ok[/] {src.name} ({res.report.target_opaque_colors} colours)")
    if rows:
        sheet_path = output / "contact_sheet.png"
        make_contact_sheet(rows, sheet_path)
        console.print(f"\n[bold]Look here:[/] {sheet_path}   (source | revamp | reference for every sprite)")
        console.print(f"  per-sprite folders: {output / '<name>' / 'revamped.png'}")
    console.print(f"done: {len(rows)} converted, {failures} failed")
    if failures and not rows:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
