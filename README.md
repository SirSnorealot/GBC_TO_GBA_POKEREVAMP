# GBC-to-GBA PokeRevamp

A deterministic, reference-guided converter that turns a Game Boy Color-era Pokémon-style
battle sprite (Pokémon Crystal look: ~4 colours, 40–56 px) into a plausible Game Boy
Advance-era draft (FireRed/LeafGreen/Emerald look: 64×64 canvas, ≤15 opaque colours +
transparency, richer shade ramps, selective coloured outlines).

It is **not** an image upscaler, **not** a neural generator, and **not** a recreation of the
official Gen III artwork. It automates the repeatable parts of the GBC→GBA transition —
canvas normalisation, palette expansion, outline reconstruction, shade/highlight synthesis,
pixel cleanup, strict palette enforcement — while preserving the pose and silhouette of the
source. The output is a draft a pixel artist can accept or touch up, and every stage is
inspectable.

```text
Crystal source (40x40, 4 colours)  ->  GBA-style draft (64x64, 10-15 colours)
```

## What it is / what it is not

| It does | It does not |
|---|---|
| Preserve source pose & silhouette | Redraw anatomy or infer a new pose |
| Build a GBA-style palette from the source's own hues plus reference *relative* shade behaviour | Copy a reference's literal colours onto the source |
| Add shadows/deep shadows/lights/highlights by edge distance and light direction | Blur, anti-alias, dither, or interpolate |
| Rebuild exterior outlines and soften internal linework | Use k-means or generic quantisation as the "algorithm" |
| Output 64×64 PNG with hard alpha and ≤15 opaque colours | Reproduce official Gen III sprites pixel for pixel |

## Requirements

- Python 3.11 or newer
- Git (only for automatic reference acquisition)
- VS Code (optional)

No Conda, Docker, WSL, Make, compiler, ROM file, or ML framework.

## Setup — Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
gbc_to_gba_pokerevamp assets bootstrap      # clones the pret repos into vendor/ and copies sprites into data/ (one time, needs Git)
```

## Setup — macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
gbc_to_gba_pokerevamp assets bootstrap
```

## VS Code

1. Open this folder in VS Code.
2. Run the setup commands above in the integrated terminal.
3. **Python: Select Interpreter** → choose `.venv`.
4. Open a new terminal (so the venv is active) and run `python -m gbc_to_gba_pokerevamp --help`.

## How automatic sprite acquisition works

`gbc_to_gba_pokerevamp assets bootstrap` (also `scripts/bootstrap_assets.py`):

1. Shallow-clones (`git clone --depth 1`) into the gitignored `vendor/`:
   - https://github.com/pret/pokered (Red/Blue), https://github.com/pret/pokeyellow, https://github.com/pret/pokegold (Gold + Silver), https://github.com/pret/pokecrystal — the inputs
   - https://github.com/pret/pokefirered, https://github.com/pret/pokeemerald — the Gen III references
2. Discovers front battle sprites by glob + image-size heuristics, excluding icons, footprints,
   eggs, backs, overworld sheets, etc. Rules live in
   [assets/manifests/sources.json](assets/manifests/sources.json).
3. Normalises names (`farfetchd` → `farfetch_d`, `mr.mime` → `mr_mime`, `unown/a` → `unown_a`, …).
4. Copies only the first animation frame into the gitignored `data/references/` tree, one file
   per game with a game prefix:
   - `data/references/gbc/pokemon/rb_<name>.png`, `yellow_<name>.png`, `gold_<name>.png`, `silver_<name>.png`, `crystal_<name>.png`
   - `data/references/gba/pokemon/frlg_<name>.png`, `emerald_<name>.png` (RGBA, palette index 0 → transparent)
   - same for `trainers/`

   Red/Blue/Yellow graphics are stored monochrome in the games; bootstrap applies the
   Super Game Boy / GBC palette each species was shown with (parsed from the decomp's
   `data/pokemon/palettes.asm` + `data/sgb/sgb_palettes.asm`) so you get the colours players saw.
5. Writes generated manifests with attribution + commit SHA:
   `data/manifests/sources.json`, `data/manifests/pokemon_pairs.json` (one entry per source
   game × species with a Gen III counterpart), `data/manifests/trainer_pairs.json`, and the
   curated `data/sample_set.json` (Crystal sources).

`--update` pulls existing clones with `git pull --ff-only`; `--skip-clone` only indexes what is
already in `vendor/`. Files in `vendor/` are never modified. If Git is missing the tool prints
the three URLs so you can clone manually.

## First commands to run

```powershell
gbc_to_gba_pokerevamp revamp .\data\references\gbc\pokemon\crystal_pikachu.png
```

That is the whole thing. Defaults: FRLG style, reference picked automatically (same species
if the file name matches, otherwise the closest silhouettes), 64×64 output. Results go in a
folder named after the sprite:

```text
output\crystal_pikachu\
  compare.png     source | revamp | reference, magnified   <- open this first
  revamped.png    the 64x64 result
  report.json     palettes, roles, geometry, warnings
  debug\          only with --debug: one PNG per pipeline stage (01_... 09_...)
```

The command prints this same "Look here" list when it finishes. Any of the five source games
works the same way — `rb_charizard.png`, `yellow_charizard.png`, `gold_charizard.png`,
`silver_charizard.png`, `crystal_charizard.png` all revamp toward the FRLG Charizard.
Useful variations:

```powershell
gbc_to_gba_pokerevamp revamp .\input\my_sprite.png --debug                # any pixel-art PNG, plus stage images
gbc_to_gba_pokerevamp revamp .\input\my_sprite.png --style emerald          # Emerald references instead of FRLG
gbc_to_gba_pokerevamp revamp .\input\my_sprite.png -r .\data\references\gba\pokemon\frlg_pikachu.png   # pick the reference yourself
gbc_to_gba_pokerevamp inspect .\input\my_sprite.png                          # what the tool sees in the source
gbc_to_gba_pokerevamp references list --game yellow                          # browse available source sprites
```

Convert the curated sample set (12 Pokémon + 7 trainers) and view one contact sheet:

```powershell
gbc_to_gba_pokerevamp batch .\data\sample_set.json
# -> output\batch\contact_sheet.png   (Crystal source | revamp | official Gen III reference, one row each)
# -> output\batch\<name>\revamped.png
```

## Providing a reference sprite

```powershell
gbc_to_gba_pokerevamp revamp .\input\sprite.png --reference .\data\references\gba\pokemon\frlg_pikachu.png
```

`--reference` may be repeated; without it references are chosen automatically. What is taken
from the reference depends on how it was chosen:

| how the reference was chosen | colours | shading / outline style |
|---|---|---|
| **same species** (file named `pikachu.png`, `rb_charizard.png`, `yellow_charizard.png`, … and that Gen III sprite exists) | adopted outright — the official sprite is the answer key: Crystal yellow becomes FRLG yellow, white shine becomes pale yellow, even a re-hued body (green → teal Bulbasaur) follows | adopted |
| **you passed `--reference`** | each source hue is pulled toward the same hue in the reference by `--reference-weight` (0.65); unmatched hues keep their colour | adopted |
| **auto-picked shape-alikes** (unknown sprite) | kept — a random look-alike must never recolour your character | adopted (how dark shadows are, how much highlight, black vs coloured outline share) |

`--palette-mode reference-palette` snaps every colour to the reference palette (experiment);
`--palette-mode source-expanded` ignores references for colours entirely.

## How auto-reference works

Auto-reference is on by default (`--no-auto-reference` disables it; `--reference-count`
changes how many are blended, default 3).

Silhouette descriptors (aspect ratio, area ratio, compactness, contour complexity,
protrusions, vertical centre of mass, symmetry, edge density, dominant hue) are computed for
the source and for every local Gen III reference matching `--style` and `--kind`. Descriptors
are cached in `data/cache/reference_descriptors.json`. The closest N references are aggregated
(best match weighted highest). If the input file name is a known species (`pikachu.png`,
`frlg_pikachu.png`, `pikachu_crystal.png` …) and a same-species reference exists, it ranks
first. Silhouettes are never copied from references.

## Pipeline and debug stages

The stages follow the classic revamping rules (never keep the old outline, black only in
shadow, shading that curves around the form with a top-left light, plenty of highlight tone,
no stray dots outside the outline):

```text
load → background/transparency → subject mask → colour roles → 64×64 geometry
→ reference style → hue families (thick black = body, white shine = light shade, dither = solid shade)
→ shade ramps → outline rebuild (1px; black in shadow, outline-base colour elsewhere, highlight colour on lit edges)
→ shade/highlight synthesis (illumination field quantised to reference proportions)
→ cluster cleanup → ≤15-colour enforcement → export + compare sheet + report
```

`--debug` adds `output\<name>\debug\`:

| file | content |
|---|---|
| `01_source_rgba.png` | input as RGBA with inferred background removed |
| `02_mask.png` | subject mask |
| `03_normalized.png` | source colours placed/scaled on the 64×64 canvas |
| `04_roles.png` | diagnostic colour-role map (outline / shadow / base / light …) |
| `05_palette_preview.png` | outline colour + every family ramp (deep, shadow, base, light, highlight, line) |
| `06_outlined.png` | render after outline reconstruction, before synthesised shading |
| `07_shaded.png`, `07_levels.png` | render after shading, plus a level/line diagnostic map (black = black outline, purple = outline-base colour, orange = outline highlight) |
| `08_cleaned.png` | after isolated-pixel cleanup |
| `09_final.png` | after palette enforcement (identical to the output) |

Diagnostic colours in `04_roles.png`/`07_levels.png` never enter the final image.

Every run also writes `report.json` (source/target palettes, roles, families and
ramps, geometry decision, reference statistics, diagnostics such as isolated-pixel count and
connected-component count, full config, and warnings) and `compare.png`.

## CLI overview

```text
gbc_to_gba_pokerevamp revamp INPUT [--style frlg|emerald|gen3-mixed] [-r REF]... [--debug] [-o output]
gbc_to_gba_pokerevamp batch DIR_OR_MANIFEST [--style ...] [--debug] [-o output/batch]
gbc_to_gba_pokerevamp inspect PATH [--json]
gbc_to_gba_pokerevamp compare SOURCE RESULT [REFERENCE] [--scale N]
gbc_to_gba_pokerevamp references list [--kind pokemon|trainer] [--game rb|yellow|gold|silver|crystal|frlg|emerald] [--sources]
gbc_to_gba_pokerevamp assets bootstrap [--update] [--skip-clone]
```

`revamp --help` groups the rest under **Tuning** (`--shading-strength`, `--outline-strength`,
`--highlight-strength`, `--reference-weight`, `--hue-shift-strength`, `--cleanup-strength`,
`--palette-mode`, `--geometry-mode`) and **Advanced** (`--kind`, `--auto-reference`,
`--reference-count`, `--reference-mode`, `--max-colors`, `--canvas-size`, `--light-x/-y`,
`--anchor-x/-y`, `--indexed`, `--compare`, `--report`). You should not need any of them for a
first result; reach for `--reference-weight` first if you want the output closer to (1.0) or
further from (0.0) the reference colours.

For repeatable tuning put options in a JSON file and pass `--config`; flags still override it:

```json
{
  "style": "frlg",
  "canvas_size": 64,
  "max_colors": 16,
  "palette_mode": "reference-guided",
  "geometry_mode": "none",
  "outline_strength": 0.75,
  "shading_strength": 0.7,
  "highlight_strength": 0.45,
  "cleanup_strength": 0.5,
  "reference_weight": 0.65,
  "source_weight": 0.35
}
```

## Manual smoke test (no test framework)

```powershell
.\.venv\Scripts\Activate.ps1
gbc_to_gba_pokerevamp revamp .\data\references\gbc\pokemon\crystal_pikachu.png --debug
#   -> open output\crystal_pikachu\compare.png, then output\crystal_pikachu\debug\
gbc_to_gba_pokerevamp batch .\data\sample_set.json
#   -> open output\batch\contact_sheet.png
# any pixel-art PNG you drop into input\:
gbc_to_gba_pokerevamp revamp .\input\my_sprite.png
```

Helper scripts (run with the venv Python): `scripts/bootstrap_assets.py`,
`scripts/build_pair_manifest.py`, `scripts/inspect_sprite.py`, `scripts/make_contact_sheet.py`.

## Pixel-art invariants

No anti-aliasing, no bilinear/bicubic/Lanczos resampling, alpha is exactly 0 or 255, integer
pixel coordinates only, palette reduction is explicit (perceptual merge of the closest
non-protected colours, outline and accent colours preserved), output is 64×64 by default, and
`--indexed` additionally writes a ≤16-entry paletted PNG with index 0 transparent.

Geometry modes: `none` (default) keeps the source pixel size and centres it on the 64×64 canvas
(downscaling only if the subject is larger than the canvas); `integer` allows integer
nearest-neighbour enlargement; `pixel-aware` allows modest fractional nearest-neighbour
enlargement and then re-thins dark lines that were doubled by duplicated rows/columns.

## Important limitations

- This is a **style revamp**, not a **historical redraw**. Official Gen III sprites frequently
  change pose, proportions, and expression; this tool keeps the source pose.
- Shading is synthesised from silhouette orientation under a configurable upper-left light.
  It cannot know real volume; expect to touch up drafts.
- Dark GBC regions are kept as evidence: a sprite the original artist drew mostly in its dark
  colour stays dark-heavy.
- Accents the GBC artist drew in the shading colour (Pikachu's cheeks, ear tips) cannot be told
  apart from shading and take the shade colour, not the official accent colour.
- Inputs with many colours (e.g. an existing GBA sprite) are treated gently: each hue family is
  capped at five shade levels, so some source shades can collapse.
- Trainer handling is generic (tighter hue-family grouping, halved hue shifts, smaller accent
  protection); it has no notion of skin/hair/clothing semantics yet.
- Same-colour regions that touch the canvas edge and match the background colour are treated as
  background; enclosed same-colour regions are kept (flood fill from edges).
- Red/Blue/Yellow sprites are coloured with their 4-colour SGB/GBC palette before revamping.
  Yellow's outside-the-outline anti-aliasing dots are stripped (they'd show on dark backgrounds).
- Quality is judged visually (compare sheets, contact sheets, debug stages), never by pixel
  equality with official sprites.

## Data / source notice

GBC-to-GBA PokeRevamp does not include Pokémon game ROMs or bundled game graphics. Optional
bootstrap tooling can clone public source-decompilation repositories into a local, gitignored
directory for research and personal comparison. Users are responsible for complying with
applicable rights, licenses, and laws for any assets they use or redistribute. This project is
not affiliated with or endorsed by Nintendo, Game Freak, or The Pokémon Company.

## Attribution

Reference graphics are obtained locally from the pret decompilation projects:

- [pret/pokered](https://github.com/pret/pokered), [pret/pokeyellow](https://github.com/pret/pokeyellow)
- [pret/pokegold](https://github.com/pret/pokegold), [pret/pokecrystal](https://github.com/pret/pokecrystal)
- [pret/pokefirered](https://github.com/pret/pokefirered), [pret/pokeemerald](https://github.com/pret/pokeemerald)

Each copied image is recorded in `data/manifests/sources.json` with its source repository,
commit SHA, and upstream path.

## Roadmap

1. **Deterministic MVP (this release)** — masks/roles/ramps/outlines/shading as separate
   layers combined only in the renderer.
2. **Paired statistical style model** — learn occupancy, outline ratio, shade densities and
   colour-count changes from all Crystal→Gen III pairs, aggregated by coarse morphology class.
3. **Correspondence-guided redraw suggestions** — signed-distance/contour correspondence on pairs
   to propose conservative silhouette masks (switchable off).
4. **Optional learned components** (`pip install -e ".[ml]"`, not in the MVP) — predict
   silhouette/outline/shadow/highlight masks rather than RGB, then reuse the deterministic
   renderer and palette engine.

## Project layout

```text
assets/manifests/sources.json   static acquisition rules + name normalisation
scripts/                        bootstrap_assets, build_pair_manifest, inspect_sprite, make_contact_sheet
src/gbc_to_gba_pokerevamp/
  cli.py          Typer CLI            config.py     RevampConfig (+ JSON config files)
  models.py       data models          colorspace.py sRGB <-> Lab/LCh (NumPy)
  sprite_io.py    load/save, hard alpha analyze.py   background, masks, colour roles, descriptors
  normalize.py    64x64 geometry       palette.py    hue families, shade ramps, colour-limit enforcement
  outline.py      outline rebuild      shading.py    shadow/highlight synthesis
  geometry.py     lighting + cleanup   render.py     layer compositor, debug + compare renderers
  references.py   discovery, style extraction, auto-reference ranking
  revamp.py       pipeline             report.py     JSON report
  assets.py       bootstrap logic      contact_sheet.py
data/ vendor/ output/ input/          gitignored local data (input/ and output/ keep a .gitkeep)
```
