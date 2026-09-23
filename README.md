# PokeRevamp Assistant

An interactive editor for turning Game Boy / Game Boy Color Pokémon battle sprites
(Red/Blue, Yellow, Gold, Silver, Crystal) into Game Boy Advance-style revamps
(FireRed/LeafGreen/Emerald look: 64×64 canvas, ≤15 colors + transparency, colored
outlines, real shading).

It is an **assistant, not a converter**. An automatic pass gives you a starting point — the
official Gen III sprite's colors laid onto your sprite, a rebuilt outline, synthesized
shading — and then you finish it by hand, live: pick any color from the reference and paint
pixels or whole regions on the original or on the result, remap a source color wholesale,
toggle each effect, drag the sliders. Everything updates in the preview as you work.

## Requirements

- Python 3.11 or newer (tkinter is included with the standard Windows/macOS installers)
- Git (only for the one-time reference download)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
gbc_to_gba_pokerevamp --bootstrap     # one time: downloads the Gen I–III sprites (a few minutes)
```

`--bootstrap` shallow-clones the pret decompilation repositories into the gitignored
`vendor/` folder and copies only the front battle sprites into the gitignored
`data/references/`. You can also do this later from inside the app (**Download refs…**).

## Run

```powershell
gbc_to_gba_pokerevamp                          # open the editor
gbc_to_gba_pokerevamp .\input\crystal_totodile.png   # open with a sprite loaded
```

There are ~1,100 ready-made inputs under `data\references\gbc\pokemon\` and `...\trainers\`
(`rb_`, `yellow_`, `gold_`, `silver_`, `crystal_` prefixes), or drop your own PNG into `input\`.

## The editor

Dark theme by default (**Theme** button toggles). Left column = controls, right = preview.

| panel | what it does |
|---|---|
| **Sprite** | open any PNG; kind (pokemon / trainer) and style (emerald by default, frlg, or gen3-mixed) |
| **Gen III reference** | *Auto* (same species when the file name matches, else shape-alikes for shading style only), *None* (keep the source's own colors), or *Pick* from the searchable list / browse to any PNG. **Shiny palette** swaps the reference for its shiny version (from the decomp's `shiny.pal`) so you can preview the revamp in shiny colors. *Download refs…* fetches the sprite corpus. |
| **Paint** | **Pick color** (click any panel, including the reference), **Pencil** (click / drag), **Fill region** (click recolors the connected same-color area), **Undo** (per stroke). The palette shows every color of the reference (one row per color region, dark → light), the source colors and the result colors; *Custom…* and *Transparent* too. Paint on the **Source** or on the **Result** — either way, pixels you paint are kept exactly as painted; the automatic coloring and shading never touch them. Right-click a painted pixel to erase that edit. |
| **Color mapping** | one row per source color: swatch, role, pixel count, what it currently becomes, and an **override** to send every pixel of that color to a chosen reference color, *Keep*, *Outline*, *Transparent* (fixes background mis-detection) or *Custom*. Clicking a source pixel with the Pick tool highlights its row. |
| **Effects** | toggle positional recolor, outline rebuild, shading, 15-color limit; *Treat reference as same Pokémon* forces full color adoption for a hand-picked reference |
| **Tuning** | shading / outline / highlight / hue-shift / reference-weight / cleanup strengths and light direction |
| **Preview** | Source ❘ Result ❘ Reference; zoom or *fit*; checker / dark / light background; *Show* switches the result panel to any intermediate stage (flat recolor, outlined, shaded, level map…). Hovering shows the pixel coordinate and color. **Frames**: Crystal sprites carry their extra battle frames; Emerald references carry two, and both are used — the first chosen frame is revamped against Emerald's first frame, the second against its second (the reference shown on each row is the one that frame used). Sprites with frames open in the **All frames** view: one row per frame, Source on the left and Result on the right, all editable at once — painting on any frame edits that frame and makes it current (blue outline). Untick *All frames* to work on one frame at a time with ◀ ▶; **Edits → all** copies the current frame's paint onto every frame. Every frame uses the same settings, color map and canvas placement; paint edits are per frame. **Frames in the output**: tick the frames you want (default all; **1+2** keeps just the first two, the usual Gen III pair; **all** resets). Unticked frames disappear from the preview and are left out of the saved sheet. |
| **Output** | *Save to output/* writes `output\<name>\revamped.png` (the current frame), `compare.png`, `report.json` and `session.json`; when more than one frame is selected also `revamped_frames.png` (the chosen frames stacked vertically). *Save / Load session* stores every setting, the color map, the frame selection and all paint edits for every frame. |

## What the automatic pass does

```text
detect background → color roles → center on 64×64 (no enlargement)
→ pick reference → RECOLOR (same species: every body pixel → the official color region at
  that position; else hue-matched families) → manual color map
→ outline rebuild (1 px; black only in shadow, dark local color elsewhere, highlight tone on
  lit edges) → shading (each body part lit as its own rounded form, in the reference's
  proportions) → stray-pixel cleanup → ≤15 colors
```

Red/Blue/Yellow sprites are stored monochrome in the games; bootstrap applies each species'
Super Game Boy palette so they arrive in color. Yellow's outside-the-outline anti-aliasing is
folded into the outline. GBC dithering becomes solid shades; thick black masses become the
official dark body color. Crystal's stacked battle frames are extracted as `<name>_frames.png`
(frames stacked vertically, each as tall as the sheet is wide); Emerald's two-frame
`anim_front.png` likewise. When you open `foo.png` the app looks for `foo_frames.png` next to
it; any tall stacked PNG you draw yourself works the same way.

Pixel-art rules are never broken: no resampling, no anti-aliasing, alpha is exactly 0 or 255.

## Limitations

- The pose is never redrawn. The sprite stays in its GBC pose; only colors, outline and
  shading change. Where the official Gen III sprite has a different pose, position-based
  recoloring can put a color on the wrong part — that is what the paint tools are for.
- A color the source does not have cannot be inferred (Yellow Pikachu has no red cheeks).
  Paint it.
- Trainers get hue-matched colors only; their Gen III redesigns rarely share a layout.

## Data notice

PokeRevamp Assistant does not include Pokémon game ROMs or bundled game graphics. The
bootstrap clones public source-decompilation repositories into a local, gitignored directory
for research and personal comparison. Users are responsible for complying with applicable
rights, licenses, and laws for any assets they use or redistribute. Not affiliated with or
endorsed by Nintendo, Game Freak, or The Pokémon Company.

Reference graphics come from the pret decompilation projects:
[pokered](https://github.com/pret/pokered), [pokeyellow](https://github.com/pret/pokeyellow),
[pokegold](https://github.com/pret/pokegold), [pokecrystal](https://github.com/pret/pokecrystal),
[pokefirered](https://github.com/pret/pokefirered), [pokeemerald](https://github.com/pret/pokeemerald).
Every copied image is recorded in `data/manifests/sources.json` with repository, commit and path.

## Project layout

```text
assets/manifests/sources.json   acquisition rules + name normalization
scripts/bootstrap_assets.py     same as `gbc_to_gba_pokerevamp --bootstrap`
src/gbc_to_gba_pokerevamp/
  cli.py        launcher (--bootstrap, --version)
  gui.py        the editor (tkinter)
  revamp.py     automatic pass: revamp_sprite() -> final image + every stage
  recolor.py    same-species positional recolor
  palette.py    color families, shade ramps, color-limit enforcement
  outline.py    outline rebuild        shading.py   shade synthesis
  analyze.py    background, roles      normalize.py canvas placement
  references.py reference indexing / style extraction / ranking
  assets.py     sprite acquisition     config.py    RevampConfig
data/ vendor/ output/ input/          gitignored local data
```
