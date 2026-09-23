"""PokeRevamp Assistant — the interactive editor.

The automatic pass is only a starting point. Everything it produces can be steered
(reference, effects, sliders, per-color mapping) and then finished by hand: paint single
pixels or whole regions on the original *or* on the result with any color from the Gen III
reference, and watch the preview update live.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage

from gbc_to_gba_pokerevamp.colorspace import rgb_to_lch
from gbc_to_gba_pokerevamp.config import RevampConfig
from gbc_to_gba_pokerevamp.models import RGB, SpriteImage
from gbc_to_gba_pokerevamp.paths import project_root
from gbc_to_gba_pokerevamp.references import canonical_name_from_path, list_references, normal_variant, shiny_variant
from gbc_to_gba_pokerevamp.render import compare_sheet
from gbc_to_gba_pokerevamp.revamp import RevampError, RevampOutcome, infer_kind, revamp_sprite, rgba_to_image
from gbc_to_gba_pokerevamp.sprite_io import SpriteLoadError, find_frame_sheet, frame_count, frame_ref, load_frames, load_sprite, save_rgba_png

CANVAS = 64
STAGES = [
    ("Final result", "09_final"),
    ("Recolored (flat)", "05_recolored"),
    ("Outlined", "06_outlined"),
    ("Shaded", "07_shaded"),
    ("Cleaned", "08_cleaned"),
    ("Level map (diagnostic)", "07_levels"),
    ("Role map (diagnostic)", "04_roles"),
]
SLIDERS = [
    ("Shading", "shading_strength", 0.0, 1.0),
    ("Outline", "outline_strength", 0.0, 1.0),
    ("Highlights", "highlight_strength", 0.0, 1.0),
    ("Hue shift", "hue_shift_strength", 0.0, 1.0),
    ("Reference weight", "reference_weight", 0.0, 1.0),
    ("Cleanup", "cleanup_strength", 0.0, 1.0),
    ("Light X", "light_x", -1.0, 1.0),
    ("Light Y", "light_y", -1.0, 1.0),
]
EFFECTS = [
    ("Recolor to reference layout", "recolor"),
    ("Rebuild outline", "rebuild_outline"),
    ("Add shading", "shade"),
    ("Lock output palette to the reference's colors", "lock_palette"),
    ("Limit to 15 colors", "enforce_palette"),
    ("Treat reference as same Pokémon", "force_same_subject"),
]

THEMES = {
    "dark": {
        "bg": "#1f2023", "panel": "#26272b", "field": "#2e3035", "fg": "#e8e8e8", "muted": "#9a9da3",
        "accent": "#4c8bf5", "border": "#3a3c42", "select": "#3b4a6b", "canvas": "#151618", "log": "#1a1b1e",
    },
    "light": {
        "bg": "#f0f0f0", "panel": "#f7f7f7", "field": "#ffffff", "fg": "#1a1a1a", "muted": "#606060",
        "accent": "#2a6fdb", "border": "#c8c8c8", "select": "#cfe0ff", "canvas": "#7a7a7a", "log": "#ffffff",
    },
}

Edit = dict[tuple[int, int], RGB | None]  # pixel -> color, None = transparent
PICK_ONLY = {"reference", "shiny_result", "shiny_reference"}  # panels you can pick colors from but not paint on


def _hex(rgb: RGB) -> str:
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def _key(rgb: RGB) -> str:
    return f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"


def _parse(text: str) -> RGB | None:
    parts = text.split(",")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _lightness(rgb: RGB) -> float:
    return float(rgb_to_lch(np.array(rgb, np.uint8))[0])


def apply_source_edits(sprite: SpriteImage, edits: Edit) -> None:
    """Paint edits onto the loaded source before the pipeline sees it."""
    if not edits:
        return
    h, w = sprite.opaque_mask.shape
    for (x, y), rgb in edits.items():
        if not (0 <= x < w and 0 <= y < h):
            continue
        if rgb is None:
            sprite.opaque_mask[y, x] = False
            sprite.rgba[y, x] = (0, 0, 0, 0)
        else:
            sprite.opaque_mask[y, x] = True
            sprite.rgba[y, x, :3] = rgb
            sprite.rgba[y, x, 3] = 255
    if sprite.opaque_mask.any():
        ys, xs = np.nonzero(sprite.opaque_mask)
        sprite.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    sprite.image = Image.fromarray(sprite.rgba, "RGBA")


def apply_result_edits(final: np.ndarray, edits: Edit) -> np.ndarray:
    out = final.copy()
    h, w = out.shape[:2]
    for (x, y), rgb in edits.items():
        if not (0 <= x < w and 0 <= y < h):
            continue
        if rgb is None:
            out[y, x] = (0, 0, 0, 0)
        else:
            out[y, x, :3] = rgb
            out[y, x, 3] = 255
    return out


def flood_region(rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """4-connected region of identical color (or of transparency) containing (x, y)."""
    same = np.all(rgba == rgba[y, x], axis=-1)
    labels, _ = ndimage.label(same, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    return labels == labels[y, x]


class RevampApp(tk.Tk):
    def __init__(self, input_path: Path | None = None) -> None:
        super().__init__()
        self.title("PokeRevamp Assistant")
        self.geometry("1440x900")
        self.minsize(1150, 720)

        self.theme_name = tk.StringVar(value="dark")
        self.input_path: Path | None = None
        self.config_model = RevampConfig()
        # Animation frames: one outcome / edit set per frame; `outcome` etc. refer to the current frame.
        self.frame_sheet: Path | None = None
        self.frame_size: int | None = None
        self.frame_count = 1
        self.current_frame = 0
        self.frames_selected: set[int] = {0}
        self.frame_select_vars: dict[int, tk.BooleanVar] = {}
        self.view_all = tk.BooleanVar(value=False)
        self.frame_outcomes: dict[int, RevampOutcome] = {}
        self.frame_edited_source: dict[int, np.ndarray] = {}
        self.frame_source_edits: dict[int, Edit] = {0: {}}
        self.frame_result_edits: dict[int, Edit] = {0: {}}
        self.shiny_outcome: RevampOutcome | None = None
        self.reference_mode = tk.StringVar(value="auto")
        self.reference_path: Path | None = None
        self.shiny = tk.BooleanVar(value=False)
        self.zoom = tk.IntVar(value=6)
        self.fit_zoom = tk.BooleanVar(value=True)
        self.bg_mode = tk.StringVar(value="dark")
        self.stage_name = tk.StringVar(value=STAGES[0][0])
        self.style_var = tk.StringVar(value="emerald")
        self.kind_var = tk.StringVar(value="pokemon")
        self.tool = tk.StringVar(value="pencil")  # pick | pencil | fill | erase | erase_fill
        self.brush: RGB | None = (0, 0, 0)
        self.brush_is_transparent = False
        self.undo_stack: list[list[tuple[int, str, tuple[int, int], RGB | None, bool]]] = []
        self._group: list[tuple[int, str, tuple[int, int], RGB | None, bool]] | None = None
        self.effect_vars: dict[str, tk.BooleanVar] = {}
        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.color_rows: dict[str, dict] = {}
        self._tk_widgets: list[tuple[tk.Widget, str]] = []  # (widget, role) for theming plain tk widgets
        self._photo: ImageTk.PhotoImage | None = None
        self._panel_boxes: list[tuple] = []
        self._pending: str | None = None
        self._running = False
        self._dirty = False
        self._results: queue.Queue = queue.Queue()
        self._drag_painted: set[tuple[str, int, int]] = set()

        self._build_ui()
        self.apply_theme()
        self.after(50, self._poll)
        if input_path is not None:
            self.load_sprite_file(input_path)

    # ------------------------------------------------------------------ current-frame accessors
    @property
    def outcome(self) -> RevampOutcome | None:
        return self.frame_outcomes.get(self.current_frame)

    @property
    def edited_source(self) -> np.ndarray | None:
        return self.frame_edited_source.get(self.current_frame)

    @property
    def source_edits(self) -> Edit:
        return self.frame_source_edits.setdefault(self.current_frame, {})

    @property
    def result_edits(self) -> Edit:
        return self.frame_result_edits.setdefault(self.current_frame, {})

    def display_final_for(self, frame: int) -> np.ndarray | None:
        oc = self.frame_outcomes.get(frame)
        if oc is None:
            return None
        return apply_result_edits(oc.final, self.frame_result_edits.get(frame, {}))

    @property
    def display_final(self) -> np.ndarray | None:
        return self.display_final_for(self.current_frame)

    # ------------------------------------------------------------------ theming
    def _reg(self, widget: tk.Widget, role: str) -> tk.Widget:
        self._tk_widgets.append((widget, role))
        return widget

    def apply_theme(self) -> None:
        t = THEMES[self.theme_name.get()]
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure(".", background=t["bg"], foreground=t["fg"], fieldbackground=t["field"], bordercolor=t["border"],
                     lightcolor=t["panel"], darkcolor=t["bg"], troughcolor=t["field"], focuscolor=t["accent"])
        st.configure("TFrame", background=t["bg"])
        st.configure("TLabel", background=t["bg"], foreground=t["fg"])
        st.configure("Muted.TLabel", foreground=t["muted"])
        st.configure("TLabelframe", background=t["bg"], bordercolor=t["border"])
        st.configure("TLabelframe.Label", background=t["bg"], foreground=t["accent"])
        st.configure("TButton", background=t["panel"], foreground=t["fg"], bordercolor=t["border"], padding=3)
        st.map("TButton", background=[("active", t["select"]), ("pressed", t["accent"])])
        st.configure("Tool.TButton", padding=2)
        st.configure("Selected.TButton", background=t["accent"], foreground="#ffffff")
        st.map("Selected.TButton", background=[("active", t["accent"])])
        st.configure("TCheckbutton", background=t["bg"], foreground=t["fg"])
        st.map("TCheckbutton", background=[("active", t["bg"])], indicatorcolor=[("selected", t["accent"])])
        st.configure("TRadiobutton", background=t["bg"], foreground=t["fg"])
        st.map("TRadiobutton", background=[("active", t["bg"])], indicatorcolor=[("selected", t["accent"])])
        st.configure("TEntry", fieldbackground=t["field"], foreground=t["fg"], insertcolor=t["fg"])
        st.configure("TCombobox", fieldbackground=t["field"], foreground=t["fg"], background=t["panel"], arrowcolor=t["fg"])
        st.map("TCombobox", fieldbackground=[("readonly", t["field"])], foreground=[("readonly", t["fg"])])
        st.configure("TSpinbox", fieldbackground=t["field"], foreground=t["fg"], background=t["panel"], arrowcolor=t["fg"])
        st.configure("TScale", background=t["bg"], troughcolor=t["field"])
        st.configure("TScrollbar", background=t["panel"], troughcolor=t["bg"], arrowcolor=t["fg"])
        st.configure("TNotebook", background=t["bg"])
        st.configure("TNotebook.Tab", background=t["panel"], foreground=t["fg"])
        st.map("TNotebook.Tab", background=[("selected", t["bg"])])
        self.configure(bg=t["bg"])
        self.option_add("*TCombobox*Listbox.background", t["field"])
        self.option_add("*TCombobox*Listbox.foreground", t["fg"])
        self.option_add("*TCombobox*Listbox.selectBackground", t["select"])
        for w, role in self._tk_widgets:
            if not w.winfo_exists():
                continue
            if role == "canvas":
                w.configure(bg=t["canvas"])
            elif role == "list":
                w.configure(bg=t["field"], fg=t["fg"], selectbackground=t["select"], selectforeground=t["fg"], highlightbackground=t["border"])
            elif role == "text":
                w.configure(bg=t["log"], fg=t["fg"], insertbackground=t["fg"], highlightbackground=t["border"])
            elif role == "frame":
                w.configure(bg=t["bg"])
            elif role == "swatch":
                w.configure(highlightbackground=t["border"])
        self.redraw()

    def toggle_theme(self) -> None:
        self.theme_name.set("light" if self.theme_name.get() == "dark" else "dark")
        if self.bg_mode.get() in ("dark", "light"):
            self.bg_mode.set(self.theme_name.get())
        self.apply_theme()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left_outer = ttk.Frame(root, width=400)
        left_outer.grid(row=0, column=0, sticky="nsw")
        left_outer.grid_propagate(False)
        canvas = self._reg(tk.Canvas(left_outer, highlightthickness=0, width=380), "frame")
        vsb = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(canvas)
        left_id = canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(left_id, width=e.width))

        def _wheel(e: tk.Event) -> None:
            x, y = self.winfo_pointerxy()
            w = self.winfo_containing(x, y)
            while w is not None:
                if w is canvas or w is left:
                    canvas.yview_scroll(int(-e.delta / 120), "units")
                    return
                if w is self.preview:
                    if e.state & 0x1:  # Shift held -> horizontal
                        self.preview.xview_scroll(int(-e.delta / 120), "units")
                    else:
                        self.preview.yview_scroll(int(-e.delta / 120), "units")
                    return
                w = w.master  # type: ignore[assignment]

        self.bind_all("<MouseWheel>", _wheel)
        self.bind_all("<Shift-MouseWheel>", lambda e: self.preview.xview_scroll(int(-e.delta / 120), "units"))
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Control-Z>", lambda e: self.undo())
        # Tool shortcuts (only when not typing in an entry).
        for key, tool in (("i", "pick"), ("b", "pencil"), ("g", "fill"), ("e", "erase"), ("x", "erase_fill")):
            self.bind_all(f"<KeyPress-{key}>", lambda ev, t=tool: None if isinstance(ev.widget, (ttk.Entry, tk.Entry, tk.Text)) else self.set_tool(t))

        # Sprite
        f = ttk.LabelFrame(left, text="Sprite", padding=6)
        f.pack(fill="x", pady=(0, 6))
        ttk.Button(f, text="Open sprite…", command=self.open_sprite).grid(row=0, column=0, sticky="w")
        self.file_label = ttk.Label(f, text="(none)", width=30)
        self.file_label.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(f, text="Kind").grid(row=1, column=0, sticky="w", pady=(4, 0))
        kind = ttk.Combobox(f, textvariable=self.kind_var, values=["pokemon", "trainer"], state="readonly", width=10)
        kind.grid(row=1, column=1, sticky="w", padx=6, pady=(4, 0))
        kind.bind("<<ComboboxSelected>>", lambda e: (self._refresh_ref_list(), self.schedule()))
        ttk.Label(f, text="Style").grid(row=2, column=0, sticky="w", pady=(4, 0))
        style = ttk.Combobox(f, textvariable=self.style_var, values=["emerald", "frlg", "gen3-mixed"], state="readonly", width=10)
        style.grid(row=2, column=1, sticky="w", padx=6, pady=(4, 0))
        style.bind("<<ComboboxSelected>>", lambda e: (self._refresh_ref_list(), self.schedule()))
        ttk.Button(f, text="Theme", command=self.toggle_theme, style="Tool.TButton").grid(row=0, column=2, sticky="e")

        # Source library: every GB/GBC sprite the bootstrap copied, ready to open.
        f = ttk.LabelFrame(left, text="Source library  (sprites from the games)", padding=6)
        f.pack(fill="x", pady=(0, 6))
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.lib_game = tk.StringVar(value="crystal")
        game = ttk.Combobox(row, textvariable=self.lib_game, values=["all", "rb", "yellow", "gold", "silver", "crystal"], state="readonly", width=8)
        game.pack(side="left")
        game.bind("<<ComboboxSelected>>", lambda e: self._refresh_library())
        self.lib_kind = tk.StringVar(value="pokemon")
        lk = ttk.Combobox(row, textvariable=self.lib_kind, values=["pokemon", "trainer"], state="readonly", width=8)
        lk.pack(side="left", padx=4)
        lk.bind("<<ComboboxSelected>>", lambda e: self._refresh_library())
        self.lib_filter = tk.StringVar()
        self.lib_filter.trace_add("write", lambda *a: self._refresh_library())
        ttk.Entry(row, textvariable=self.lib_filter, width=14).pack(side="left", padx=(4, 0))
        self.lib_list = self._reg(tk.Listbox(f, height=7, exportselection=False, relief="flat"), "list")
        self.lib_list.pack(fill="x", pady=(4, 0))
        self.lib_list.bind("<Double-Button-1>", lambda e: self._open_library_selection())
        self.lib_list.bind("<Return>", lambda e: self._open_library_selection())
        lrow = ttk.Frame(f)
        lrow.pack(fill="x", pady=(4, 0))
        ttk.Button(lrow, text="Open selected", style="Tool.TButton", command=self._open_library_selection).pack(side="left")
        self.lib_label = ttk.Label(lrow, text="double-click a sprite to open it", style="Muted.TLabel")
        self.lib_label.pack(side="left", padx=8)

        # Reference
        f = ttk.LabelFrame(left, text="Gen III reference", padding=6)
        f.pack(fill="x", pady=(0, 6))
        for text, val in (("Auto (same species, else shape-alikes)", "auto"), ("None (keep source colors)", "none"), ("Pick from list:", "pick")):
            ttk.Radiobutton(f, text=text, value=val, variable=self.reference_mode, command=self.schedule).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x")
        self.ref_filter = tk.StringVar()
        self.ref_filter.trace_add("write", lambda *a: self._refresh_ref_list())
        ttk.Entry(row, textvariable=self.ref_filter, width=18).pack(side="left")
        ttk.Button(row, text="Browse file…", command=self.browse_reference).pack(side="left", padx=4)
        ttk.Button(row, text="Download refs…", command=self.download_references).pack(side="left")
        self.ref_list = self._reg(tk.Listbox(f, height=5, exportselection=False, relief="flat"), "list")
        self.ref_list.pack(fill="x", pady=(4, 0))
        self.ref_list.bind("<<ListboxSelect>>", self._on_ref_selected)
        self.shiny_check = ttk.Checkbutton(f, text="Also show the shiny version (second preview row)", variable=self.shiny, command=self.schedule)
        self.shiny_check.pack(anchor="w", pady=(4, 0))
        self.ref_label = ttk.Label(f, text="", style="Muted.TLabel")
        self.ref_label.pack(anchor="w")

        # Paint palettes (the tools themselves live in the toolbar above the preview)
        f = ttk.LabelFrame(left, text="Brush colors", padding=6)
        f.pack(fill="x", pady=(0, 6))
        ttk.Label(f, text="Reference colors (one row per region, dark → light):", style="Muted.TLabel").pack(anchor="w")
        self.palette_frame = ttk.Frame(f)
        self.palette_frame.pack(fill="x")
        ttk.Label(f, text="Source colors:", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        self.src_palette_frame = ttk.Frame(f)
        self.src_palette_frame.pack(fill="x")
        erow = ttk.Frame(f)
        erow.pack(fill="x", pady=(6, 0))
        self.edit_label = ttk.Label(erow, text="edits: source 0, result 0", style="Muted.TLabel")
        self.edit_label.pack(side="left")
        ttk.Button(erow, text="Clear result edits", style="Tool.TButton", command=lambda: self.clear_edits("result")).pack(side="right")
        ttk.Button(erow, text="Clear source edits", style="Tool.TButton", command=lambda: self.clear_edits("source")).pack(side="right", padx=3)
        ttk.Label(f, text="Paint on the Source or the Result panel. Pixels you paint on the Source are kept exactly as painted. Right-click a painted pixel to undo just that edit. Keys: I pick, B pencil, G fill, E eraser, X erase region, Ctrl+Z undo.", style="Muted.TLabel", wraplength=340).pack(anchor="w", pady=(4, 0))

        # Color map
        f = ttk.LabelFrame(left, text="Color mapping  (whole source color → target)", padding=6)
        f.pack(fill="x", pady=(0, 6))
        self.colors_body = ttk.Frame(f)
        self.colors_body.pack(fill="x")
        ttk.Button(f, text="Clear all overrides", style="Tool.TButton", command=self.clear_color_map).pack(anchor="w", pady=(4, 0))

        # Effects
        f = ttk.LabelFrame(left, text="Effects", padding=6)
        f.pack(fill="x", pady=(0, 6))
        for text, field in EFFECTS:
            var = tk.BooleanVar(value=bool(getattr(self.config_model, field)))
            self.effect_vars[field] = var
            ttk.Checkbutton(f, text=text, variable=var, command=self.schedule).pack(anchor="w")

        # Tuning
        f = ttk.LabelFrame(left, text="Tuning", padding=6)
        f.pack(fill="x", pady=(0, 6))
        f.columnconfigure(1, weight=1)
        for i, (text, field, lo, hi) in enumerate(SLIDERS):
            var = tk.DoubleVar(value=float(getattr(self.config_model, field)))
            self.slider_vars[field] = var
            ttk.Label(f, text=text, width=16).grid(row=i, column=0, sticky="w")
            ttk.Scale(f, from_=lo, to=hi, variable=var, command=lambda v: self.schedule()).grid(row=i, column=1, sticky="ew", padx=4)
            lbl = ttk.Label(f, text=f"{var.get():.2f}", width=5)
            lbl.grid(row=i, column=2)
            var.trace_add("write", lambda *a, l=lbl, v=var: l.configure(text=f"{v.get():.2f}"))
        ttk.Button(f, text="Reset tuning", style="Tool.TButton", command=self.reset_tuning).grid(row=len(SLIDERS), column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Output
        f = ttk.LabelFrame(left, text="Output", padding=6)
        f.pack(fill="x", pady=(0, 6))
        ttk.Button(f, text="Save to output/", command=self.save_result).pack(side="left")
        ttk.Button(f, text="Save session…", command=self.save_preset).pack(side="left", padx=4)
        ttk.Button(f, text="Load session…", command=self.load_preset).pack(side="left")

        # Right: toolbar (tools + brush), view bar, preview
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)
        tools = ttk.Frame(right)
        tools.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.tool_buttons: dict[str, ttk.Button] = {}
        for text, val in (("Pick (I)", "pick"), ("Pencil (B)", "pencil"), ("Fill region (G)", "fill"), ("Eraser (E)", "erase"), ("Erase region (X)", "erase_fill")):
            b = ttk.Button(tools, text=text, style="Tool.TButton", command=lambda v=val: self.set_tool(v))
            b.pack(side="left", padx=(0, 3))
            self.tool_buttons[val] = b
        ttk.Button(tools, text="Undo (Ctrl+Z)", style="Tool.TButton", command=self.undo).pack(side="left", padx=(10, 0))
        ttk.Separator(tools, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(tools, text="Brush").pack(side="left")
        self.brush_swatch = self._reg(tk.Canvas(tools, width=26, height=20, highlightthickness=1), "swatch")
        self.brush_swatch.pack(side="left", padx=4)
        self.brush_label = ttk.Label(tools, text="0,0,0", style="Muted.TLabel", width=12)
        self.brush_label.pack(side="left")
        ttk.Button(tools, text="Custom…", style="Tool.TButton", command=self.custom_brush).pack(side="left", padx=3)
        ttk.Button(tools, text="Transparent", style="Tool.TButton", command=lambda: self.set_brush(None)).pack(side="left")

        bar = ttk.Frame(right)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(bar, text="Zoom").pack(side="left")
        ttk.Spinbox(bar, from_=2, to=14, textvariable=self.zoom, width=4, command=lambda: (self.fit_zoom.set(False), self.redraw())).pack(side="left", padx=(2, 4))
        ttk.Checkbutton(bar, text="fit", variable=self.fit_zoom, command=self.redraw).pack(side="left", padx=(0, 10))
        ttk.Label(bar, text="Background").pack(side="left")
        for text, val in (("checker", "checker"), ("dark", "dark"), ("light", "light")):
            ttk.Radiobutton(bar, text=text, value=val, variable=self.bg_mode, command=self.redraw).pack(side="left")
        ttk.Label(bar, text="   Show").pack(side="left")
        st = ttk.Combobox(bar, textvariable=self.stage_name, values=[s[0] for s in STAGES], state="readonly", width=24)
        st.pack(side="left", padx=4)
        st.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        self.status = ttk.Label(bar, text="", style="Muted.TLabel")
        self.status.pack(side="right")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.frame_prev = ttk.Button(bar, text="◀", width=3, style="Tool.TButton", command=lambda: self.set_frame(self.current_frame - 1))
        self.frame_prev.pack(side="left")
        self.frame_label = ttk.Label(bar, text="Frame 1/1", width=11, anchor="center")
        self.frame_label.pack(side="left")
        self.frame_next = ttk.Button(bar, text="▶", width=3, style="Tool.TButton", command=lambda: self.set_frame(self.current_frame + 1))
        self.frame_next.pack(side="left")
        self.view_all_check = ttk.Checkbutton(bar, text="All frames", variable=self.view_all, command=self.redraw)
        self.view_all_check.pack(side="left", padx=(6, 0))
        self.copy_edits_button = ttk.Button(bar, text="Edits → all", style="Tool.TButton", command=self.copy_edits_to_all_frames)
        self.copy_edits_button.pack(side="left", padx=(6, 0))
        self.frame_select_frame = ttk.Frame(right)
        self.frame_select_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.preview = self._reg(tk.Canvas(right, highlightthickness=0), "canvas")
        self.preview.grid(row=3, column=0, sticky="nsew")
        pv_y = ttk.Scrollbar(right, orient="vertical", command=self.preview.yview)
        pv_y.grid(row=3, column=1, sticky="ns")
        pv_x = ttk.Scrollbar(right, orient="horizontal", command=self.preview.xview)
        pv_x.grid(row=4, column=0, sticky="ew")
        self.preview.configure(yscrollcommand=pv_y.set, xscrollcommand=pv_x.set)
        self.preview.bind("<Button-1>", self._on_press)
        self.preview.bind("<B1-Motion>", self._on_drag)
        self.preview.bind("<ButtonRelease-1>", self._on_release)
        self.preview.bind("<Button-3>", self._on_right_click)
        self.preview.bind("<Motion>", self._on_motion)
        self.preview.bind("<Configure>", lambda e: self.redraw())
        self.hover = ttk.Label(right, text="", style="Muted.TLabel")
        self.hover.grid(row=5, column=0, sticky="w")
        self.log = self._reg(tk.Text(right, height=5, wrap="word", state="disabled", font=("Consolas", 9), relief="flat"), "text")
        self.log.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        self.set_tool("pencil")
        self._refresh_ref_list()
        self._refresh_library()
        self._update_brush_ui()
        self._update_frame_ui()

    # ------------------------------------------------------------------ config
    def current_config(self) -> RevampConfig:
        updates = {field: bool(var.get()) for field, var in self.effect_vars.items()}
        updates.update({field: float(var.get()) for field, var in self.slider_vars.items()})
        updates["style"] = self.style_var.get()
        updates["kind"] = self.kind_var.get()
        updates["auto_reference"] = self.reference_mode.get() == "auto"
        updates["color_map"] = dict(self.config_model.color_map)
        cfg = self.config_model.model_copy(update=updates)
        if self.reference_mode.get() == "none":
            cfg = cfg.model_copy(update={"auto_reference": False, "palette_mode": "source-expanded"})
        return cfg

    def _base_ref(self) -> Path | None:
        """The reference the run is based on: the picked one, or the same-species one in Auto."""
        mode = self.reference_mode.get()
        if mode == "pick" and self.reference_path:
            return normal_variant(self.reference_path)
        if mode == "auto" and self.input_path is not None:
            name = canonical_name_from_path(self.input_path)
            cands = [e for e in list_references(kind=self.kind_var.get(), era="gba") if e.name == name and e.matches_style(self.style_var.get())]
            if cands:
                return cands[0].path
        return None

    def _refs(self) -> list[Path]:
        """Explicit reference list for the normal run (empty lets the pipeline rank shape-alikes)."""
        if self.reference_mode.get() == "pick" and self.reference_path:
            return [normal_variant(self.reference_path)]
        return []

    def _frame_refs(self, refs: list[Path], frame: int) -> list[Path]:
        """Reference list for one source frame: the n-th chosen frame uses the n-th reference frame."""
        if self.frame_count <= 1:
            return refs
        base = refs[0] if refs else self._base_ref()
        if base is None:
            return refs
        n_ref = frame_count(base)
        if n_ref <= 1:
            return refs
        chosen = sorted(self.frames_selected)
        idx = chosen.index(frame) if frame in chosen else 0
        return [frame_ref(base, min(idx, n_ref - 1))] + refs[1:]

    def _shiny_refs(self) -> list[Path]:
        """Shiny sibling of the base reference, when the toggle is on and one exists."""
        if not self.shiny.get() or self.reference_mode.get() == "none":
            return []
        base = self._base_ref()
        shiny = shiny_variant(base) if base is not None else None
        return [shiny] if shiny is not None else []

    def reset_tuning(self) -> None:
        d = RevampConfig()
        for field, var in self.slider_vars.items():
            var.set(float(getattr(d, field)))
        self.schedule()

    # ------------------------------------------------------------------ files
    def open_sprite(self) -> None:
        start = project_root() / "input"
        p = filedialog.askopenfilename(title="Open sprite", initialdir=str(start if start.exists() else project_root()), filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if p:
            self.load_sprite_file(Path(p))

    def load_sprite_file(self, path: Path) -> None:
        try:
            load_sprite(path)
        except SpriteLoadError as exc:
            messagebox.showerror("Cannot open sprite", str(exc))
            return
        self.input_path = path
        self.file_label.configure(text=path.name)
        self.kind_var.set(infer_kind(path, RevampConfig()))
        self.config_model = self.config_model.model_copy(update={"color_map": {}})
        # Frames: a sibling <stem>_frames.png or a tall frame-sheet PNG (frames stacked vertically).
        self.frame_sheet, self.frame_size, self.frame_count = None, None, 1
        sheet = find_frame_sheet(path)
        if sheet is not None:
            try:
                frames = load_frames(sheet)
            except SpriteLoadError:
                frames = []
            if len(frames) > 1:
                self.frame_sheet = sheet
                self.frame_size = frames[0].size[0]
                self.frame_count = len(frames)
        self.current_frame = 0
        self.frames_selected = set(range(self.frame_count))
        self.view_all.set(self.frame_count > 1)
        self.frame_outcomes, self.frame_edited_source = {}, {}
        self.frame_source_edits = {i: {} for i in range(self.frame_count)}
        self.frame_result_edits = {i: {} for i in range(self.frame_count)}
        self.undo_stack = []
        self.title(f"PokeRevamp Assistant — {path.name}" + (f"  ({self.frame_count} frames)" if self.frame_count > 1 else ""))
        self._refresh_ref_list()
        self._update_edit_label()
        self._update_frame_ui()
        self.schedule()

    def _load_frame_sprite(self, frame: int, kind: str):
        """Fresh SpriteImage for one frame (the pipeline mutates its mask, so never cache)."""
        if self.frame_sheet is not None:
            frames = load_frames(self.frame_sheet, kind=kind, frame_size=self.frame_size)  # type: ignore[arg-type]
            if frame < len(frames):
                return frames[frame]
        return load_sprite(self.input_path, kind=kind)  # type: ignore[arg-type]

    def _refresh_library(self) -> None:
        game = self.lib_game.get()
        entries = list_references(kind=self.lib_kind.get(), game=None if game == "all" else game, era="gbc")
        flt = self.lib_filter.get().strip().lower()
        self._library = [e for e in entries if flt in e.name.lower()]
        self.lib_list.delete(0, "end")
        for e in self._library:
            anim = " +" if e.path.with_name(e.path.stem + "_frames.png").exists() else ""
            self.lib_list.insert("end", f"{e.game}: {e.name}{anim}")
        n = len(self._library)
        self.lib_label.configure(text=f"{n} sprites — + = has extra frames" if n else "none found — run Download refs…")

    def _open_library_selection(self) -> None:
        sel = self.lib_list.curselection()
        if not sel:
            return
        e = self._library[int(sel[0])]
        self.load_sprite_file(e.path)

    def browse_reference(self) -> None:
        start = project_root() / "data" / "references" / "gba"
        p = filedialog.askopenfilename(title="Pick reference PNG", initialdir=str(start if start.exists() else project_root()), filetypes=[("PNG", "*.png")])
        if p:
            self.reference_path = Path(p)
            self.reference_mode.set("pick")
            self.ref_label.configure(text=self.reference_path.name)
            self.schedule()

    def download_references(self) -> None:
        if not messagebox.askyesno("Download references", "Clone the pret decompilation repositories into vendor/ and copy sprites into data/?\nThis needs Git and may take several minutes."):
            return
        self.status.configure(text="downloading references…")

        def work() -> None:
            from gbc_to_gba_pokerevamp.assets import BootstrapError, bootstrap

            try:
                summary = bootstrap()
                self._results.put(("bootstrap", json.dumps(summary, indent=1)))
            except BootstrapError as exc:
                self._results.put(("bootstrap", f"ERROR: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _refresh_ref_list(self) -> None:
        kind = self.kind_var.get()
        style = self.style_var.get()
        refs = [e for e in list_references(kind=kind, era="gba") if e.matches_style(style)]
        flt = self.ref_filter.get().strip().lower()
        self.ref_list.delete(0, "end")
        self._visible_refs = [e for e in refs if flt in e.name.lower()]
        for e in self._visible_refs:
            self.ref_list.insert("end", f"{e.game}: {e.name}")
        if self.input_path is not None:
            name = canonical_name_from_path(self.input_path)
            for i, e in enumerate(self._visible_refs):
                if e.name == name:
                    self.ref_list.selection_clear(0, "end")
                    self.ref_list.selection_set(i)
                    self.ref_list.see(i)
                    break

    def _on_ref_selected(self, _event=None) -> None:
        sel = self.ref_list.curselection()
        if not sel:
            return
        e = self._visible_refs[int(sel[0])]
        self.reference_path = e.path
        self.reference_mode.set("pick")
        self.ref_label.configure(text=e.path.name)
        self.schedule()

    # ------------------------------------------------------------------ pipeline
    def schedule(self) -> None:
        if self.input_path is None:
            return
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(120, self.run_now)

    def run_now(self) -> None:
        self._pending = None
        if self.input_path is None:
            return
        if self._running:
            self._dirty = True
            return
        self._running = True
        self.status.configure(text="working…")
        config = self.current_config()
        refs = self._refs()
        shiny_refs = self._shiny_refs()
        current = self.current_frame
        all_edits = {k: dict(v) for k, v in self.frame_source_edits.items()}
        n_frames = self.frame_count
        frame_refs = {k: self._frame_refs(refs, k) for k in range(n_frames)}
        frame_shiny_refs = {k: self._frame_refs(shiny_refs, k) if shiny_refs else [] for k in range(n_frames)}

        def work() -> None:
            try:
                # Every frame is placed with frame 0's centering so the animation does not jitter.
                cfg = config
                if n_frames > 1:
                    f0 = self._load_frame_sprite(0, config.kind)
                    apply_source_edits(f0, all_edits.get(0, {}))
                    x0, y0, x1, y1 = f0.bbox
                    origin = ((config.canvas_size - (x1 - x0)) // 2 - x0, (config.canvas_size - (y1 - y0)) // 2 - y0)
                    cfg = config.model_copy(update={"canvas_origin": origin})
                # Current frame first so the preview responds quickly; the rest follow.
                order = [current] + [i for i in range(n_frames) if i != current]
                for k in order:
                    edits = all_edits.get(k, {})
                    sprite = self._load_frame_sprite(k, cfg.kind)
                    apply_source_edits(sprite, edits)
                    edited = sprite.rgba.copy()
                    pinned = {pos: rgb for pos, rgb in edits.items() if rgb is not None}
                    outcome = revamp_sprite(sprite, cfg, frame_refs[k], input_name=f"{self.input_path} [frame {k}]", pinned_pixels=pinned)
                    shiny_outcome = None
                    if k == current and frame_shiny_refs[k]:
                        sprite2 = self._load_frame_sprite(k, cfg.kind)
                        apply_source_edits(sprite2, edits)
                        shiny_outcome = revamp_sprite(sprite2, cfg, frame_shiny_refs[k], input_name=str(self.input_path), pinned_pixels=pinned)
                    self._results.put(("frame", (k, outcome, edited, shiny_outcome, k == current)))
                self._results.put(("done", None))
            except (SpriteLoadError, RevampError, ValueError, IndexError) as exc:  # shown in the log, never crash the UI
                self._results.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self._results.get_nowait()
                if kind == "frame":
                    self._frame_finished(*payload)
                elif kind == "done":
                    self._running = False
                    if self._dirty:
                        self._dirty = False
                        self.schedule()
                elif kind == "error":
                    self._running = False
                    self.status.configure(text="error")
                    self._log(f"ERROR: {payload}")
                elif kind == "bootstrap":
                    self.status.configure(text="references ready")
                    self._log(payload)
                    self._refresh_ref_list()
                    self.schedule()
        except queue.Empty:
            pass
        try:
            self.after(50, self._poll)
        except tk.TclError:
            pass  # window destroyed

    def _frame_finished(self, frame: int, outcome: RevampOutcome, edited_source: np.ndarray, shiny_outcome: RevampOutcome | None, is_current: bool) -> None:
        self.frame_outcomes[frame] = outcome
        self.frame_edited_source[frame] = edited_source
        if is_current:
            self.shiny_outcome = shiny_outcome
            rep = outcome.report
            frame_note = f"  frame {frame + 1}/{self.frame_count}" if self.frame_count > 1 else ""
            self.status.configure(text=f"{rep.source_opaque_colors} → {rep.target_opaque_colors} colors" + ("  + shiny" if shiny_outcome else "") + frame_note)
            warn = list(outcome.warnings)
            if self.shiny.get() and shiny_outcome is None and self.reference_mode.get() != "none":
                warn.append("no shiny palette available for this reference")
            self._log("\n".join(warn) if warn else "(no warnings)")
            self._rebuild_color_rows()
            self._rebuild_palettes()
            self.redraw()
        elif self.view_all.get():
            self.redraw()

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("1.0", text)
        self.log.configure(state="disabled")

    # ------------------------------------------------------------------ preview
    def _frame_images(self, frame: int) -> tuple[Image.Image, Image.Image]:
        """(source, result) images for one frame at the selected stage."""
        oc = self.frame_outcomes[frame]
        src = self.frame_edited_source.get(frame, oc.sprite.rgba)
        stage_key = dict(STAGES).get(self.stage_name.get(), "09_final")
        if stage_key == "09_final":
            result = self.display_final_for(frame)
            if result is None:
                result = oc.final
        else:
            result = oc.stages.get(stage_key, oc.final)
        return Image.fromarray(np.ascontiguousarray(src), "RGBA"), Image.fromarray(np.ascontiguousarray(result), "RGBA")

    def _panel_rows(self) -> list[list[tuple[str, int, Image.Image, str]]]:
        """Rows of (panel kind, frame, image, label).

        Each row is Source | Result for one frame, followed by that frame's reference (shown
        whenever it differs from the row above). The all-frames view shows one row per
        selected frame; otherwise only the current frame.
        """
        assert self.outcome is not None
        rows: list[list[tuple[str, int, Image.Image, str]]] = []
        stage = self.stage_name.get()
        result_name = stage if stage != STAGES[0][0] else "Result"
        if self.view_all.get() and self.frame_count > 1:
            shown = [k for k in sorted(self.frames_selected) if k in self.frame_outcomes]
        else:
            shown = [self.current_frame]
        last_ref: Path | None = None
        for k in shown:
            s, r = self._frame_images(k)
            tag = f" {k + 1}" if self.frame_count > 1 else ""
            cur = " <" if self.frame_count > 1 and k == self.current_frame and self.view_all.get() else ""
            row = [("source", k, s, f"Source{tag}{cur}"), ("result", k, r, f"{result_name}{tag}{cur}")]
            refs = self.frame_outcomes[k].used_refs
            if refs and refs[0] != last_ref:
                try:
                    row.append(("reference", k, load_sprite(refs[0]).image, f"Reference: {refs[0].name}"))
                    last_ref = refs[0]
                except SpriteLoadError:
                    pass
            rows.append(row)
        if self.shiny_outcome is not None:
            so = self.shiny_outcome
            stage_key = dict(STAGES).get(stage, "09_final")
            shiny_result = so.stages.get(stage_key, so.final)
            row2: list[tuple[str, int, Image.Image, str]] = [
                ("shiny_result", self.current_frame, Image.fromarray(np.ascontiguousarray(shiny_result), "RGBA"), f"Shiny result{' ' + str(self.current_frame + 1) if self.frame_count > 1 else ''} (pick only)"),
            ]
            if so.used_refs:
                try:
                    row2.append(("shiny_reference", -1, load_sprite(so.used_refs[0]).image, f"Shiny reference: {so.used_refs[0].stem}"))
                except SpriteLoadError:
                    pass
            rows.append(row2)
        return rows

    # ------------------------------------------------------------------ frames
    def _update_frame_ui(self) -> None:
        multi = self.frame_count > 1
        self.frame_label.configure(text=f"Frame {self.current_frame + 1}/{self.frame_count}")
        state = "normal" if multi else "disabled"
        for w in (self.frame_prev, self.frame_next, self.copy_edits_button, self.view_all_check):
            w.configure(state=state)
        # Output-frame checkboxes.
        for child in self.frame_select_frame.winfo_children():
            child.destroy()
        self.frame_select_vars = {}
        if multi:
            ttk.Label(self.frame_select_frame, text="Frames in the output:", style="Muted.TLabel").pack(side="left")
            for k in range(self.frame_count):
                var = tk.BooleanVar(value=k in self.frames_selected)
                self.frame_select_vars[k] = var
                ttk.Checkbutton(self.frame_select_frame, text=str(k + 1), variable=var, command=lambda kk=k: self._toggle_frame_selected(kk)).pack(side="left", padx=(4, 0))
            ttk.Button(self.frame_select_frame, text="all", style="Tool.TButton", width=4, command=lambda: self._select_frames(set(range(self.frame_count)))).pack(side="left", padx=(8, 0))
            ttk.Button(self.frame_select_frame, text="1+2", style="Tool.TButton", width=4, command=lambda: self._select_frames({0, min(1, self.frame_count - 1)})).pack(side="left", padx=(3, 0))
            ttk.Label(self.frame_select_frame, text="   unselected frames are hidden and left out of the saved sheet", style="Muted.TLabel").pack(side="left")

    def _toggle_frame_selected(self, k: int) -> None:
        if self.frame_select_vars[k].get():
            self.frames_selected.add(k)
        else:
            self.frames_selected.discard(k)
        if not self.frames_selected:
            self.frames_selected.add(k)
            self.frame_select_vars[k].set(True)
        self._after_selection_change()

    def _select_frames(self, frames: set[int]) -> None:
        self.frames_selected = set(frames)
        for k, var in self.frame_select_vars.items():
            var.set(k in self.frames_selected)
        self._after_selection_change()

    def _after_selection_change(self) -> None:
        if self.current_frame not in self.frames_selected:
            self.set_frame(min(self.frames_selected))
        else:
            self.redraw()
        self.schedule()  # reference frames follow the chosen-frame order

    def set_frame(self, frame: int) -> None:
        if not (0 <= frame < self.frame_count):
            return
        self.current_frame = frame
        self._update_frame_ui()
        self._update_edit_label()
        if self.outcome is not None:
            self._rebuild_color_rows()
            self._rebuild_palettes()
            self.redraw()
            if self.shiny.get():
                self.schedule()  # the shiny row is computed for the current frame only
        else:
            self.schedule()

    def copy_edits_to_all_frames(self) -> None:
        """Copy this frame's paint edits (pixel-wise) onto every other frame."""
        src = dict(self.source_edits)
        res = dict(self.result_edits)
        for k in range(self.frame_count):
            if k != self.current_frame:
                self.frame_source_edits[k] = dict(src)
                self.frame_result_edits[k] = dict(res)
        self.undo_stack = []
        self.schedule()

    def redraw(self) -> None:
        if not hasattr(self, "preview"):
            return
        self.preview.delete("all")
        self._panel_boxes = []
        if self.outcome is None:
            return
        rows = self._panel_rows()
        ncols = max(len(r) for r in rows)
        pad = 12
        label_h = 18
        z = max(1, int(self.zoom.get()))
        if self.fit_zoom.get():
            avail_w = max(100, self.preview.winfo_width())
            avail_h = max(100, self.preview.winfo_height())
            z = max(1, min((avail_w - pad * (ncols + 1)) // (CANVAS * ncols), (avail_h - (pad + label_h) * len(rows) - pad) // (CANVAS * len(rows))))
            if int(self.zoom.get()) != z:
                self.zoom.set(z)
        cw = CANVAS * z
        row_h = cw + pad + label_h
        total_w = ncols * (cw + pad) + pad
        total_h = len(rows) * row_h + pad
        mode = self.bg_mode.get()
        if mode == "checker":
            yy, xx = np.mgrid[0:total_h, 0:total_w]
            tile = ((yy // 8 + xx // 8) % 2).astype(bool)
            a, b = ((70, 70, 76), (56, 56, 62)) if self.theme_name.get() == "dark" else ((200, 200, 200), (160, 160, 160))
            sheet = Image.fromarray(np.where(tile[..., None], np.array(a, np.uint8), np.array(b, np.uint8)).astype(np.uint8), "RGB").convert("RGBA")
        else:
            sheet = Image.new("RGBA", (total_w, total_h), (40, 40, 46, 255) if mode == "dark" else (236, 236, 236, 255))
        draw = ImageDraw.Draw(sheet)
        text_fill = (230, 230, 230) if (mode == "dark" or (mode == "checker" and self.theme_name.get() == "dark")) else (20, 20, 20)
        for r, panels in enumerate(rows):
            y_top = pad + r * row_h
            # The shiny row lines up under the Result column.
            shiny_row = bool(panels) and panels[0][0] == "shiny_result"
            x = pad + (cw + pad) if shiny_row else pad
            for which, frame, img, label in panels:
                canvas_img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
                ox, oy = (CANVAS - img.width) // 2, (CANVAS - img.height) // 2
                canvas_img.paste(img, (ox, oy), img)
                big = canvas_img.resize((cw, cw), Image.Resampling.NEAREST)
                sheet.alpha_composite(big, (x, y_top + label_h))
                current = which in ("source", "result") and self.view_all.get() and self.frame_count > 1 and frame == self.current_frame
                draw.rectangle([x - 1, y_top + label_h - 1, x + cw, y_top + label_h + cw], outline=(76, 139, 245) if current else (90, 90, 96), width=2 if current else 1)
                draw.text((x, y_top), label, fill=text_fill)
                self._panel_boxes.append((which, frame, x, y_top + label_h, z, ox, oy, img.width, img.height))
                x += cw + pad
        self._photo = ImageTk.PhotoImage(sheet)
        self.preview.create_image(0, 0, anchor="nw", image=self._photo)
        self.preview.configure(scrollregion=(0, 0, total_w, total_h))

    def _hit(self, ex: int, ey: int) -> tuple[str, int, int, int] | None:
        """(panel kind, frame, px, py) under a window coordinate (accounts for scrolling)."""
        ex = int(self.preview.canvasx(ex))
        ey = int(self.preview.canvasy(ey))
        for which, frame, x0, y0, z, ox, oy, w, h in self._panel_boxes:
            px, py = (ex - x0) // z - ox, (ey - y0) // z - oy
            if 0 <= px < w and 0 <= py < h and x0 <= ex < x0 + CANVAS * z and y0 <= ey < y0 + CANVAS * z:
                return which, frame, int(px), int(py)
        return None

    def _panel_array(self, which: str, frame: int | None = None) -> np.ndarray | None:
        if self.outcome is None:
            return None
        f = self.current_frame if frame is None or frame < 0 else frame
        oc = self.frame_outcomes.get(f)
        if which == "source" and oc is not None:
            return self.frame_edited_source.get(f, oc.sprite.rgba)
        if which == "result" and oc is not None:
            res = self.display_final_for(f)
            return res if res is not None else oc.final
        if which == "reference" and oc is not None and oc.used_refs:
            return load_sprite(oc.used_refs[0]).rgba
        if which == "shiny_result" and self.shiny_outcome is not None:
            return self.shiny_outcome.final
        if which == "shiny_reference" and self.shiny_outcome is not None and self.shiny_outcome.used_refs:
            return load_sprite(self.shiny_outcome.used_refs[0]).rgba
        return None

    def _on_motion(self, event: tk.Event) -> None:
        hit = self._hit(event.x, event.y)
        if hit is None:
            self.hover.configure(text="")
            return
        which, frame, px, py = hit
        arr = self._panel_array(which, frame)
        if arr is None:
            return
        p = arr[py, px]
        ftag = f" f{frame + 1}" if frame >= 0 and self.frame_count > 1 else ""
        self.hover.configure(text=f"{which}{ftag} ({px},{py})  " + (f"rgb {p[0]},{p[1]},{p[2]}" if p[3] else "transparent"))

    def _focus_frame(self, frame: int) -> None:
        """Make a clicked frame current without leaving the all-frames view."""
        if frame >= 0 and frame != self.current_frame and frame < self.frame_count:
            self.current_frame = frame
            self._update_frame_ui()
            self._update_edit_label()
            self._rebuild_color_rows()
            self._rebuild_palettes()
            if self.shiny.get():
                self.schedule()

    def _on_press(self, event: tk.Event) -> None:
        hit = self._hit(event.x, event.y)
        if hit is None:
            return
        which, frame, px, py = hit
        tool = self.tool.get()
        if which in PICK_ONLY or tool == "pick":
            arr = self._panel_array(which, frame)
            if arr is not None and arr[py, px, 3]:
                self.set_brush(tuple(int(v) for v in arr[py, px, :3]))  # type: ignore[arg-type]
                if which == "source":
                    self._highlight_color(_key(self.brush))  # type: ignore[arg-type]
            return
        self._focus_frame(frame)
        if tool in ("pencil", "erase"):
            self._group = None
            self._paint(which, [(px, py)], erase=(tool == "erase"))
        elif tool in ("fill", "erase_fill"):
            arr = self._panel_array(which, frame)
            if arr is None:
                return
            region = flood_region(arr, px, py)
            ys, xs = np.nonzero(region)
            self._group = None
            self._paint(which, list(zip(xs.tolist(), ys.tolist())), erase=(tool == "erase_fill"))
            self._group = None

    def _on_drag(self, event: tk.Event) -> None:
        tool = self.tool.get()
        if tool not in ("pencil", "erase"):
            return
        hit = self._hit(event.x, event.y)
        if hit is None:
            return
        which, frame, px, py = hit
        if which in PICK_ONLY or frame != self.current_frame:
            return  # a stroke stays on the frame it started on
        if (which, px, py) in self._drag_painted:
            return
        self._drag_painted.add((which, px, py))
        self._paint(which, [(px, py)], erase=(tool == "erase"))

    def _record(self, which: str, pos: tuple[int, int], old: RGB | None, had: bool) -> None:
        if self._group is None:
            self._group = []
            self.undo_stack.append(self._group)
        self._group.append((self.current_frame, which, pos, old, had))

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_painted.clear()
        self._group = None

    def _on_right_click(self, event: tk.Event) -> None:
        hit = self._hit(event.x, event.y)
        if hit is None or hit[0] in PICK_ONLY:
            return
        which, frame, px, py = hit
        self._focus_frame(frame)
        edits = self.source_edits if which == "source" else self.result_edits
        if (px, py) in edits:
            old = edits.pop((px, py))
            self._group = None
            self._record(which, (px, py), old, True)
            self._group = None
            self._after_edit(which)

    def _paint(self, which: str, pixels: list[tuple[int, int]], erase: bool = False) -> None:
        if self.outcome is None:
            return
        edits = self.source_edits if which == "source" else self.result_edits
        color = None if (erase or self.brush_is_transparent) else self.brush
        arr = self._panel_array(which)
        for x, y in pixels:
            if arr is not None and color is None and not arr[y, x, 3]:
                continue  # erasing an already-transparent pixel is a no-op
            had = (x, y) in edits
            self._record(which, (x, y), edits.get((x, y)), had)
            edits[(x, y)] = color
        self._after_edit(which)

    def _after_edit(self, which: str) -> None:
        self._update_edit_label()
        if which == "result" and self.outcome is not None:
            self.redraw()  # result edits are applied at draw time
        else:
            self.schedule()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        group = self.undo_stack.pop()
        self._group = None
        touched: set[str] = set()
        frame = group[-1][0] if group else self.current_frame
        if frame != self.current_frame:
            self.set_frame(frame)
        for _frame, which, pos, old, had in reversed(group):
            edits = self.frame_source_edits.setdefault(_frame, {}) if which == "source" else self.frame_result_edits.setdefault(_frame, {})
            if had:
                edits[pos] = old
            else:
                edits.pop(pos, None)
            touched.add(which)
        for which in touched:
            self._after_edit(which)

    def clear_edits(self, which: str) -> None:
        if which == "source":
            self.frame_source_edits[self.current_frame] = {}
        else:
            self.frame_result_edits[self.current_frame] = {}
        self.undo_stack = [g for g in self.undo_stack if not any(u[0] == self.current_frame and u[1] == which for u in g)]
        self._after_edit(which)

    def _update_edit_label(self) -> None:
        total_s = sum(len(v) for v in self.frame_source_edits.values())
        total_r = sum(len(v) for v in self.frame_result_edits.values())
        text = f"edits: source {len(self.source_edits)}, result {len(self.result_edits)}"
        if self.frame_count > 1:
            text += f"  (all frames: {total_s} / {total_r})"
        self.edit_label.configure(text=text)

    # ------------------------------------------------------------------ brush / tools
    def set_tool(self, name: str) -> None:
        self.tool.set(name)
        for val, b in self.tool_buttons.items():
            b.configure(style="Selected.TButton" if val == name else "Tool.TButton")
        self.preview.configure(cursor={"pick": "target", "pencil": "pencil", "fill": "spraycan", "erase": "X_cursor", "erase_fill": "X_cursor"}.get(name, ""))

    def set_brush(self, rgb: RGB | None) -> None:
        self.brush_is_transparent = rgb is None
        if rgb is not None:
            self.brush = tuple(int(v) for v in rgb)  # type: ignore[assignment]
        self._update_brush_ui()

    def custom_brush(self) -> None:
        rgb, _ = colorchooser.askcolor(parent=self, title="Brush color", initialcolor=_hex(self.brush) if self.brush else None)
        if rgb:
            self.set_brush(tuple(int(round(v)) for v in rgb))  # type: ignore[arg-type]

    def _update_brush_ui(self) -> None:
        self.brush_swatch.delete("all")
        if self.brush_is_transparent:
            for i in range(0, 26, 6):
                for j in range(0, 20, 6):
                    self.brush_swatch.create_rectangle(i, j, i + 6, j + 6, fill="#999" if (i // 6 + j // 6) % 2 else "#666", outline="")
            self.brush_label.configure(text="transparent")
        elif self.brush is not None:
            self.brush_swatch.create_rectangle(0, 0, 26, 20, fill=_hex(self.brush), outline="")
            self.brush_label.configure(text=_key(self.brush))

    def _swatch_button(self, parent: tk.Widget, rgb: RGB, text: str = "", command=None) -> tk.Button:
        L = _lightness(rgb)
        b = tk.Button(parent, bg=_hex(rgb), activebackground=_hex(rgb), fg="#fff" if L < 55 else "#000", text=text,
                      width=2, relief="flat", bd=0, highlightthickness=1, highlightbackground=THEMES[self.theme_name.get()]["border"],
                      command=command or (lambda v=rgb: self.set_brush(v)))
        return b

    def _rebuild_palettes(self) -> None:
        for fr in (self.palette_frame, self.src_palette_frame):
            for child in fr.winfo_children():
                child.destroy()
        if self.outcome is None:
            return
        fams = sorted(self.outcome.style.families, key=lambda rf: -rf["pixels"])
        for r, rf in enumerate(fams):
            lv = sorted(((int(k), tuple(v)) for k, v in rf["levels"].items()), key=lambda kv: kv[0])
            share = rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"]))
            ttk.Label(self.palette_frame, text=f"{share:4.0%}", width=5, style="Muted.TLabel").grid(row=r, column=0)
            for c, (lvl, rgb) in enumerate(lv):
                self._swatch_button(self.palette_frame, rgb, str(lvl)).grid(row=r, column=c + 1, padx=1, pady=1)  # type: ignore[arg-type]
            if rf.get("line"):
                self._swatch_button(self.palette_frame, tuple(rf["line"]), "ln").grid(row=r, column=len(lv) + 1, padx=(6, 1), pady=1)  # type: ignore[arg-type]
        if self.outcome.style.outline_colors:
            r = len(fams)
            ttk.Label(self.palette_frame, text="line", width=5, style="Muted.TLabel").grid(row=r, column=0)
            self._swatch_button(self.palette_frame, tuple(self.outcome.style.outline_colors[0]), "").grid(row=r, column=1, padx=1, pady=1)  # type: ignore[arg-type]
        for c, info in enumerate(sorted(self.outcome.infos, key=lambda c: -c.count)):
            self._swatch_button(self.src_palette_frame, info.rgb).grid(row=0, column=c, padx=1, pady=1)
        # Result palette (whatever the automatic pass produced) is handy for touch-ups too.
        if self.outcome.report.target_palette:
            ttk.Label(self.src_palette_frame, text="  result:", style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w")
            for c, rgb in enumerate(sorted(self.outcome.report.target_palette, key=_lightness)):
                self._swatch_button(self.src_palette_frame, tuple(rgb)).grid(row=2, column=c, padx=1, pady=1)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ color mapping rows
    def _rebuild_color_rows(self) -> None:
        for child in self.colors_body.winfo_children():
            child.destroy()
        self.color_rows = {}
        if self.outcome is None:
            return
        t = THEMES[self.theme_name.get()]
        infos = sorted(self.outcome.infos, key=lambda c: -c.count)
        base_of: dict[RGB, RGB] = {}
        for fam in self.outcome.families:
            if fam.ramp is None:
                continue
            for rgb, lvl in fam.levels.items():
                base_of.setdefault(rgb, fam.ramp.at(lvl))
        hdr = ttk.Frame(self.colors_body)
        hdr.pack(fill="x")
        for text, w in (("source", 8), ("px", 5), ("→ result", 9), ("override", 14)):
            ttk.Label(hdr, text=text, width=w, style="Muted.TLabel").pack(side="left")
        for c in infos:
            key = _key(c.rgb)
            row = ttk.Frame(self.colors_body)
            row.pack(fill="x", pady=1)
            sw = tk.Canvas(row, width=22, height=18, highlightthickness=1, highlightbackground=t["border"], bg=t["bg"])
            sw.create_rectangle(0, 0, 22, 18, fill=_hex(c.rgb), outline="")
            sw.pack(side="left", padx=(0, 4))
            ttk.Label(row, text=c.role.name.lower()[:7], width=8).pack(side="left")
            ttk.Label(row, text=str(c.count), width=5).pack(side="left")
            res = base_of.get(c.rgb)
            rs = tk.Canvas(row, width=22, height=18, highlightthickness=1, highlightbackground=t["border"], bg=t["bg"])
            if res is not None:
                rs.create_rectangle(0, 0, 22, 18, fill=_hex(res), outline="")
            elif self.outcome.style.outline_colors:
                rs.create_rectangle(0, 0, 22, 18, fill=_hex(tuple(self.outcome.style.outline_colors[0])), outline="")  # type: ignore[arg-type]
            rs.pack(side="left", padx=(0, 4))
            override = self.config_model.color_map.get(key, "auto")
            btn = ttk.Button(row, text=self._override_text(override), width=14, style="Tool.TButton", command=lambda k=key, rgb=c.rgb: self._open_map_popup(k, rgb))
            btn.pack(side="left")
            self.color_rows[key] = {"frame": row, "swatch": sw, "button": btn}

    @staticmethod
    def _override_text(value: str) -> str:
        if value in ("auto", ""):
            return "auto"
        if value in ("keep", "outline", "transparent"):
            return value
        return f"→ {value}"

    def _highlight_color(self, key: str) -> None:
        t = THEMES[self.theme_name.get()]
        for k, row in self.color_rows.items():
            row["swatch"].configure(highlightbackground="#ff5050" if k == key else t["border"], highlightthickness=3 if k == key else 1)

    def clear_color_map(self) -> None:
        self.config_model = self.config_model.model_copy(update={"color_map": {}})
        self.schedule()

    def _set_override(self, key: str, value: str) -> None:
        cm = dict(self.config_model.color_map)
        if value == "auto":
            cm.pop(key, None)
        else:
            cm[key] = value
        self.config_model = self.config_model.model_copy(update={"color_map": cm})
        if key in self.color_rows:
            self.color_rows[key]["button"].configure(text=self._override_text(value))
        self.schedule()

    def _open_map_popup(self, key: str, src_rgb: RGB) -> None:
        t = THEMES[self.theme_name.get()]
        top = tk.Toplevel(self, bg=t["bg"])
        top.title(f"Map {key} to…")
        top.transient(self)
        top.resizable(False, False)
        frm = ttk.Frame(top, padding=8)
        frm.pack()
        hdr = ttk.Frame(frm)
        hdr.pack(fill="x", pady=(0, 6))
        sw = tk.Canvas(hdr, width=28, height=22, highlightthickness=1, highlightbackground=t["border"], bg=t["bg"])
        sw.create_rectangle(0, 0, 28, 22, fill=_hex(src_rgb), outline="")
        sw.pack(side="left")
        ttk.Label(hdr, text=f"  every pixel of {key} becomes…").pack(side="left")
        opts = ttk.Frame(frm)
        opts.pack(fill="x", pady=(0, 6))
        for text, val in (("Auto", "auto"), ("Keep", "keep"), ("Outline", "outline"), ("Transparent", "transparent")):
            ttk.Button(opts, text=text, style="Tool.TButton", command=lambda v=val: (self._set_override(key, v), top.destroy())).pack(side="left", padx=2)
        ttk.Button(opts, text="Custom…", style="Tool.TButton", command=lambda: self._pick_custom_map(key, top)).pack(side="left", padx=2)
        if self.outcome is not None and self.outcome.style.families:
            ttk.Label(frm, text="Reference colors (one row per region, dark → light):", style="Muted.TLabel").pack(anchor="w")
            grid = ttk.Frame(frm)
            grid.pack()
            fams = sorted(self.outcome.style.families, key=lambda rf: -rf["pixels"])
            for r, rf in enumerate(fams):
                lv = sorted(((int(k), tuple(v)) for k, v in rf["levels"].items()), key=lambda kv: kv[0])
                share = rf["pixels"] / max(1.0, rf.get("total_pixels", rf["pixels"]))
                ttk.Label(grid, text=f"{share:4.0%}", width=5, style="Muted.TLabel").grid(row=r, column=0)
                for cidx, (lvl, rgb) in enumerate(lv):
                    self._swatch_button(grid, rgb, str(lvl), command=lambda v=rgb: (self._set_override(key, _key(v)), top.destroy())).grid(row=r, column=cidx + 1, padx=1, pady=1)  # type: ignore[arg-type]
        else:
            ttk.Label(frm, text="No reference loaded — use Custom… to choose a color.", style="Muted.TLabel").pack(anchor="w")

    def _pick_custom_map(self, key: str, parent: tk.Toplevel) -> None:
        rgb, _ = colorchooser.askcolor(parent=parent, title="Target color")
        if rgb:
            self._set_override(key, _key(tuple(int(round(v)) for v in rgb)))  # type: ignore[arg-type]
            parent.destroy()

    # ------------------------------------------------------------------ output
    def save_result(self) -> None:
        if self.outcome is None or self.input_path is None or self.display_final is None:
            return
        if self.frame_count > 1 and len(self.frame_outcomes) < self.frame_count:
            messagebox.showinfo("Still working", "Not every frame has been rendered yet — wait a moment and save again.")
            return
        run_dir = project_root() / "output" / self.input_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        chosen = sorted(self.frames_selected) if self.frame_count > 1 else [0]
        finals = [self.display_final_for(k) for k in chosen]
        if any(f is None for f in finals):
            return
        final = finals[0]  # type: ignore[assignment]
        try:
            save_rgba_png(final, run_dir / "revamped.png")
            images = [rgba_to_image(self.frame_edited_source.get(chosen[0], self.frame_outcomes[chosen[0]].sprite.rgba)), rgba_to_image(final)]
            labels = [f"source: {self.input_path.name}", "revamp"]
            if self.outcome.used_refs:
                images.append(load_sprite(self.outcome.used_refs[0]).image)
                labels.append(f"reference: {self.outcome.used_refs[0].name}")
            saved = ["revamped.png", "compare.png", "report.json", "session.json"]
            if self.shiny_outcome is not None:
                save_rgba_png(self.shiny_outcome.final, run_dir / "revamped_shiny.png")
                images.append(rgba_to_image(self.shiny_outcome.final))
                labels.append("shiny revamp")
                saved.append("revamped_shiny.png")
            if len(chosen) > 1:
                sheet = np.concatenate(finals, axis=0)  # type: ignore[arg-type]
                save_rgba_png(sheet, run_dir / "revamped_frames.png")
                saved.append("revamped_frames.png")
            compare_sheet(images, labels, scale=4).save(run_dir / "compare.png", format="PNG")
            rep = self.outcome.report
            rep.output = str(run_dir / "revamped.png")
            rep.write(run_dir / "report.json")
            self._write_session(run_dir / "session.json")
        except (OSError, SpriteLoadError) as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._log(f"saved to {run_dir}\n  " + "  ".join(saved))
        self.status.configure(text="saved")

    def _session_data(self) -> dict:
        def dump(edits: dict[int, Edit]) -> dict:
            return {str(f): {f"{x},{y}": (list(v) if v is not None else None) for (x, y), v in e.items()} for f, e in edits.items() if e}

        data = self.current_config().model_dump()
        data["_input_path"] = str(self.input_path) if self.input_path else None
        data["_reference_mode"] = self.reference_mode.get()
        data["_reference_path"] = str(self.reference_path) if self.reference_path else None
        data["_shiny"] = bool(self.shiny.get())
        data["_source_edits"] = dump(self.frame_source_edits)
        data["_result_edits"] = dump(self.frame_result_edits)
        data["_current_frame"] = self.current_frame
        data["_frames_selected"] = sorted(self.frames_selected)
        data["_theme"] = self.theme_name.get()
        return data

    def _write_session(self, path: Path) -> None:
        path.write_text(json.dumps(self._session_data(), indent=2), encoding="utf-8")

    def save_preset(self) -> None:
        p = filedialog.asksaveasfilename(title="Save session", defaultextension=".json", filetypes=[("JSON", "*.json")], initialdir=str(project_root() / "output"))
        if p:
            self._write_session(Path(p))

    def load_preset(self) -> None:
        p = filedialog.askopenfilename(title="Load session", filetypes=[("JSON", "*.json")], initialdir=str(project_root() / "output"))
        if not p:
            return
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        meta = {k: data.pop(k) for k in list(data) if k.startswith("_")}
        try:
            cfg = RevampConfig(**data)
        except Exception as exc:  # pydantic validation error: report, do not crash
            messagebox.showerror("Bad session file", str(exc))
            return
        inp = meta.get("_input_path")
        if inp and Path(inp).exists() and (self.input_path is None or Path(inp) != self.input_path):
            self.load_sprite_file(Path(inp))
        self.config_model = cfg
        for field, var in self.effect_vars.items():
            var.set(bool(getattr(cfg, field)))
        for field, var in self.slider_vars.items():
            var.set(float(getattr(cfg, field)))
        self.style_var.set(cfg.style)
        self.kind_var.set(cfg.kind if cfg.kind != "unknown" else self.kind_var.get())
        self.reference_mode.set(meta.get("_reference_mode", "auto"))
        ref = meta.get("_reference_path")
        self.reference_path = Path(ref) if ref else None
        self.ref_label.configure(text=self.reference_path.name if self.reference_path else "")
        self.shiny.set(bool(meta.get("_shiny", False)))

        def _edits(d: dict) -> Edit:
            out: Edit = {}
            for k, v in d.items():
                pos = _parse(k + ",0")
                if pos is None:
                    continue
                out[(pos[0], pos[1])] = tuple(v) if v is not None else None  # type: ignore[assignment]
            return out

        def _frames(d: dict) -> dict[int, Edit]:
            # Old sessions stored a flat pixel map (frame 0); new ones are keyed by frame index.
            if d and all("," in k for k in d):
                return {0: _edits(d)}
            return {int(f): _edits(e) for f, e in d.items() if str(f).isdigit()}

        self.frame_source_edits = {i: {} for i in range(self.frame_count)}
        self.frame_result_edits = {i: {} for i in range(self.frame_count)}
        self.frame_source_edits.update(_frames(meta.get("_source_edits", {})))
        self.frame_result_edits.update(_frames(meta.get("_result_edits", {})))
        self.undo_stack = []
        cur = int(meta.get("_current_frame", 0))
        self.current_frame = cur if 0 <= cur < self.frame_count else 0
        sel = {int(k) for k in meta.get("_frames_selected", []) if 0 <= int(k) < self.frame_count}
        self.frames_selected = sel or set(range(self.frame_count))
        self._update_frame_ui()
        if meta.get("_theme") in THEMES and meta["_theme"] != self.theme_name.get():
            self.theme_name.set(meta["_theme"])
            self.apply_theme()
        self._update_edit_label()
        self.schedule()


def launch(input_path: Path | None = None) -> None:
    app = RevampApp(input_path)
    app.mainloop()
