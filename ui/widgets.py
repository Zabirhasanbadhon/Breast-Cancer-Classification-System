"""Reusable Tkinter widgets styled for the dark healthcare dashboard."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .theme import C, font, mono


# --------------------------------------------------------------------------- ttk theme
def apply_ttk_style(root: tk.Misc) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Treeview", background=C["card"], fieldbackground=C["card"], foreground=C["text"],
                    rowheight=28, borderwidth=0, font=font(10))
    style.configure("Treeview.Heading", background=C["sidebar"], foreground=C["text"], relief="flat",
                    font=font(10, "bold"), padding=(6, 6))
    style.map("Treeview", background=[("selected", C["primary_dark"])], foreground=[("selected", "#FFFFFF")])
    style.map("Treeview.Heading", background=[("active", C["border"])])
    style.configure("TCombobox", fieldbackground=C["input"], background=C["border"], foreground=C["text"],
                    arrowcolor=C["text"], bordercolor=C["border"], lightcolor=C["border"],
                    darkcolor=C["border"], padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", C["input"])], foreground=[("readonly", C["text"])],
              selectbackground=[("readonly", C["input"])], selectforeground=[("readonly", C["text"])])
    for orient in ("Vertical", "Horizontal"):
        style.configure(f"{orient}.TScrollbar", background=C["border"], troughcolor=C["bg"],
                        bordercolor=C["bg"], arrowcolor=C["muted"], lightcolor=C["border"], darkcolor=C["border"])
    root.option_add("*TCombobox*Listbox.background", C["input"])
    root.option_add("*TCombobox*Listbox.foreground", C["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["primary_dark"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")


# --------------------------------------------------------------------------- buttons
_BUTTON_STYLES = {
    "primary": (C["primary"], "#2DD4BF", "#06201D"),
    "secondary": (C["secondary"], "#60A5FA", "#FFFFFF"),
    "danger": (C["malignant"], "#F87171", "#FFFFFF"),
    "ghost": (C["border"], "#34496D", C["text"]),
}


class FlatButton(tk.Label):
    """A label-based button: looks identical on Windows, macOS and Linux (native buttons ignore colours on macOS)."""

    def __init__(self, parent, text, command=None, kind="primary", padx=16, pady=8, **kw):
        self._bg, self._hover, self._fg = _BUTTON_STYLES[kind]
        self._command = command
        self._enabled = True
        super().__init__(parent, text=text, bg=self._bg, fg=self._fg, font=font(10, "bold"), padx=padx,
                         pady=pady, cursor="hand2", **kw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, _e):
        if self._enabled:
            self.configure(bg=self._hover)

    def _on_leave(self, _e):
        if self._enabled:
            self.configure(bg=self._bg)

    def _on_release(self, event):
        inside = 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height()
        if self._enabled and inside and self._command:
            self._command()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.configure(bg=self._bg if enabled else C["border"], fg=self._fg if enabled else C["muted"],
                       cursor="hand2" if enabled else "arrow")

    def set_kind(self, kind: str) -> None:
        self._bg, self._hover, self._fg = _BUTTON_STYLES[kind]
        self.configure(bg=self._bg, fg=self._fg)


# --------------------------------------------------------------------------- containers
class Card(tk.Frame):
    """Rounded-look panel. Put child widgets into ``card.body``."""

    def __init__(self, parent, title=None, subtitle=None, bg=None, padx=18, pady=16, **kw):
        bg = bg or C["card"]
        super().__init__(parent, bg=bg, highlightthickness=1, highlightbackground=C["border"],
                         highlightcolor=C["border"], **kw)
        self.body = tk.Frame(self, bg=bg)
        self.body.pack(fill="both", expand=True, padx=padx, pady=pady)
        if title:
            tk.Label(self.body, text=title, font=font(12, "bold"), fg=C["text"], bg=bg, anchor="w").pack(fill="x")
        if subtitle:
            tk.Label(self.body, text=subtitle, font=font(9), fg=C["muted"], bg=bg, anchor="w", justify="left",
                     wraplength=520).pack(fill="x", pady=(2, 0))
        if title or subtitle:
            tk.Frame(self.body, bg=bg, height=8).pack()


class StatCard(Card):
    def __init__(self, parent, title, accent=None):
        super().__init__(parent, padx=18, pady=14)
        self._title = tk.Label(self.body, text=title.upper(), font=font(8, "bold"), fg=C["muted"], bg=C["card"],
                               anchor="w")
        self._title.pack(fill="x")
        self._value = tk.Label(self.body, text="-", font=font(24, "bold"), fg=accent or C["text"], bg=C["card"],
                               anchor="w")
        self._value.pack(fill="x", pady=(4, 0))
        self._sub = tk.Label(self.body, text=" ", font=font(9), fg=C["muted"], bg=C["card"], anchor="w")
        self._sub.pack(fill="x")
        self._accent = accent or C["text"]

    def set(self, value, color=None, sub=" "):
        self._value.configure(text=str(value), fg=color or self._accent)
        self._sub.configure(text=sub)


class ScrollableFrame(tk.Frame):
    """Vertically scrollable container; add children to ``.inner``."""

    def __init__(self, parent, bg=None, **kw):
        bg = bg or C["bg"]
        super().__init__(parent, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(sequence, self._on_wheel, add="+")

    def _is_mine(self, widget) -> bool:
        while widget is not None:
            if widget is self:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_wheel(self, event):
        try:
            if not self.winfo_viewable():
                return
            target = self.winfo_containing(event.x_root, event.y_root)
            if target is None or not self._is_mine(target):
                return
            if target.winfo_class() in ("Treeview", "Text", "Listbox"):
                return  # those widgets scroll themselves
            if self.inner.winfo_reqheight() <= self.canvas.winfo_height():
                return
        except tk.TclError:
            return
        if getattr(event, "num", None) == 4:
            units = -2
        elif getattr(event, "num", None) == 5:
            units = 2
        else:
            units = -2 if event.delta > 0 else 2
        self.canvas.yview_scroll(units, "units")

    def scroll_to_top(self):
        self.canvas.yview_moveto(0)

    def scroll_to_widget(self, widget):
        self.update_idletasks()
        total = max(1, self.inner.winfo_height())
        offset = widget.winfo_rooty() - self.inner.winfo_rooty() - 90
        self.canvas.yview_moveto(max(0.0, min(1.0, offset / total)))


class Page(tk.Frame):
    """Base class of every page: optional scrolling, header and an ``on_show`` hook."""

    title = ""
    subtitle = ""
    scrollable = True

    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app = app
        if self.scrollable:
            self.scroller = ScrollableFrame(self)
            self.scroller.pack(fill="both", expand=True)
            self.content = self.scroller.inner
        else:
            self.scroller = None
            self.content = tk.Frame(self, bg=C["bg"])
            self.content.pack(fill="both", expand=True)
        self.content.configure(padx=28, pady=22)
        header = tk.Frame(self.content, bg=C["bg"])
        header.pack(fill="x")
        tk.Label(header, text=self.title, font=font(20, "bold"), fg=C["text"], bg=C["bg"], anchor="w").pack(fill="x")
        if self.subtitle:
            tk.Label(header, text=self.subtitle, font=font(10), fg=C["muted"], bg=C["bg"], anchor="w",
                     justify="left", wraplength=900).pack(fill="x", pady=(2, 0))
        tk.Frame(self.content, bg=C["bg"], height=14).pack()

    def on_show(self) -> None:  # pragma: no cover - overridden
        pass


# --------------------------------------------------------------------------- misc
def key_value_rows(parent, rows, bg=None, label_width=26):
    """Two-column label/value list. ``rows`` = [(label, value) | (label, value, colour)]."""
    bg = bg or C["card"]
    for row in rows:
        label, value = row[0], row[1]
        color = row[2] if len(row) > 2 else C["text"]
        line = tk.Frame(parent, bg=bg)
        line.pack(fill="x", pady=2)
        tk.Label(line, text=label, width=label_width, anchor="w", font=font(10), fg=C["muted"], bg=bg).pack(side="left")
        tk.Label(line, text=str(value), anchor="w", justify="left", font=font(10, "bold"), fg=color, bg=bg,
                 wraplength=520).pack(side="left", fill="x", expand=True)


def make_treeview(parent, columns, height=10, selectmode="browse"):
    """Treeview + vertical scrollbar inside a frame. ``columns`` = [(id, heading, width, anchor)]."""
    frame = tk.Frame(parent, bg=C["card"])
    tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=height,
                        selectmode=selectmode)
    for col_id, heading, width, anchor in columns:
        tree.heading(col_id, text=heading)
        tree.column(col_id, width=width, minwidth=50, anchor=anchor, stretch=True)
    bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    tree.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    tree.tag_configure("Benign", foreground=C["benign"])
    tree.tag_configure("Malignant", foreground=C["malignant"])
    return frame, tree


class ChartFrame(tk.Frame):
    """Hosts a Matplotlib ``Figure`` (via FigureCanvasTkAgg) and lets it be swapped."""

    def __init__(self, parent):
        super().__init__(parent, bg=C["card"])
        self._canvas = None
        self.figure = None

    def show(self, figure) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.clear()
        self.figure = figure
        self._canvas = FigureCanvasTkAgg(figure, master=self)
        widget = self._canvas.get_tk_widget()
        widget.configure(bg=C["card"], highlightthickness=0)
        widget.pack(fill="both", expand=True)
        self._canvas.draw()

    def clear(self) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self.figure = None


class ProbabilityBar(tk.Canvas):
    """Horizontal bar: green = P(benign), red = P(malignant)."""

    def __init__(self, parent, height=20, bg=None):
        super().__init__(parent, height=height, bg=bg or C["card"], highlightthickness=0)
        self._p_malignant = 0.0
        self.bind("<Configure>", lambda _e: self._redraw())

    def set(self, p_malignant: float) -> None:
        self._p_malignant = max(0.0, min(1.0, float(p_malignant)))
        self._redraw()

    def _redraw(self) -> None:
        self.delete("all")
        width, height = max(self.winfo_width(), 10), int(self.cget("height"))
        split = width * (1 - self._p_malignant)
        self.create_rectangle(0, 0, split, height, fill=C["benign"], width=0)
        self.create_rectangle(split, 0, width, height, fill=C["malignant"], width=0)


def read_only_text(parent, height=14):
    """Read-only monospace Text with scrollbar. Returns (frame, text_widget, set_text_fn)."""
    frame = tk.Frame(parent, bg=C["card"])
    text = tk.Text(frame, height=height, wrap="none", bg=C["input"], fg=C["text"], relief="flat", font=mono(9),
                   highlightthickness=1, highlightbackground=C["border"], padx=10, pady=8, state="disabled")
    bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")

    def set_text(value: str) -> None:
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", value)
        text.configure(state="disabled")

    return frame, text, set_text
