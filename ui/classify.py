"""Tumor classification page: 30-feature input form, prediction and result display."""
from __future__ import annotations

import re
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox

from modules.config import (
    DISCLAIMER_LONG,
    FEATURE_GROUPS,
    FEATURE_INFO,
    FEATURE_NAMES,
    REPORTS_DIR,
)

from modules.predictor import PredictionResult, validate_one
from modules.report_generator import generate_record_report

from .theme import C, font, result_color
from .widgets import Card, FlatButton, Page, ProbabilityBar, make_treeview


class ClassifyPage(Page):
    title = "Classify Tumor"
    subtitle = ("Enter the 30 numerical measurements computed from a digitized FNA image of a breast mass. "
                "The trained model classifies them as Benign or Malignant.")

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.vars = {f: tk.StringVar() for f in FEATURE_NAMES}
        self.entries: dict[str, tk.Entry] = {}
        body = self.content

        # disclaimer banner -------------------------------------------------------
        banner = tk.Frame(body, bg=C["card"], highlightthickness=1, highlightbackground=C["warning"])
        banner.pack(fill="x", pady=(0, 12))
        tk.Label(banner, text="\u26A0  " + DISCLAIMER_LONG.replace("MEDICAL DISCLAIMER: ", ""), font=font(9),
                 fg="#FCD34D", bg=C["card"], justify="left", anchor="w", wraplength=880, padx=14, pady=10
                 ).pack(fill="x")

        # result area (hidden until the first prediction) ---------------------------
        self.result_holder = tk.Frame(body, bg=C["bg"])
        self.result_holder.pack(fill="x")
        self._last_result: PredictionResult | None = None
        self._last_record_id: str | None = None

        # actions + messages --------------------------------------------------------
        self._build_actions(body)
        self.message = tk.Label(body, text="", font=font(10), fg=C["malignant"], bg=C["bg"], anchor="w",
                                justify="left", wraplength=880)
        self.message.pack(fill="x", pady=(0, 6))

        # the three form sections ---------------------------------------------------
        for group, features in FEATURE_GROUPS:
            self._build_section(body, group, features)

        self._build_actions(body, bottom=True)
        for feature in FEATURE_NAMES:
            self.vars[feature].trace_add("write", lambda *_a, f=feature: self._live_validate(f))

    # ------------------------------------------------------------------ building
    def _build_actions(self, parent, bottom=False):
        bar = tk.Frame(parent, bg=C["bg"])
        bar.pack(fill="x", pady=(8 if bottom else 0, 10))
        FlatButton(bar, "Classify", self.classify).pack(side="left")
        if not bottom:
            FlatButton(bar, "Paste 30 values", self.paste_values, kind="ghost").pack(side="left", padx=(8, 0))
            FlatButton(bar, "Example: Benign", lambda: self.load_example("Benign"), kind="ghost").pack(
                side="left", padx=(8, 0))
            FlatButton(bar, "Example: Malignant", lambda: self.load_example("Malignant"), kind="ghost").pack(
                side="left", padx=(8, 0))
        FlatButton(bar, "Clear", self.clear_form, kind="ghost").pack(side="left", padx=(8, 0))

    def _build_section(self, parent, group, features):
        card = Card(parent, group, "Dataset range shown under each field is the minimum-maximum seen in training data.")
        card.pack(fill="x", pady=(0, 12))
        grid = tk.Frame(card.body, bg=C["card"])
        grid.pack(fill="x")
        for col in range(2):
            grid.columnconfigure(col, weight=1, uniform="field")
        stats = self.app.predictor.feature_stats
        for index, feature in enumerate(features):
            info = FEATURE_INFO[feature]
            cell = tk.Frame(grid, bg=C["card"])
            cell.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0 if index % 2 == 0 else 12, 0), pady=6)
            tk.Label(cell, text=f"{index + 1}. {info['display']}  ({info['unit']})", font=font(10, "bold"),
                     fg=C["text"], bg=C["card"], anchor="w").pack(fill="x")
            entry = tk.Entry(cell, textvariable=self.vars[feature], bg=C["input"], fg=C["text"],
                             insertbackground=C["text"], relief="flat", font=font(11), highlightthickness=1,
                             highlightbackground=C["border"], highlightcolor=C["primary"])
            entry.pack(fill="x", ipady=5, pady=(3, 2))
            entry.bind("<Return>", lambda _e: self.classify())
            self.entries[feature] = entry
            st = stats[feature]
            tk.Label(cell, text=f"{info['description']}\nRange: {st['min']:.4g} to {st['max']:.4g}",
                     font=font(8), fg=C["muted"], bg=C["card"], anchor="w", justify="left", wraplength=360
                     ).pack(fill="x")

    # ------------------------------------------------------------------ form helpers
    def _mark(self, feature, state):
        color = {"ok": C["border"], "error": C["malignant"]}[state]
        self.entries[feature].configure(highlightbackground=color)

    def _live_validate(self, feature):
        text = self.vars[feature].get()
        if not text.strip():
            self._mark(feature, "ok")
            return
        _value, error = validate_one(feature, text)
        self._mark(feature, "error" if error else "ok")

    def clear_form(self):
        for var in self.vars.values():
            var.set("")
        self.message.configure(text="")
        self._hide_result()
        first = self.entries[FEATURE_NAMES[0]]
        first.focus_set()

    def load_example(self, label):
        values = self.app.predictor.example_values(label)
        for feature, value in values.items():
            self.vars[feature].set(f"{value:.6g}")
        self.message.configure(text=f"Loaded the median {label.lower()} measurements from the dataset as an example. "
                                    "Press Classify to run the model.", fg=C["muted"])

    def paste_values(self):
        try:
            raw = self.clipboard_get()
        except tk.TclError:
            raw = ""
        tokens = [t for t in re.split(r"[\s,;]+", raw.strip()) if t]
        if len(tokens) != len(FEATURE_NAMES):
            messagebox.showwarning(
                "Paste values",
                f"The clipboard must contain exactly {len(FEATURE_NAMES)} numbers (found {len(tokens)}).\n\n"
                "They must be in the order of the form: 10 mean values, 10 standard-error values, "
                "10 worst values; separated by commas, spaces, tabs or new lines.",
                parent=self)
            return
        for feature, token in zip(FEATURE_NAMES, tokens):
            self.vars[feature].set(token)
        self.message.configure(text="Pasted 30 values. Check them, then press Classify.", fg=C["muted"])

    # ------------------------------------------------------------------ classification
    def classify(self):
        raw = {f: v.get() for f, v in self.vars.items()}
        validation = self.app.predictor.validate(raw)
        for feature in FEATURE_NAMES:
            self._mark(feature, "error" if feature in validation.errors else "ok")
        if not validation.ok:
            errors = list(validation.errors.values())
            shown = "\n".join(f"\u2022 {e}" for e in errors[:6])
            extra = f"\n\u2026 and {len(errors) - 6} more." if len(errors) > 6 else ""
            self.message.configure(text=f"Please correct {len(errors)} field(s):\n{shown}{extra}", fg=C["malignant"])
            first = next(f for f in FEATURE_NAMES if f in validation.errors)
            self.entries[first].focus_set()
            self.scroller.scroll_to_widget(self.entries[first])
            return
        try:
            result = self.app.predictor.predict(validation.values)
        except Exception as exc:  # model problems must never crash the GUI
            messagebox.showerror("Classification failed", f"The model could not classify these values:\n\n{exc}",
                                 parent=self)
            return
        record_id = None
        try:
            record_id = self.app.db.add_prediction(result.to_record())
        except (sqlite3.Error, OSError, ValueError) as exc:
            messagebox.showwarning("History not saved",
                                   f"The classification worked but could not be saved to the history:\n\n{exc}",
                                   parent=self)
        self.message.configure(text="")
        self._show_result(result, record_id, validation.warnings)
        self.scroller.scroll_to_top()

    # ------------------------------------------------------------------ result display
    def _hide_result(self):
        for child in self.result_holder.winfo_children():
            child.destroy()
        self._last_result = None
        self._last_record_id = None

    def _show_result(self, result: PredictionResult, record_id, warnings):
        self._hide_result()
        self._last_result, self._last_record_id = result, record_id
        color = result_color(result.label)
        card = Card(self.result_holder, bg=C["card"])
        card.pack(fill="x", pady=(0, 12))
        card.configure(highlightbackground=color, highlightcolor=color, highlightthickness=2)
        body = card.body

        top = tk.Frame(body, bg=C["card"])
        top.pack(fill="x")
        tk.Label(top, text=f"  {result.label.upper()}  ", font=font(13, "bold"), bg=color, fg="#FFFFFF").pack(
            side="left")
        tk.Label(top, text=f"{result.label} Tumor Classification", font=font(18, "bold"), fg=color,
                 bg=C["card"]).pack(side="left", padx=14)

        meaning = ("The measurements resemble benign samples in the training data."
                   if result.label == "Benign" else
                   "The measurements resemble malignant samples in the training data.")
        tk.Label(body, text=meaning + " This is a model estimate, not a diagnosis.", font=font(10), fg=C["muted"],
                 bg=C["card"], anchor="w").pack(fill="x", pady=(8, 10))

        if result.probability is not None:
            tk.Label(body, text=f"Estimated class probability: {result.probability:.1%} {result.label}",
                     font=font(12, "bold"), fg=C["text"], bg=C["card"], anchor="w").pack(fill="x")
            bar = ProbabilityBar(body)
            bar.pack(fill="x", pady=(6, 2))
            bar.set(result.prob_malignant)
            tk.Label(body, text=f"Benign {result.prob_benign:.1%}   |   Malignant {result.prob_malignant:.1%}",
                     font=font(9), fg=C["muted"], bg=C["card"], anchor="w").pack(fill="x")

        meta = tk.Frame(body, bg=C["card"])
        meta.pack(fill="x", pady=(12, 4))
        for label, value in (("Date & time", result.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
                             ("Record ID", record_id or "not saved"),
                             ("Model", result.model_name)):
            cell = tk.Frame(meta, bg=C["card"])
            cell.pack(side="left", padx=(0, 28))
            tk.Label(cell, text=label.upper(), font=font(8, "bold"), fg=C["muted"], bg=C["card"]).pack(anchor="w")
            tk.Label(cell, text=value, font=font(10, "bold"), fg=C["text"], bg=C["card"]).pack(anchor="w")

        if warnings:
            shown = warnings[:4] + ([f"\u2026 and {len(warnings) - 4} more notes."] if len(warnings) > 4 else [])
            tk.Label(body, text="Input notes:\n" + "\n".join(f"\u2022 {w}" for w in shown), font=font(9),
                     fg=C["warning"], bg=C["card"], anchor="w", justify="left", wraplength=860).pack(
                fill="x", pady=(10, 0))

        tk.Label(body, text="Tumor measurement summary", font=font(11, "bold"), fg=C["text"], bg=C["card"],
                 anchor="w").pack(fill="x", pady=(14, 4))
        frame, tree = make_treeview(
            body, [("group", "Group", 110, "w"), ("feature", "Measurement", 190, "w"),
                   ("value", "Value", 100, "e"), ("range", "Dataset range", 170, "e")], height=8)
        frame.pack(fill="x")
        stats = self.app.predictor.feature_stats
        for group, features in FEATURE_GROUPS:
            short = group.split()[0]
            for f in features:
                tree.insert("", "end", values=(short, FEATURE_INFO[f]["display"], f"{result.values[f]:.6g}",
                                               f"{stats[f]['min']:.4g} to {stats[f]['max']:.4g}"))

        tk.Label(body, text=DISCLAIMER_LONG, font=font(9), fg="#FCD34D", bg=C["card"], anchor="w", justify="left",
                 wraplength=860).pack(fill="x", pady=(12, 8))
        buttons = tk.Frame(body, bg=C["card"])
        buttons.pack(fill="x")
        FlatButton(buttons, "Export this report (.txt)", self.export_report, kind="secondary").pack(side="left")
        FlatButton(buttons, "View history", lambda: self.app.show_page("history"), kind="ghost").pack(
            side="left", padx=(8, 0))

    def export_report(self):
        if not self._last_record_id:
            messagebox.showinfo("Export report", "This classification was not saved, so it cannot be exported.",
                                parent=self)
            return
        record = next((r for r in self.app.db.get_records(search=self._last_record_id)), None)
        if record is None:
            messagebox.showerror("Export report", "The saved record could not be found.", parent=self)
            return
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="Save classification report", defaultextension=".txt",
            initialdir=str(REPORTS_DIR), initialfile=f"classification_{record['record_id']}.txt",
            filetypes=[("Text file", "*.txt")])
        if not path:
            return
        try:
            generate_record_report(path, record, self.app.metadata)
            messagebox.showinfo("Report saved", f"Saved to:\n{path}", parent=self)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
