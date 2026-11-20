"""Analytics page: seven Matplotlib charts, clearly labelled by data source."""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

from modules import analytics
from modules.config import REPORTS_DIR

from .theme import C, font
from .widgets import Card, ChartFrame, FlatButton, Page

# key, button label, data source, caption
CHARTS = [
    ("pred_dist", "Predictions", "App predictions",
     "Share of Benign vs Malignant among the classifications saved by this application."),
    ("history", "Over time", "App predictions",
     "Number of saved classifications per day (last 30 active days)."),
    ("models", "Model comparison", "Dataset",
     "Held-out test-split metrics of all seven trained models. The star marks the selected model."),
    ("dataset", "Class distribution", "Dataset",
     "Benign/malignant counts in the original WDBC dataset and in the training and test splits."),
    ("cm", "Confusion matrix", "Dataset",
     "Selected model on the test split. Rows = actual class, columns = predicted class."),
    ("roc", "ROC curve", "Dataset",
     "Trade-off between sensitivity and false-positive rate for the selected model (other models in grey)."),
    ("fi", "Feature importance", "Dataset",
     "Which measurements influence the selected model most (method shown on the x-axis)."),
]


class AnalyticsPage(Page):
    title = "Analytics"
    subtitle = ("Charts tagged DATASET describe the original dataset and the training run. "
                "Charts tagged APP PREDICTIONS describe classifications saved by this application.")
    scrollable = False

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.current = "pred_dist"
        self.buttons: dict[str, FlatButton] = {}

        nav = tk.Frame(self.content, bg=C["bg"])
        nav.pack(fill="x", pady=(0, 10))
        for i, (key, label, _src, _cap) in enumerate(CHARTS):
            btn = FlatButton(nav, label, lambda k=key: self.select(k), kind="ghost", padx=12, pady=6)
            btn.grid(row=i // 4, column=i % 4, padx=(0, 6), pady=(0, 6), sticky="w")
            self.buttons[key] = btn

        self.card = Card(self.content, padx=14, pady=12)
        self.card.pack(fill="both", expand=True)
        head = tk.Frame(self.card.body, bg=C["card"])
        head.pack(fill="x")
        self.tag = tk.Label(head, text="", font=font(8, "bold"), padx=8, pady=2)
        self.tag.pack(side="left")
        self.caption = tk.Label(head, text="", font=font(9), fg=C["muted"], bg=C["card"], anchor="w",
                                justify="left", wraplength=620)
        self.caption.pack(side="left", padx=10)
        FlatButton(head, "Save PNG", self.save_png, kind="ghost", padx=10, pady=4).pack(side="right")
        self.chart = ChartFrame(self.card.body)
        self.chart.pack(fill="both", expand=True, pady=(8, 0))

    def on_show(self) -> None:
        self.select(self.current)

    def _build_figure(self, key):
        meta = self.app.metadata
        return {
            "pred_dist": lambda: analytics.fig_prediction_distribution(self.app.db.stats()),
            "history": lambda: analytics.fig_history_over_time(self.app.db.timeline()),
            "models": lambda: analytics.fig_model_comparison(meta),
            "dataset": lambda: analytics.fig_dataset_distribution(meta),
            "cm": lambda: analytics.fig_confusion_matrix(meta),
            "roc": lambda: analytics.fig_roc_curve(meta),
            "fi": lambda: analytics.fig_feature_importance(meta),
        }[key]()

    def select(self, key: str) -> None:
        self.current = key
        for k, button in self.buttons.items():
            button.set_kind("primary" if k == key else "ghost")
        _k, _label, source, caption = next(c for c in CHARTS if c[0] == key)
        is_dataset = source == "Dataset"
        self.tag.configure(text=("DATASET" if is_dataset else "APP PREDICTIONS"),
                           bg=C["secondary"] if is_dataset else C["primary"],
                           fg="#FFFFFF" if is_dataset else "#06201D")
        self.caption.configure(text=caption)
        try:
            self.chart.show(self._build_figure(key))
        except Exception as exc:  # a chart problem must not take the app down
            messagebox.showerror("Chart error", f"The chart could not be drawn:\n\n{exc}", parent=self)

    def save_png(self) -> None:
        if self.chart.figure is None:
            return
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="Save chart", defaultextension=".png", initialdir=str(REPORTS_DIR),
            initialfile=f"chart_{self.current}_{datetime.now():%Y%m%d_%H%M%S}.png", filetypes=[("PNG image", "*.png")])
        if not path:
            return
        try:
            self.chart.figure.savefig(path, dpi=150, facecolor=self.chart.figure.get_facecolor())
            messagebox.showinfo("Chart saved", f"Saved to:\n{path}", parent=self)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
