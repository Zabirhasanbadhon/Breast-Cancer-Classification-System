"""Dashboard: key numbers, model summary, prediction distribution and recent history."""
from __future__ import annotations

import tkinter as tk

from modules import analytics
from modules.config import APP_NAME, DISCLAIMER_SHORT

from .theme import C, font, result_color
from .widgets import Card, ChartFrame, FlatButton, Page, StatCard, key_value_rows, make_treeview


class DashboardPage(Page):
    title = APP_NAME
    subtitle = "Classifies fine-needle-aspirate (FNA) tumor measurements as Benign or Malignant. " + DISCLAIMER_SHORT

    def __init__(self, parent, app):
        super().__init__(parent, app)
        body = self.content

        # --- stat cards -----------------------------------------------------
        row = tk.Frame(body, bg=C["bg"])
        row.pack(fill="x")
        self.cards = {
            "total": StatCard(row, "Total predictions", C["text"]),
            "benign": StatCard(row, "Benign predictions", C["benign"]),
            "malignant": StatCard(row, "Malignant predictions", C["malignant"]),
            "latest": StatCard(row, "Latest prediction", C["muted"]),
        }
        for i, card in enumerate(self.cards.values()):
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 12, 0))
            row.columnconfigure(i, weight=1, uniform="stat")

        # --- model summary + chart -------------------------------------------
        middle = tk.Frame(body, bg=C["bg"])
        middle.pack(fill="x", pady=14)
        middle.columnconfigure(0, weight=1, uniform="mid")
        middle.columnconfigure(1, weight=1, uniform="mid")
        self.model_card = Card(middle, "Model evaluation summary",
                               "Held-out test split of the WDBC dataset (positive class = Malignant)")
        self.model_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self.model_rows = tk.Frame(self.model_card.body, bg=C["card"])
        self.model_rows.pack(fill="x")
        FlatButton(self.model_card.body, "View full evaluation", lambda: app.show_page("reports"),
                   kind="ghost").pack(anchor="w", pady=(12, 0))

        self.chart_card = Card(middle, "Prediction distribution", "Predictions saved by this application")
        self.chart_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.chart = ChartFrame(self.chart_card.body)
        self.chart.pack(fill="both", expand=True)

        # --- recent history ---------------------------------------------------
        self.recent_card = Card(body, "Recent prediction history")
        self.recent_card.pack(fill="x")
        frame, self.tree = make_treeview(
            self.recent_card.body,
            [("record_id", "Record ID", 170, "w"), ("created_at", "Date & time", 160, "w"),
             ("result", "Result", 100, "w"), ("probability", "Confidence", 100, "e"),
             ("model_name", "Model", 200, "w")],
            height=7,
        )
        frame.pack(fill="x")
        self.empty_label = tk.Label(self.recent_card.body, text="", font=font(10), fg=C["muted"], bg=C["card"])
        self.empty_label.pack(anchor="w", pady=(6, 0))
        action = FlatButton(self.recent_card.body, "Classify a tumor", lambda: app.show_page("classify"))
        action.pack(anchor="w", pady=(10, 0))

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        stats = self.app.db.stats()
        self.cards["total"].set(stats["total"], sub="saved in history")
        total = stats["total"]
        share = lambda n: f"{n / total:.0%} of predictions" if total else " "  # noqa: E731
        self.cards["benign"].set(stats["benign"], sub=share(stats["benign"]))
        self.cards["malignant"].set(stats["malignant"], sub=share(stats["malignant"]))
        latest = stats["latest"]
        if latest:
            self.cards["latest"].set(latest["result"], color=result_color(latest["result"]),
                                     sub=latest["created_at"])
        else:
            self.cards["latest"].set("None yet", color=C["muted"], sub="no predictions saved")

        for child in self.model_rows.winfo_children():
            child.destroy()
        meta = self.app.metadata
        test = meta["models"][meta["selected_model"]]["test"]
        key_value_rows(self.model_rows, [
            ("Selected model", meta["selected_model"], C["primary"]),
            ("Accuracy", f"{test['accuracy']:.1%}"),
            ("Sensitivity (malignant recall)", f"{test['recall_malignant']:.1%}"),
            ("Specificity", f"{test['specificity']:.1%}"),
            ("Precision (malignant)", f"{test['precision_malignant']:.1%}"),
            ("F1-score (malignant)", f"{test['f1_malignant']:.1%}"),
            ("ROC-AUC", f"{test['roc_auc']:.3f}"),
            ("Test samples", sum(meta["dataset"]["test_counts"].values())),
        ], label_width=28)

        self.chart.show(analytics.fig_prediction_distribution(stats, figsize=(4.4, 3.1)))

        self.tree.delete(*self.tree.get_children())
        records = self.app.db.get_records(limit=7)
        for rec in records:
            prob = "n/a" if rec["probability"] is None else f"{rec['probability']:.1%}"
            self.tree.insert("", "end", iid=str(rec["id"]), tags=(rec["result"],),
                             values=(rec["record_id"], rec["created_at"], rec["result"], prob,
                                     rec["model_name"] or ""))
        self.empty_label.configure(text="" if records else "No predictions yet. Use 'Classify Tumor' to create the first one.")
