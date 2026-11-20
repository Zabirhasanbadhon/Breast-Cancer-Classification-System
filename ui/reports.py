"""Reports page: exports, model evaluation table and classification reports."""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from modules.config import REPORTS_DIR
from modules.report_generator import export_model_comparison, generate_summary_report

from .theme import C, font
from .widgets import Card, FlatButton, Page, make_treeview, read_only_text


class ReportsPage(Page):
    title = "Reports & Evaluation"
    subtitle = "Export results and review how every model performed on the held-out test split (positive class = Malignant)."

    def __init__(self, parent, app):
        super().__init__(parent, app)
        body = self.content

        # --- export actions --------------------------------------------------------
        actions = Card(body, "Exports", "Files are written where you choose; the default folder is reports/ inside the project.")
        actions.pack(fill="x", pady=(0, 12))
        row = tk.Frame(actions.body, bg=C["card"])
        row.pack(fill="x")
        FlatButton(row, "Export prediction history (CSV)", self.export_history).pack(side="left")
        FlatButton(row, "Classification summary report (.txt)", self.export_summary, kind="secondary").pack(
            side="left", padx=8)
        FlatButton(row, "Export model comparison (CSV)", self.export_comparison, kind="ghost").pack(side="left")

        # --- model table -------------------------------------------------------------
        table_card = Card(body, "Model evaluation metrics",
                          "Test-split metrics. 'Score' is the cross-validated selection score used to choose the model "
                          "(sensitivity-weighted, accuracy is not used).")
        table_card.pack(fill="x", pady=(0, 12))
        columns = [("model", "Model", 190, "w"), ("accuracy", "Accuracy", 80, "e"), ("precision", "Precision", 80, "e"),
                   ("sensitivity", "Sensitivity", 90, "e"), ("specificity", "Specificity", 90, "e"),
                   ("f1", "F1", 70, "e"), ("auc", "ROC-AUC", 80, "e"), ("fn", "FN", 45, "e"), ("fp", "FP", 45, "e"),
                   ("score", "Score", 70, "e")]
        frame, self.tree = make_treeview(table_card.body, columns, height=7)
        frame.pack(fill="x")
        self.tree.tag_configure("selected_model", foreground=C["primary"])
        self.note = tk.Label(table_card.body, text="", font=font(9), fg=C["muted"], bg=C["card"], anchor="w",
                             justify="left", wraplength=860)
        self.note.pack(fill="x", pady=(8, 0))

        # --- classification report -----------------------------------------------------
        report_card = Card(body, "Classification report")
        report_card.pack(fill="x", pady=(0, 12))
        pick = tk.Frame(report_card.body, bg=C["card"])
        pick.pack(fill="x", pady=(0, 8))
        tk.Label(pick, text="Model", font=font(10), fg=C["muted"], bg=C["card"]).pack(side="left")
        self.model_var = tk.StringVar()
        self.combo = ttk.Combobox(pick, textvariable=self.model_var, state="readonly", width=28)
        self.combo.pack(side="left", padx=8)
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._show_report())
        frame, _text, self._set_report = read_only_text(report_card.body, height=11)
        frame.pack(fill="x")

        # --- selection strategy ---------------------------------------------------------
        strategy = Card(body, "How the model was selected")
        strategy.pack(fill="x")
        self.strategy_label = tk.Label(strategy.body, text="", font=font(10), fg=C["text"], bg=C["card"],
                                       anchor="w", justify="left", wraplength=860)
        self.strategy_label.pack(fill="x")

    def on_show(self) -> None:
        meta = self.app.metadata
        selected = meta["selected_model"]
        self.tree.delete(*self.tree.get_children())
        for name, info in sorted(meta["models"].items(), key=lambda kv: -kv[1]["selection_score"]):
            t, c = info["test"], info["confusion_counts"]
            label = ("\u2605 " if name == selected else "") + name
            self.tree.insert("", "end", tags=("selected_model",) if name == selected else (), values=(
                label, f"{t['accuracy']:.3f}", f"{t['precision_malignant']:.3f}", f"{t['recall_malignant']:.3f}",
                f"{t['specificity']:.3f}", f"{t['f1_malignant']:.3f}", f"{t['roc_auc']:.3f}", c["FN"], c["FP"],
                f"{info['selection_score']:.4f}"))
        sel = meta["models"][selected]["confusion_counts"]
        self.note.configure(text=(
            f"\u2605 = selected model ({selected}). FN = malignant cases predicted Benign (the costly error): {sel['FN']}. "
            f"FP = benign cases predicted Malignant: {sel['FP']}. Small test set ({sum(meta['dataset']['test_counts'].values())} samples): "
            "differences of one or two cases between models are not statistically meaningful."))
        names = list(meta["models"])
        self.combo.configure(values=names)
        if not self.model_var.get():
            self.model_var.set(selected)
        self._show_report()
        self.strategy_label.configure(text=meta["selection"]["strategy"])

    def _show_report(self) -> None:
        name = self.model_var.get()
        self._set_report(self.app.metadata["models"][name]["classification_report"])

    # ------------------------------------------------------------------ exports
    def _ask_path(self, title, default_name, ext, kind):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        return filedialog.asksaveasfilename(parent=self, title=title, defaultextension=ext,
                                            initialdir=str(REPORTS_DIR), initialfile=default_name,
                                            filetypes=[(kind, f"*{ext}")])

    def export_history(self) -> None:
        if self.app.db.count() == 0:
            messagebox.showinfo("Export history", "There are no saved predictions to export yet.", parent=self)
            return
        path = self._ask_path("Export prediction history", f"prediction_history_{datetime.now():%Y%m%d_%H%M%S}.csv",
                              ".csv", "CSV file")
        if path:
            self._run(lambda: f"{self.app.db.export_csv(path)} record(s) exported to:\n{path}", "Export complete")

    def export_summary(self) -> None:
        path = self._ask_path("Save summary report", f"classification_summary_{datetime.now():%Y%m%d_%H%M%S}.txt",
                              ".txt", "Text file")
        if path:
            self._run(lambda: f"Report saved to:\n{generate_summary_report(path, self.app.db, self.app.metadata)}",
                      "Report saved")

    def export_comparison(self) -> None:
        path = self._ask_path("Export model comparison", "model_comparison.csv", ".csv", "CSV file")
        if path:
            self._run(lambda: f"Model comparison saved to:\n{export_model_comparison(path, self.app.metadata)}",
                      "Export complete")

    def _run(self, action, success_title) -> None:
        try:
            messagebox.showinfo(success_title, action(), parent=self)
        except OSError as exc:
            messagebox.showerror("Export failed", f"The file could not be written:\n\n{exc}", parent=self)
