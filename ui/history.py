"""Prediction history: search, filter, details, delete and CSV export."""
from __future__ import annotations

import sqlite3
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from modules.config import DISCLAIMER_LONG, FEATURE_GROUPS, FEATURE_INFO, REPORTS_DIR
from modules.report_generator import generate_record_report

from .theme import C, font, result_color
from .widgets import Card, FlatButton, Page, make_treeview

COLUMNS = [
    ("record_id", "Record ID", 150, "w"),
    ("created_at", "Date & time", 150, "w"),
    ("result", "Result", 90, "w"),
    ("probability", "Confidence", 90, "e"),
    ("prob_malignant", "P(Malignant)", 100, "e"),
    ("radius_mean", "Radius Mean", 95, "e"),
    ("area_mean", "Area Mean", 95, "e"),
    ("concavity_mean", "Concavity Mean", 115, "e"),
    ("model_name", "Model", 160, "w"),
]


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


class HistoryPage(Page):
    title = "Prediction History"
    subtitle = "Every classification is saved locally with a generated record ID. No patient-identifying information is stored."
    scrollable = False

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._after_id = None
        self.records: list[dict] = []

        toolbar = tk.Frame(self.content, bg=C["bg"])
        toolbar.pack(fill="x", pady=(0, 10))
        tk.Label(toolbar, text="Search", font=font(10), fg=C["muted"], bg=C["bg"]).pack(side="left")
        self.search_var = tk.StringVar()
        entry = tk.Entry(toolbar, textvariable=self.search_var, width=26, bg=C["input"], fg=C["text"],
                         insertbackground=C["text"], relief="flat", font=font(10), highlightthickness=1,
                         highlightbackground=C["border"], highlightcolor=C["primary"])
        entry.pack(side="left", padx=(8, 12), ipady=4)
        self.filter_var = tk.StringVar(value="All")
        combo = ttk.Combobox(toolbar, textvariable=self.filter_var, values=["All", "Benign", "Malignant"],
                             state="readonly", width=11)
        combo.pack(side="left", padx=(0, 12))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        self.search_var.trace_add("write", self._schedule_refresh)

        FlatButton(toolbar, "Refresh", self.refresh, kind="ghost", padx=12, pady=6).pack(side="left", padx=(0, 6))
        FlatButton(toolbar, "View details", self.view_details, kind="secondary", padx=12, pady=6).pack(
            side="left", padx=(0, 6))
        FlatButton(toolbar, "Export CSV", self.export_csv, kind="primary", padx=12, pady=6).pack(side="left", padx=(0, 6))
        FlatButton(toolbar, "Delete selected", self.delete_selected, kind="danger", padx=12, pady=6).pack(side="left")

        card = Card(self.content, padx=10, pady=10)
        card.pack(fill="both", expand=True)
        frame, self.tree = make_treeview(card.body, COLUMNS, height=14, selectmode="extended")
        frame.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda _e: self.view_details())
        self.tree.bind("<Return>", lambda _e: self.view_details())
        self.tree.bind("<Delete>", lambda _e: self.delete_selected())
        self.count_label = tk.Label(self.content, text="", font=font(9), fg=C["muted"], bg=C["bg"], anchor="w")
        self.count_label.pack(fill="x", pady=(8, 0))

    def on_show(self) -> None:
        self.refresh()

    # ------------------------------------------------------------------ data
    def _schedule_refresh(self, *_args):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(250, self.refresh)

    def refresh(self) -> None:
        self._after_id = None
        try:
            self.records = self.app.db.get_records(self.search_var.get(), self.filter_var.get())
        except sqlite3.Error as exc:
            messagebox.showerror("Database error", f"The history could not be read:\n\n{exc}", parent=self)
            return
        self.tree.delete(*self.tree.get_children())
        for rec in self.records:
            self.tree.insert("", "end", iid=str(rec["id"]), tags=(rec["result"],), values=(
                rec["record_id"], rec["created_at"], rec["result"], _pct(rec["probability"]),
                _pct(rec["prob_malignant"]), f"{rec['radius_mean']:.4g}", f"{rec['area_mean']:.5g}",
                f"{rec['concavity_mean']:.4g}", rec["model_name"] or ""))
        total = self.app.db.count()
        shown = len(self.records)
        suffix = f" (filtered from {total})" if shown != total else ""
        self.count_label.configure(
            text=f"{shown} record(s){suffix}. Double-click a row for details; Ctrl/Shift-click selects several.")

    def _selected_ids(self) -> list[int]:
        return [int(i) for i in self.tree.selection()]

    # ------------------------------------------------------------------ actions
    def view_details(self) -> None:
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("View details", "Select a record first.", parent=self)
            return
        record = self.app.db.get_record(ids[0])
        if record is None:
            messagebox.showwarning("View details", "That record no longer exists.", parent=self)
            self.refresh()
            return
        DetailsDialog(self, self.app, record)

    def delete_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            messagebox.showinfo("Delete", "Select one or more records first.", parent=self)
            return
        if not messagebox.askyesno(
                "Confirm delete",
                f"Permanently delete {len(ids)} selected record(s)?\n\nThis cannot be undone.",
                icon="warning", parent=self):
            return
        try:
            removed = self.app.db.delete_records(ids)
        except sqlite3.Error as exc:
            messagebox.showerror("Delete failed", str(exc), parent=self)
            return
        self.refresh()
        messagebox.showinfo("Deleted", f"{removed} record(s) deleted.", parent=self)

    def export_csv(self) -> None:
        if not self.records:
            messagebox.showinfo("Export CSV", "There are no records to export.", parent=self)
            return
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="Export prediction history", defaultextension=".csv", initialdir=str(REPORTS_DIR),
            initialfile=f"prediction_history_{datetime.now():%Y%m%d_%H%M%S}.csv", filetypes=[("CSV file", "*.csv")])
        if not path:
            return
        try:
            count = self.app.db.export_csv(path, self.records)
        except OSError as exc:
            messagebox.showerror("Export failed", f"The file could not be written:\n\n{exc}", parent=self)
            return
        messagebox.showinfo("Export complete", f"{count} record(s) exported to:\n{path}", parent=self)


class DetailsDialog(tk.Toplevel):
    """Shows every stored value of one classification."""

    def __init__(self, parent, app, record: dict):
        super().__init__(parent)
        self.app, self.record = app, record
        self.title(f"Classification details - {record['record_id']}")
        self.configure(bg=C["bg"])
        self.geometry("640x700")
        self.minsize(520, 480)
        self.transient(parent.winfo_toplevel())

        color = result_color(record["result"])
        head = tk.Frame(self, bg=C["bg"], padx=20, pady=16)
        head.pack(fill="x")
        tk.Label(head, text=f"  {record['result'].upper()}  ", font=font(12, "bold"), bg=color, fg="#FFFFFF").pack(
            side="left")
        tk.Label(head, text=f"{record['result']} Tumor Classification", font=font(15, "bold"), fg=color,
                 bg=C["bg"]).pack(side="left", padx=12)

        info = tk.Frame(self, bg=C["bg"], padx=20)
        info.pack(fill="x")
        for label, value in (("Record ID", record["record_id"]), ("Date & time", record["created_at"]),
                             ("Confidence", _pct(record["probability"])),
                             ("P(Malignant)", _pct(record["prob_malignant"])), ("Model", record["model_name"] or "n/a")):
            line = tk.Frame(info, bg=C["bg"])
            line.pack(fill="x", pady=1)
            tk.Label(line, text=label, width=14, anchor="w", font=font(10), fg=C["muted"], bg=C["bg"]).pack(side="left")
            tk.Label(line, text=value, anchor="w", font=font(10, "bold"), fg=C["text"], bg=C["bg"]).pack(side="left")

        body = tk.Frame(self, bg=C["card"], highlightthickness=1, highlightbackground=C["border"])
        body.pack(fill="both", expand=True, padx=20, pady=14)
        frame, tree = make_treeview(body, [("group", "Group", 90, "w"), ("feature", "Measurement", 200, "w"),
                                           ("value", "Value", 110, "e"), ("unit", "Unit", 120, "w")], height=12)
        frame.pack(fill="both", expand=True)
        for group, features in FEATURE_GROUPS:
            for f in features:
                tree.insert("", "end", values=(group.split()[0], FEATURE_INFO[f]["display"], f"{record[f]:.6g}",
                                               FEATURE_INFO[f]["unit"]))

        tk.Label(self, text=DISCLAIMER_LONG, font=font(8), fg="#FCD34D", bg=C["bg"], justify="left", anchor="w",
                 wraplength=590, padx=20).pack(fill="x")
        buttons = tk.Frame(self, bg=C["bg"], padx=20, pady=12)
        buttons.pack(fill="x")
        FlatButton(buttons, "Export report (.txt)", self.export, kind="secondary").pack(side="left")
        FlatButton(buttons, "Close", self.destroy, kind="ghost").pack(side="left", padx=8)
        self.bind("<Escape>", lambda _e: self.destroy())

    def export(self) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="Save classification report", defaultextension=".txt", initialdir=str(REPORTS_DIR),
            initialfile=f"classification_{self.record['record_id']}.txt", filetypes=[("Text file", "*.txt")])
        if not path:
            return
        try:
            generate_record_report(path, self.record, self.app.metadata)
            messagebox.showinfo("Report saved", f"Saved to:\n{path}", parent=self)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
