"""Settings page: application, model, dataset and database information plus the About section."""
from __future__ import annotations

import platform
import sys
import tkinter as tk

import sklearn

from modules.config import (
    APP_NAME,
    APP_VERSION,
    DATABASE_PATH,
    DISCLAIMER_LONG,
    FEATURE_GROUPS,
    FEATURE_INFO,
    METADATA_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
)

from .theme import C, font
from .widgets import Card, Page, key_value_rows


class SettingsPage(Page):
    title = "Settings & Information"
    subtitle = "Read-only information about this installation, the trained model and the data it was trained on."

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.sections: list[Card] = []

    def on_show(self) -> None:
        for card in self.sections:
            card.destroy()
        self.sections.clear()
        meta, db = self.app.metadata, self.app.db
        ds = meta["dataset"]
        sel = meta["models"][meta["selected_model"]]["test"]
        counts = ds["class_counts"]

        self._section("Application", [
            ("Name", APP_NAME), ("Version", APP_VERSION), ("Python", sys.version.split()[0]),
            ("scikit-learn (installed)", sklearn.__version__), ("Platform", platform.platform()),
            ("Project folder", PROJECT_ROOT)])
        self._section("Model", [
            ("Selected model", meta["selected_model"], C["primary"]), ("Trained at", meta["trained_at"]),
            ("Trained with scikit-learn", meta["sklearn_version"]),
            ("Input features", f"{ds['n_features']} numerical measurements (mean, SE, worst)"),
            ("Target labels", "0 = Malignant, 1 = Benign (positive class for metrics: Malignant)"),
            ("Random seed / test size", f"{meta['random_seed']} / {meta['test_size']:.0%}"),
            ("Model file", MODEL_PATH), ("Metadata file", METADATA_PATH)])
        self._section("Dataset", [
            ("Name", ds["name"]), ("Source", ds["source"]),
            ("Samples (after cleaning)", ds["clean_rows"]),
            ("Malignant / Benign", f"{counts['Malignant']} / {counts['Benign']}"),
            ("Train / test samples",
             f"{sum(ds['train_counts'].values())} / {sum(ds['test_counts'].values())} (stratified split)"),
            ("Missing values / duplicates removed", f"{ds['missing_values_raw']} / {ds['duplicates_removed']}")])
        self._section("Database", [("Location", DATABASE_PATH), ("Saved predictions", db.count()),
                                   ("Stored data", "Generated record ID, 30 measurements, result, probability, timestamp. "
                                                   "No personal information.")])
        self._section("Model evaluation summary (test split)", [
            ("Accuracy", f"{sel['accuracy']:.1%}"), ("Sensitivity (malignant recall)", f"{sel['recall_malignant']:.1%}"),
            ("Specificity", f"{sel['specificity']:.1%}"), ("Precision (malignant)", f"{sel['precision_malignant']:.1%}"),
            ("F1-score (malignant)", f"{sel['f1_malignant']:.1%}"), ("ROC-AUC", f"{sel['roc_auc']:.3f}")])

        glossary = Card(self.content, "Feature guide", "What the 30 measurements mean (computed per nucleus in the FNA image).")
        glossary.pack(fill="x", pady=(0, 12))
        self.sections.append(glossary)
        shown = set()
        for _group, features in FEATURE_GROUPS[:1]:
            for f in features:
                info = FEATURE_INFO[f]
                if info["key"] in shown:
                    continue
                shown.add(info["key"])
                key_value_rows(glossary.body, [(info["display"].replace(" Mean", ""), info["description"].split(" (")[0])],
                               label_width=20)
        tk.Label(glossary.body, text="Each measurement exists three times: Mean, SE (standard error) and Worst "
                 "(mean of the three largest values). Units are image-derived; the dataset does not define absolute "
                 "physical units.", font=font(9), fg=C["muted"], bg=C["card"], anchor="w", justify="left",
                 wraplength=860).pack(fill="x", pady=(8, 0))

        about = Card(self.content, "About")
        about.pack(fill="x", pady=(0, 12))
        self.sections.append(about)
        tk.Label(about.body, text=(f"{APP_NAME} v{APP_VERSION} is an educational project built with Python, Tkinter, "
                                   "Pillow, Pandas, NumPy, scikit-learn, Matplotlib, Joblib and SQLite."),
                 font=font(10), fg=C["text"], bg=C["card"], anchor="w", justify="left", wraplength=860).pack(fill="x")
        tk.Label(about.body, text=DISCLAIMER_LONG, font=font(10), fg="#FCD34D", bg=C["card"], anchor="w",
                 justify="left", wraplength=860).pack(fill="x", pady=(10, 0))

    def _section(self, title, rows):
        card = Card(self.content, title)
        card.pack(fill="x", pady=(0, 12))
        key_value_rows(card.body, rows, label_width=30)
        self.sections.append(card)
