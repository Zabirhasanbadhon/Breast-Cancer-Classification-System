"""CSV / text exports: model comparison, summary report and single-record report."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .config import (
    APP_NAME,
    APP_VERSION,
    DISCLAIMER_LONG,
    FEATURE_GROUPS,
    FEATURE_INFO,
)

METRIC_COLUMNS = [
    ("accuracy", "Accuracy"),
    ("precision_malignant", "Precision (Malignant)"),
    ("recall_malignant", "Sensitivity (Malignant recall)"),
    ("specificity", "Specificity"),
    ("f1_malignant", "F1 (Malignant)"),
    ("roc_auc", "ROC-AUC"),
]


def comparison_rows(metadata: dict) -> list[dict]:
    """One flat dict per model: held-out test metrics + cross-validation means."""
    selected = metadata.get("selected_model")
    rows = []
    for name, info in metadata["models"].items():
        row = {"Model": name, "Selected": "Yes" if name == selected else ""}
        for key, label in METRIC_COLUMNS:
            row[f"Test {label}"] = round(info["test"][key], 4)
        counts = info.get("confusion_counts", {})
        row["Test False Negatives"] = counts.get("FN", "")
        row["Test False Positives"] = counts.get("FP", "")
        for key, label in METRIC_COLUMNS:
            cv = info["cv"][key]
            row[f"CV {label} (mean)"] = round(cv["mean"], 4)
            row[f"CV {label} (std)"] = round(cv["std"], 4)
        row["Selection Score"] = round(info["selection_score"], 4)
        rows.append(row)
    rows.sort(key=lambda r: -r["Selection Score"])
    return rows


def export_model_comparison(path, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(metadata)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt_prob(value) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _model_block(metadata: dict) -> list[str]:
    name = metadata["selected_model"]
    test = metadata["models"][name]["test"]
    ds = metadata["dataset"]
    lines = [
        f"Selected model      : {name}",
        f"Trained at          : {metadata['trained_at']}",
        f"scikit-learn        : {metadata['sklearn_version']}",
        f"Dataset             : {ds['name']} ({ds['clean_rows']} samples after cleaning)",
        f"Evaluation          : held-out test split ({ds['test_counts']}), positive class = Malignant",
        f"  Accuracy          : {test['accuracy']:.4f}",
        f"  Precision         : {test['precision_malignant']:.4f}",
        f"  Sensitivity       : {test['recall_malignant']:.4f}",
        f"  Specificity       : {test['specificity']:.4f}",
        f"  F1-score          : {test['f1_malignant']:.4f}",
        f"  ROC-AUC           : {test['roc_auc']:.4f}",
    ]
    return lines


def generate_summary_report(path, db, metadata: dict, recent: int = 25) -> Path:
    """Plain-text classification summary: model metrics + statistics of saved predictions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = db.stats()
    records = db.get_records(limit=recent)
    total = stats["total"]
    pct = lambda n: f"{(n / total * 100):.1f}%" if total else "-"  # noqa: E731
    lines = [
        APP_NAME.upper() + " - CLASSIFICATION SUMMARY REPORT",
        "=" * 72,
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Version   : {APP_VERSION}",
        "",
        DISCLAIMER_LONG,
        "",
        "MODEL",
        "-" * 72,
        *_model_block(metadata),
        "",
        "SAVED PREDICTIONS (made with this application - not dataset statistics)",
        "-" * 72,
        f"Total predictions   : {total}",
        f"Benign              : {stats['benign']} ({pct(stats['benign'])})",
        f"Malignant           : {stats['malignant']} ({pct(stats['malignant'])})",
        "",
        f"MOST RECENT {len(records)} PREDICTIONS",
        "-" * 72,
        f"{'Record ID':<22}{'Date & time':<21}{'Result':<11}{'Confidence':<12}Model",
    ]
    for r in records:
        lines.append(
            f"{r['record_id']:<22}{r['created_at']:<21}{r['result']:<11}{_fmt_prob(r['probability']):<12}{r['model_name'] or ''}"
        )
    if not records:
        lines.append("(no predictions saved yet)")
    lines += ["", "End of report.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_record_report(path, record: dict, metadata: dict | None = None) -> Path:
    """Plain-text report for ONE saved classification (all 30 measurements)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        APP_NAME.upper() + " - CLASSIFICATION DETAILS",
        "=" * 72,
        f"Record ID           : {record['record_id']}",
        f"Date & time         : {record['created_at']}",
        f"Classification      : {record['result']}",
        f"Predicted-class prob: {_fmt_prob(record.get('probability'))}",
        f"P(Malignant)        : {_fmt_prob(record.get('prob_malignant'))}",
        f"Model               : {record.get('model_name') or 'n/a'}",
        "",
        DISCLAIMER_LONG,
        "",
        "TUMOR MEASUREMENTS",
        "-" * 72,
    ]
    for group, features in FEATURE_GROUPS:
        lines.append(group)
        for f in features:
            info = FEATURE_INFO[f]
            lines.append(f"  {info['display']:<26}{record[f]:>12.5g}   ({info['unit']})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
