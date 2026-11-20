"""Matplotlib figures for the dashboard and the analytics page.

Every function returns a ``matplotlib.figure.Figure`` (no pyplot state), so the GUI can
embed it with ``FigureCanvasTkAgg`` and tests can save it with ``fig.savefig``.

Source of the data is always stated in the title:
* "Dataset"  - statistics of the original WDBC dataset / the training run
* "Saved predictions" - classifications made with this application (SQLite history)
"""
from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .config import COLORS as C
from .config import FEATURE_INFO

BENIGN_COLOR = C["benign"]
MALIGNANT_COLOR = C["malignant"]


# --------------------------------------------------------------------------- helpers
def _new_figure(figsize=(7.0, 4.2)) -> Figure:
    return Figure(figsize=figsize, dpi=100, facecolor=C["card"], layout="constrained")


def _style_axes(ax, title=None, xlabel=None, ylabel=None, grid_axis="y") -> None:
    ax.set_facecolor(C["card"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["border"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=C["border"], alpha=0.6, linewidth=0.8)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=C["text"], fontsize=11, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["muted"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["muted"], fontsize=9)


def _empty_figure(message: str, figsize=(7.0, 4.2)) -> Figure:
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    ax.set_facecolor(C["card"])
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=C["muted"], fontsize=11, wrap=True,
            transform=ax.transAxes)
    return fig


def _legend(ax, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=8, **kwargs)
    for text in leg.get_texts():
        text.set_color(C["muted"])


# --------------------------------------------------------------------------- saved predictions
def fig_prediction_distribution(stats: dict, figsize=(7.0, 4.2)) -> Figure:
    """Benign vs malignant among the predictions saved by this application."""
    total = stats["total"]
    if total == 0:
        return _empty_figure("No predictions saved yet.\nClassify a tumor to see this chart.", figsize)
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    counts = [stats["benign"], stats["malignant"]]
    wedges, _ = ax.pie(
        counts,
        colors=[BENIGN_COLOR, MALIGNANT_COLOR],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": C["card"], "linewidth": 2},
    )
    ax.text(0, 0.06, str(total), ha="center", va="center", color=C["text"], fontsize=22, fontweight="bold")
    ax.text(0, -0.17, "predictions", ha="center", va="center", color=C["muted"], fontsize=9)
    ax.set_title("Saved predictions: Benign vs Malignant", color=C["text"], fontsize=11,
                 fontweight="bold", loc="left")
    labels = [f"Benign: {counts[0]} ({counts[0] / total:.0%})", f"Malignant: {counts[1]} ({counts[1] / total:.0%})"]
    leg = ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False,
                    fontsize=9)
    for text in leg.get_texts():
        text.set_color(C["muted"])
    return fig


def fig_history_over_time(timeline: list, figsize=(7.0, 4.2), max_days: int = 30) -> Figure:
    """Predictions per day (stacked by result) for the last ``max_days`` active days."""
    if not timeline:
        return _empty_figure("No predictions saved yet.\nClassify a tumor to see this chart.", figsize)
    rows = timeline[-max_days:]
    days = [r[0] for r in rows]
    benign = np.array([r[1] for r in rows])
    malignant = np.array([r[2] for r in rows])
    x = np.arange(len(days))
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    ax.bar(x, benign, color=BENIGN_COLOR, label="Benign", width=0.6)
    ax.bar(x, malignant, bottom=benign, color=MALIGNANT_COLOR, label="Malignant", width=0.6)
    _style_axes(ax, "Saved predictions over time (this application)", None, "Predictions per day")
    step = max(1, len(days) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(days[::step], rotation=30, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _legend(ax, loc="upper left")
    return fig


# --------------------------------------------------------------------------- dataset / model
def fig_model_comparison(metadata: dict, figsize=(8.0, 4.6)) -> Figure:
    metrics = [
        ("recall_malignant", "Sensitivity", C["malignant"]),
        ("specificity", "Specificity", C["benign"]),
        ("f1_malignant", "F1 (malignant)", C["warning"]),
        ("roc_auc", "ROC-AUC", C["secondary"]),
        ("accuracy", "Accuracy", C["muted"]),
    ]
    models = metadata["models"]
    selected = metadata["selected_model"]
    names = list(models)
    x = np.arange(len(names))
    width = 0.16
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    lowest = 1.0
    for i, (key, label, color) in enumerate(metrics):
        values = [models[n]["test"][key] for n in names]
        lowest = min(lowest, min(values))
        ax.bar(x + (i - 2) * width, values, width, label=label, color=color, alpha=0.95)
    bottom = max(0.0, np.floor(lowest * 20) / 20 - 0.05)
    ax.set_ylim(bottom, 1.02)
    labels = [("\u2605 " if n == selected else "") + textwrap.fill(n, 12) for n in names]
    _style_axes(ax, "Model comparison - Dataset test split (held-out)", None,
                f"Score (axis starts at {bottom:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    leg = fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False, fontsize=8)
    for text in leg.get_texts():
        text.set_color(C["muted"])
    return fig


def fig_dataset_distribution(metadata: dict, figsize=(7.0, 4.2)) -> Figure:
    ds = metadata["dataset"]
    groups = [("Full dataset", ds["class_counts"]), ("Training split", ds["train_counts"]),
              ("Test split", ds["test_counts"])]
    x = np.arange(len(groups))
    width = 0.36
    benign = [g[1]["Benign"] for g in groups]
    malignant = [g[1]["Malignant"] for g in groups]
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    b1 = ax.bar(x - width / 2, benign, width, label="Benign", color=BENIGN_COLOR)
    b2 = ax.bar(x + width / 2, malignant, width, label="Malignant", color=MALIGNANT_COLOR)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{int(bar.get_height())}",
                    ha="center", va="bottom", color=C["text"], fontsize=9)
    _style_axes(ax, "Class distribution - Dataset (WDBC, not app predictions)", None, "Samples")
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, max(benign) * 1.15)
    _legend(ax, loc="upper right")
    return fig


def fig_confusion_matrix(metadata: dict, figsize=(5.4, 4.6)) -> Figure:
    name = metadata["selected_model"]
    cm = np.array(metadata["models"][name]["confusion_matrix"])
    cmap = LinearSegmentedColormap.from_list("bc", [C["card_alt"], C["primary"]])
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    ax.imshow(cm, cmap=cmap)
    labels = [["True positive", "False negative"], ["False positive", "True negative"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i - 0.08, f"{cm[i, j]}", ha="center", va="center", color=C["text"], fontsize=20,
                    fontweight="bold")
            ax.text(j, i + 0.22, labels[i][j], ha="center", va="center", color=C["text"], fontsize=8, alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Malignant", "Benign"])
    ax.set_yticklabels(["Malignant", "Benign"])
    ax.set_xlabel("Predicted", color=C["muted"], fontsize=9)
    ax.set_ylabel("Actual", color=C["muted"], fontsize=9)
    ax.tick_params(colors=C["muted"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Confusion matrix - {name}\nDataset test split (positive class = Malignant)",
                 color=C["text"], fontsize=10, fontweight="bold", loc="left")
    return fig


def fig_roc_curve(metadata: dict, figsize=(6.4, 4.8)) -> Figure:
    selected = metadata["selected_model"]
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    for name, info in metadata["models"].items():
        if name == selected:
            continue
        ax.plot(info["roc"]["fpr"], info["roc"]["tpr"], color=C["muted"], alpha=0.35, linewidth=1)
    sel = metadata["models"][selected]
    ax.plot(sel["roc"]["fpr"], sel["roc"]["tpr"], color=C["primary"], linewidth=2.6,
            label=f"{selected} (AUC = {sel['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=C["border"], label="Chance")
    ax.plot([], [], color=C["muted"], alpha=0.5, label="Other models")
    _style_axes(ax, "ROC curve - Dataset test split (positive class = Malignant)",
                "False positive rate (1 - specificity)", "True positive rate (sensitivity)", grid_axis="both")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.02)
    _legend(ax, loc="lower right")
    return fig


def fig_feature_importance(metadata: dict, top: int = 15, figsize=(7.4, 5.0)) -> Figure:
    fi = metadata["feature_importance"]
    values = np.array(fi["values"], dtype=float)
    order = np.argsort(values)[::-1][:top][::-1]
    names = [FEATURE_INFO[fi["names"][i]]["display"] for i in order]
    fig = _new_figure(figsize)
    ax = fig.add_subplot(111)
    ax.barh(names, values[order], color=C["primary"])
    _style_axes(ax, f"Feature importance (top {top}) - {metadata['selected_model']}", fi["method"], None,
                grid_axis="x")
    ax.xaxis.label.set_fontsize(8)
    ax.tick_params(axis="y", labelsize=8)
    return fig
