"""Fonts and colours shared by every GUI module."""
from __future__ import annotations

import sys

from modules.config import COLORS as C  # noqa: F401  (re-exported)

if sys.platform.startswith("win"):
    FONT_FAMILY = "Segoe UI"
    MONO_FAMILY = "Consolas"
elif sys.platform == "darwin":
    FONT_FAMILY = "Helvetica Neue"
    MONO_FAMILY = "Menlo"
else:
    FONT_FAMILY = "DejaVu Sans"
    MONO_FAMILY = "DejaVu Sans Mono"


def font(size: int = 10, weight: str = "normal") -> tuple:
    return (FONT_FAMILY, size, weight)


def mono(size: int = 10) -> tuple:
    return (MONO_FAMILY, size)


def result_color(label: str) -> str:
    return C["benign"] if label == "Benign" else C["malignant"]
