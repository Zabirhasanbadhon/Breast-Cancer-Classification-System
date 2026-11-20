"""Central configuration: paths, feature definitions, label mapping and colours.

Everything is expressed relative to the project folder, so the project can be
moved or extracted anywhere without editing any path.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
APP_NAME = "ZaBir 🩺 Breast Cancer Classification System"
APP_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Paths (all relative to the project directory)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DIR / "breast_cancer.csv"
DATASET_INFO_PATH = DATASET_DIR / "dataset_info.json"
RAW_WDBC_PATH = DATASET_DIR / "wdbc.data"  # optional manual download from UCI
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "breast_cancer_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_CSV_PATH = MODELS_DIR / "model_comparison.csv"
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "breast_cancer_history.db"
ASSETS_DIR = PROJECT_ROOT / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
LOGO_PATH = ASSETS_DIR / "logo.png"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --------------------------------------------------------------------------- #
# Target labels.
#   0 = Malignant, 1 = Benign  (this matches scikit-learn's load_breast_cancer
#   encoding; the UCI file stores the letters M / B, which are mapped to 0 / 1
#   explicitly in modules/preprocessing.py).
#   "Malignant" is treated as the POSITIVE class for sensitivity / precision.
# --------------------------------------------------------------------------- #
MALIGNANT = 0
BENIGN = 1
LABELS = {MALIGNANT: "Malignant", BENIGN: "Benign"}
LETTER_TO_CODE = {"M": MALIGNANT, "B": BENIGN}

# --------------------------------------------------------------------------- #
# Feature definitions (30 features, fixed order: mean -> se -> worst)
# --------------------------------------------------------------------------- #
# (key, display name, unit, description)
BASE_FEATURES = [
    ("radius", "Radius", "length units", "Mean distance from the nucleus centre to points on its boundary"),
    ("texture", "Texture", "gray-scale units", "Standard deviation of gray-scale values"),
    ("perimeter", "Perimeter", "length units", "Length of the nucleus boundary"),
    ("area", "Area", "area units", "Area of the nucleus"),
    ("smoothness", "Smoothness", "unitless", "Local variation in radius lengths"),
    ("compactness", "Compactness", "unitless", "perimeter\u00b2 / area \u2212 1.0"),
    ("concavity", "Concavity", "unitless", "Severity of concave portions of the contour"),
    ("concave_points", "Concave Points", "unitless", "Number of concave portions of the contour"),
    ("symmetry", "Symmetry", "unitless", "Symmetry of the nucleus shape"),
    ("fractal_dimension", "Fractal Dimension", "unitless", "'Coastline approximation' \u2212 1"),
]
# (suffix, display suffix, group title, description of the statistic)
SUFFIXES = [
    ("mean", "Mean", "Mean Measurements", "mean over all nuclei in the image"),
    ("se", "SE", "Standard Error Measurements", "standard error of the value"),
    ("worst", "Worst", "Worst Measurements", "mean of the three largest values"),
]
# Features whose value must be strictly > 0 (a nucleus cannot have zero size).
_STRICTLY_POSITIVE = {"radius", "texture", "perimeter", "area"}

FEATURE_NAMES: list[str] = []
FEATURE_INFO: dict[str, dict] = {}
for _suffix, _suffix_display, _group, _stat_desc in SUFFIXES:
    for _key, _display, _unit, _desc in BASE_FEATURES:
        _name = f"{_key}_{_suffix}"
        FEATURE_NAMES.append(_name)
        FEATURE_INFO[_name] = {
            "key": _key,
            "suffix": _suffix,
            "display": f"{_display} {_suffix_display}",
            "unit": _unit,
            "description": f"{_desc} ({_stat_desc})",
            "group": _group,
            "positive_required": _key in _STRICTLY_POSITIVE and _suffix != "se",
        }

# [(group title, [feature names])] in display order
FEATURE_GROUPS = [
    (group, [n for n in FEATURE_NAMES if FEATURE_INFO[n]["suffix"] == suffix])
    for suffix, _sd, group, _desc in SUFFIXES
]
assert len(FEATURE_NAMES) == 30

# --------------------------------------------------------------------------- #
# Disclaimers
# --------------------------------------------------------------------------- #
DISCLAIMER_SHORT = (
    "Educational machine-learning project. Not a medical device. Results are not a diagnosis."
)
DISCLAIMER_LONG = (
    "MEDICAL DISCLAIMER: This application is an educational machine-learning project. "
    "It classifies numerical measurements derived from fine needle aspirate (FNA) images; "
    "it does not analyse raw medical images. It does not confirm cancer and does not replace "
    "a biopsy, pathology examination, clinical evaluation or professional medical diagnosis. "
    "A 'Benign' result does not rule out cancer, and a 'Malignant' result is not a confirmed "
    "diagnosis. Always consult a qualified healthcare professional."
)

# --------------------------------------------------------------------------- #
# Colour palette (shared by the GUI and the Matplotlib charts)
# --------------------------------------------------------------------------- #
COLORS = {
    "bg": "#101827",
    "sidebar": "#172337",
    "card": "#1A2740",
    "card_alt": "#223351",
    "border": "#2A3B59",
    "input": "#0E1626",
    "primary": "#14B8A6",
    "primary_dark": "#0F8F81",
    "secondary": "#3B82F6",
    "benign": "#22C55E",
    "malignant": "#EF4444",
    "warning": "#F59E0B",
    "text": "#FFFFFF",
    "muted": "#9CA3AF",
}
