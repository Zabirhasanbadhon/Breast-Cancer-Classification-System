"""Dataset acquisition, inspection, cleaning and the shared preprocessing pipeline.

Training (train_model.py) and prediction (modules/predictor.py) both build their
inputs from ``FEATURE_NAMES`` and use the *same* scikit-learn ``Pipeline`` object,
so feature names, feature order and preprocessing can never drift apart.
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BENIGN,
    DATASET_INFO_PATH,
    DATASET_PATH,
    FEATURE_NAMES,
    LETTER_TO_CODE,
    MALIGNANT,
    RAW_WDBC_PATH,
)

UCI_PAGE = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
SKLEARN_DOC = "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html"

# Published properties of the WDBC dataset (used only as a sanity check).
EXPECTED_ROWS = 569
EXPECTED_MALIGNANT = 212
EXPECTED_BENIGN = 357

Log = Callable[[str], None]


# --------------------------------------------------------------------------- #
# Dataset acquisition
# --------------------------------------------------------------------------- #
MANUAL_DOWNLOAD_HELP = f"""\
Manual dataset download (only needed if you want the UCI file instead of the
scikit-learn fallback):
  1. Open {UCI_PAGE}
  2. Click 'Download' and extract the ZIP.
  3. Copy 'wdbc.data' into the 'dataset' folder of this project.
  4. Run:  python train_model.py --refresh-dataset
     (the file is converted to dataset/breast_cancer.csv automatically)
"""


def normalise_column_name(name: str) -> str:
    """'Concave Points_mean' / 'concave points_mean' -> 'concave_points_mean'."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def parse_wdbc(source) -> pd.DataFrame:
    """Parse the header-less UCI ``wdbc.data`` file (id, diagnosis, 30 features).

    The 30 features in that file are ordered radius..fractal dimension for
    mean, standard error and worst - identical to ``FEATURE_NAMES``.
    """
    columns = ["id", "diagnosis"] + FEATURE_NAMES
    df = pd.read_csv(source, header=None, names=columns)
    if df.shape[1] != 32:
        raise ValueError(f"Unexpected wdbc.data shape: {df.shape}")
    return df[FEATURE_NAMES + ["diagnosis"]]


def download_uci_dataset(timeout: int = 20) -> pd.DataFrame:
    """Download the official ZIP from the UCI repository and parse ``wdbc.data``."""
    request = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next((n for n in archive.namelist() if n.lower().endswith("wdbc.data")), None)
        if member is None:
            raise ValueError("wdbc.data not found inside the downloaded archive")
        with archive.open(member) as handle:
            return parse_wdbc(io.BytesIO(handle.read()))


def dataset_from_sklearn() -> pd.DataFrame:
    """Fallback: scikit-learn's bundled copy of the same WDBC dataset.

    scikit-learn encodes the target as 0 = malignant, 1 = benign. This is
    verified (not assumed) via ``target_names`` before the letters are written.
    """
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    target_names = [str(n).lower() for n in bunch.target_names]
    if target_names != ["malignant", "benign"]:
        raise RuntimeError(f"Unexpected scikit-learn target names: {target_names}")

    rename = {}
    for col in bunch.data.columns:
        if col.startswith("mean "):
            rename[col] = col[5:].replace(" ", "_") + "_mean"
        elif col.endswith(" error"):
            rename[col] = col[:-6].replace(" ", "_") + "_se"
        elif col.startswith("worst "):
            rename[col] = col[6:].replace(" ", "_") + "_worst"
        else:
            raise RuntimeError(f"Unexpected scikit-learn column: {col}")
    features = bunch.data.rename(columns=rename)[FEATURE_NAMES].copy()
    # index 0 -> 'malignant' -> 'M', index 1 -> 'benign' -> 'B'
    letters = bunch.target.map({0: "M", 1: "B"})
    features["diagnosis"] = letters.to_numpy()
    return features


def _check_expected_counts(df: pd.DataFrame, log: Log) -> None:
    n_m = int((df["diagnosis"] == "M").sum())
    n_b = int((df["diagnosis"] == "B").sum())
    if (len(df), n_m, n_b) != (EXPECTED_ROWS, EXPECTED_MALIGNANT, EXPECTED_BENIGN):
        log(
            f"  WARNING: dataset has {len(df)} rows ({n_m} M / {n_b} B); the published WDBC "
            f"dataset has {EXPECTED_ROWS} rows ({EXPECTED_MALIGNANT} M / {EXPECTED_BENIGN} B)."
        )
    else:
        log(f"  Verified: {len(df)} rows, {n_m} malignant (M), {n_b} benign (B) - matches published WDBC.")


def read_dataset_source() -> str:
    if DATASET_INFO_PATH.exists():
        try:
            return json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8")).get("source", "Local file")
        except (OSError, ValueError):
            pass
    return "Local file dataset/breast_cancer.csv"


def ensure_dataset(
    path: Path = DATASET_PATH,
    allow_download: bool = True,
    force_refresh: bool = False,
    log: Log = print,
) -> tuple[Path, str]:
    """Make sure ``dataset/breast_cancer.csv`` exists and return (path, source description).

    Order: existing CSV -> local ``wdbc.data`` -> UCI download -> scikit-learn fallback.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_refresh:
        log(f"  Using existing dataset: {path}")
        return path, read_dataset_source()

    df = None
    source = ""
    if RAW_WDBC_PATH.exists():
        log(f"  Converting local file {RAW_WDBC_PATH.name} ...")
        df = parse_wdbc(RAW_WDBC_PATH)
        source = "UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic), local wdbc.data"
    elif allow_download:
        log(f"  Downloading from UCI: {UCI_ZIP_URL}")
        try:
            df = download_uci_dataset()
            source = f"UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) ({UCI_PAGE})"
            log("  Download successful.")
        except Exception as exc:  # network errors, proxies, format changes ...
            log(f"  Automatic download failed ({type(exc).__name__}: {exc}).")
            log(MANUAL_DOWNLOAD_HELP)
            df = None
    if df is None:
        log("  Falling back to scikit-learn's built-in copy of the WDBC dataset.")
        df = dataset_from_sklearn()
        source = (
            "scikit-learn built-in copy of Breast Cancer Wisconsin (Diagnostic) "
            f"(sklearn.datasets.load_breast_cancer; {SKLEARN_DOC})"
        )

    _check_expected_counts(df, log)
    df.to_csv(path, index=False)
    DATASET_INFO_PATH.write_text(
        json.dumps(
            {"source": source, "created_at": datetime.now().isoformat(timespec="seconds"), "rows": len(df)},
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"  Saved dataset to {path}")
    return path, source


# --------------------------------------------------------------------------- #
# Loading, inspection, cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the CSV with Pandas and validate that it has the expected schema.

    Accepts the file written by :func:`ensure_dataset` and also common Kaggle-style
    variants (``concave points_mean`` etc.; extra ``id`` / ``Unnamed`` columns are ignored).
    """
    df = pd.read_csv(path)
    df.columns = [normalise_column_name(c) for c in df.columns]
    missing = [c for c in FEATURE_NAMES + ["diagnosis"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            "Delete it and run train_model.py again, or see README for manual download steps."
        )
    return df[FEATURE_NAMES + ["diagnosis"]].copy()


def inspect_dataset(df: pd.DataFrame, log: Log = print) -> dict:
    """Print and return basic information: shape, dtypes, missing values, duplicates, classes."""
    diag = df["diagnosis"].astype(str).str.strip().str.upper().str[:1]
    info = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes": {k: int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "missing_per_column": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts_raw": {k: int(v) for k, v in diag.value_counts().items()},
    }
    log(f"  Shape            : {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"  Column dtypes    : {info['dtypes']}")
    log(f"  Missing values   : {info['missing_total']} {info['missing_per_column'] or ''}")
    log(f"  Duplicate rows   : {info['duplicate_rows']}")
    log(f"  Diagnosis counts : {info['class_counts_raw']}  (M = malignant, B = benign)")
    return info


def clean_dataset(df: pd.DataFrame, log: Log = print) -> tuple[pd.DataFrame, dict]:
    """Handle invalid labels / values, remove duplicates and add the numeric ``target``.

    * unreadable diagnosis -> row removed
    * non-numeric, infinite or negative feature values -> set to NaN (negative sizes are
      physically meaningless); NaNs are later imputed with the *training-set* median
      inside the pipeline (no leakage)
    * rows with more than half of their features missing -> removed
    * exact duplicate rows (features + label) -> removed
    """
    report: dict = {"raw_rows": int(len(df))}
    work = df.copy()

    letters = work["diagnosis"].astype(str).str.strip().str.upper().str[:1]
    target = letters.map(LETTER_TO_CODE)
    bad_label = target.isna()
    report["invalid_label_rows_removed"] = int(bad_label.sum())
    work, target = work[~bad_label], target[~bad_label]

    original_na = work[FEATURE_NAMES].isna()
    numeric = work[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    report["non_numeric_values_nullified"] = int((numeric.isna() & ~original_na).sum().sum())
    negatives = numeric < 0
    report["negative_values_nullified"] = int(negatives.sum().sum())
    numeric = numeric.mask(negatives)

    too_sparse = numeric.isna().mean(axis=1) > 0.5
    report["sparse_rows_removed"] = int(too_sparse.sum())
    numeric, target = numeric[~too_sparse], target[~too_sparse]
    report["missing_values_remaining"] = int(numeric.isna().sum().sum())

    clean = numeric.copy()
    clean["target"] = target.astype(int)
    dup = clean.duplicated(subset=FEATURE_NAMES + ["target"])
    report["duplicates_removed"] = int(dup.sum())
    clean = clean[~dup].reset_index(drop=True)
    clean["diagnosis"] = clean["target"].map({MALIGNANT: "M", BENIGN: "B"})
    report["clean_rows"] = int(len(clean))
    report["class_counts"] = {
        "Malignant": int((clean["target"] == MALIGNANT).sum()),
        "Benign": int((clean["target"] == BENIGN).sum()),
    }
    log(
        f"  Removed: {report['invalid_label_rows_removed']} bad labels, "
        f"{report['sparse_rows_removed']} sparse rows, {report['duplicates_removed']} duplicates | "
        f"nullified: {report['non_numeric_values_nullified']} non-numeric, "
        f"{report['negative_values_nullified']} negative values"
    )
    log(f"  Clean dataset    : {report['clean_rows']} rows | classes: {report['class_counts']}")
    return clean, report


def split_data(clean: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split. Preprocessing is fitted later on the training part only."""
    X = clean[FEATURE_NAMES]
    y = clean["target"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# --------------------------------------------------------------------------- #
# Shared pipeline + prediction helpers
# --------------------------------------------------------------------------- #
def make_pipeline(estimator) -> Pipeline:
    """Preprocessing + classifier as one object (fitted on training data only).

    imputer          - median imputation of any missing value
    variance_filter  - drops constant features (none are dropped for this dataset)
    scaler           - standardisation (zero mean, unit variance)
    classifier       - the estimator being trained
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def class_index(model, code: int) -> int:
    return list(model.classes_).index(code)


def malignant_probability(model, X) -> np.ndarray:
    """P(malignant) for every row, taken from the column that belongs to class 0."""
    return model.predict_proba(X)[:, class_index(model, MALIGNANT)]


def predict_labels(model, X) -> np.ndarray:
    """Class codes from predict_proba (argmax). Ties resolve to the first class (Malignant),
    the conservative direction. Training metrics and the GUI both use this exact rule."""
    proba = model.predict_proba(X)
    return np.asarray(model.classes_)[np.argmax(proba, axis=1)]
