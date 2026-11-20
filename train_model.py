"""Train, evaluate and save the breast-cancer classification model.

Usage:
    python train_model.py                      # normal run
    python train_model.py --refresh-dataset    # re-create dataset/breast_cancer.csv
    python train_model.py --no-download        # never touch the network

Outputs:
    dataset/breast_cancer.csv
    models/breast_cancer_model.pkl     (fitted scikit-learn Pipeline)
    models/model_metadata.json         (metrics, ROC data, feature stats, ...)
    models/model_comparison.csv

Evaluation strategy (important for a medical-style classifier)
--------------------------------------------------------------
* "Malignant" is the POSITIVE class. Sensitivity = malignant recall, because missing a
  malignant case is the costly error; specificity is reported to show the false-alarm trade-off.
* Every model is scored with stratified 5-fold cross-validation on the TRAINING split only.
  The held-out test split is never used for model selection - it is only used to report
  honest final numbers.
* Selection score = 0.40 * sensitivity + 0.20 * specificity + 0.20 * F1(malignant)
                    + 0.20 * ROC-AUC            (all cross-validated means)
  Models whose cross-validated sensitivity is below 0.90 are not eligible while at least
  one model reaches that floor. Accuracy is reported but deliberately NOT used to choose.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from modules.config import (
    APP_VERSION,
    BENIGN,
    COMPARISON_CSV_PATH,
    DATASET_PATH,
    FEATURE_NAMES,
    LABELS,
    MALIGNANT,
    METADATA_PATH,
    MODEL_PATH,
)
from modules.preprocessing import (
    clean_dataset,
    ensure_dataset,
    inspect_dataset,
    load_dataset,
    make_pipeline,
    malignant_probability,
    predict_labels,
    split_data,
)

SEED = 42
CV_FOLDS = 5
MIN_CV_SENSITIVITY = 0.90
SELECTION_WEIGHTS = {"recall_malignant": 0.40, "specificity": 0.20, "f1_malignant": 0.20, "roc_auc": 0.20}


def build_models(seed: int = SEED) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        # SVC has no native probabilities; calibrate its scores (sigmoid) instead of the
        # deprecated SVC(probability=True) option.
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=seed),
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# --------------------------------------------------------------------------- #
# Metrics (positive class = Malignant)
# --------------------------------------------------------------------------- #
def evaluate_model(model, X, y) -> dict:
    y = np.asarray(y)
    y_pred = predict_labels(model, X)
    p_malignant = malignant_probability(model, X)
    # rows = true [Malignant, Benign], columns = predicted [Malignant, Benign]
    cm = confusion_matrix(y, y_pred, labels=[MALIGNANT, BENIGN])
    tp, fn, fp, tn = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr, tpr, _ = roc_curve(y == MALIGNANT, p_malignant)
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision_malignant": precision,
            "recall_malignant": recall,
            "specificity": _safe_div(tn, tn + fp),
            "f1_malignant": _safe_div(2 * precision * recall, precision + recall),
            "roc_auc": float(roc_auc_score(y == MALIGNANT, p_malignant)),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_counts": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "roc": {"fpr": [round(float(v), 5) for v in fpr], "tpr": [round(float(v), 5) for v in tpr]},
        "classification_report": classification_report(
            y,
            y_pred,
            labels=[MALIGNANT, BENIGN],
            target_names=[LABELS[MALIGNANT], LABELS[BENIGN]],
            digits=4,
            zero_division=0,
        ),
    }


def _cv_scorers() -> dict:
    """Scorers that use the same decision rule as the deployed application."""

    def wrap(fn):
        def scorer(est, X, y):
            return fn(np.asarray(y), predict_labels(est, X), malignant_probability(est, X))

        return scorer

    def cm_parts(y, yp):
        cm = confusion_matrix(y, yp, labels=[MALIGNANT, BENIGN])
        return cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    def f1(y, yp, p):
        tp, fn, fp, _ = cm_parts(y, yp)
        return _safe_div(2 * tp, 2 * tp + fp + fn)

    return {
        "accuracy": wrap(lambda y, yp, p: accuracy_score(y, yp)),
        "precision_malignant": wrap(lambda y, yp, p: _safe_div(cm_parts(y, yp)[0], cm_parts(y, yp)[0] + cm_parts(y, yp)[2])),
        "recall_malignant": wrap(lambda y, yp, p: _safe_div(cm_parts(y, yp)[0], cm_parts(y, yp)[0] + cm_parts(y, yp)[1])),
        "specificity": wrap(lambda y, yp, p: _safe_div(cm_parts(y, yp)[3], cm_parts(y, yp)[3] + cm_parts(y, yp)[2])),
        "f1_malignant": wrap(f1),
        "roc_auc": wrap(lambda y, yp, p: roc_auc_score(y == MALIGNANT, p)),
    }


def selection_score(cv_means: dict) -> float:
    return float(sum(w * cv_means[k] for k, w in SELECTION_WEIGHTS.items()))


def compute_feature_importance(pipeline, X_test, y_test, seed: int) -> dict:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values, method = classifier.feature_importances_, "Impurity-based importance (feature_importances_)"
    elif hasattr(classifier, "coef_"):
        values, method = np.abs(classifier.coef_[0]), "Absolute standardised coefficient"
    else:
        result = permutation_importance(
            pipeline, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=seed, n_jobs=1
        )
        values, method = result.importances_mean, "Permutation importance (ROC-AUC drop on the test split)"
    values = np.asarray(values, dtype=float)
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError("Feature importance length does not match the feature list")
    return {"method": method, "names": list(FEATURE_NAMES), "values": [round(float(v), 6) for v in values]}


def _correlated_pairs(X: pd.DataFrame, threshold: float = 0.95, top: int = 10) -> list:
    corr = X.corr().abs()
    pairs = []
    for i, a in enumerate(FEATURE_NAMES):
        for b in FEATURE_NAMES[i + 1 :]:
            if corr.loc[a, b] >= threshold:
                pairs.append([a, b, round(float(corr.loc[a, b]), 4)])
    pairs.sort(key=lambda p: -p[2])
    return pairs[:top]


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def run_training(
    model_path=MODEL_PATH,
    metadata_path=METADATA_PATH,
    comparison_path=COMPARISON_CSV_PATH,
    allow_download: bool = True,
    refresh_dataset: bool = False,
    test_size: float = 0.2,
    seed: int = SEED,
    log=print,
) -> dict:
    np.random.seed(seed)
    log("=" * 78)
    log(" Breast Cancer Classification System - model training")
    log("=" * 78)

    log("\n[1/6] Dataset")
    csv_path, source = ensure_dataset(DATASET_PATH, allow_download, refresh_dataset, log)
    raw = load_dataset(csv_path)

    log("\n[2/6] Inspecting dataset")
    raw_info = inspect_dataset(raw, log)

    log("\n[3/6] Cleaning and splitting")
    clean, clean_report = clean_dataset(raw, log)
    X_train, X_test, y_train, y_test = split_data(clean, test_size, seed)
    train_counts = {LABELS[c]: int((y_train == c).sum()) for c in (MALIGNANT, BENIGN)}
    test_counts = {LABELS[c]: int((y_test == c).sum()) for c in (MALIGNANT, BENIGN)}
    log(f"  Train: {len(X_train)} rows {train_counts} | Test: {len(X_test)} rows {test_counts}")
    log("  Target encoding: 0 = Malignant, 1 = Benign (verified from M/B letters; positive class = Malignant)")
    corr_pairs = _correlated_pairs(X_train)
    log(f"  Highly correlated feature pairs (|r| >= 0.95): {len(corr_pairs)} shown in metadata "
        "(all 30 features are kept so the GUI form matches the model input)")

    log(f"\n[4/6] Cross-validation ({CV_FOLDS}-fold, stratified, training split only)")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scorers = _cv_scorers()
    results: dict = {}
    fitted: dict = {}
    for name, estimator in build_models(seed).items():
        pipeline = make_pipeline(estimator)
        cv = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scorers, n_jobs=1, error_score="raise")
        cv_summary = {
            metric: {"mean": float(np.mean(cv[f"test_{metric}"])), "std": float(np.std(cv[f"test_{metric}"]))}
            for metric in scorers
        }
        cv_means = {m: v["mean"] for m, v in cv_summary.items()}
        pipeline.fit(X_train, y_train)
        if list(pipeline.classes_) != [MALIGNANT, BENIGN]:
            raise RuntimeError(f"Unexpected class order {list(pipeline.classes_)} for {name}")
        if not pipeline.named_steps["variance_filter"].get_support().all():
            raise RuntimeError("The variance filter removed a feature; the GUI expects all 30 features.")
        evaluation = evaluate_model(pipeline, X_test, y_test)
        results[name] = {
            "cv": cv_summary,
            "selection_score": selection_score(cv_means),
            "test": evaluation["metrics"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "confusion_counts": evaluation["confusion_counts"],
            "roc": evaluation["roc"],
            "classification_report": evaluation["classification_report"],
        }
        fitted[name] = pipeline
        log(
            f"  {name:<24} CV sens {cv_means['recall_malignant']:.3f} | spec {cv_means['specificity']:.3f} | "
            f"AUC {cv_means['roc_auc']:.3f} | score {results[name]['selection_score']:.4f}"
        )

    log("\n[5/6] Model selection")
    eligible = [n for n, r in results.items() if r["cv"]["recall_malignant"]["mean"] >= MIN_CV_SENSITIVITY]
    pool = eligible or list(results)
    ranking = sorted(
        pool,
        key=lambda n: (results[n]["selection_score"], results[n]["cv"]["roc_auc"]["mean"]),
        reverse=True,
    )
    selected = ranking[0]
    log(f"  Eligible (CV sensitivity >= {MIN_CV_SENSITIVITY}): {len(eligible)} of {len(results)} models")
    log(f"  Selected model: {selected}")

    log("\n  Held-out TEST results (positive class = Malignant)")
    table = pd.DataFrame(
        {
            name: {
                "Accuracy": r["test"]["accuracy"],
                "Precision": r["test"]["precision_malignant"],
                "Sensitivity": r["test"]["recall_malignant"],
                "Specificity": r["test"]["specificity"],
                "F1": r["test"]["f1_malignant"],
                "ROC-AUC": r["test"]["roc_auc"],
                "FN": r["confusion_counts"]["FN"],
                "FP": r["confusion_counts"]["FP"],
            }
            for name, r in results.items()
        }
    ).T
    log(table.round(4).to_string())
    log("\n  Classification report - " + selected)
    log(results[selected]["classification_report"])

    best = fitted[selected]
    importance = compute_feature_importance(best, X_test, y_test, seed)

    # --- sanity checks: label mapping must not be reversed -------------------------
    check = X_test.copy()
    preds = predict_labels(best, check)
    mal_recall = float(np.mean(preds[np.asarray(y_test) == MALIGNANT] == MALIGNANT))
    ben_recall = float(np.mean(preds[np.asarray(y_test) == BENIGN] == BENIGN))
    if mal_recall < 0.8 or ben_recall < 0.8:
        raise RuntimeError("Label-mapping sanity check failed (predictions look reversed).")
    log(f"\n  Sanity check passed: malignant rows -> Malignant {mal_recall:.1%}, benign rows -> Benign {ben_recall:.1%}")

    log("\n[6/6] Saving")
    feature_stats = {}
    for f in FEATURE_NAMES:
        feature_stats[f] = {
            "min": float(clean[f].min()),
            "max": float(clean[f].max()),
            "mean": float(clean[f].mean()),
            "median_malignant": float(clean.loc[clean["target"] == MALIGNANT, f].median()),
            "median_benign": float(clean.loc[clean["target"] == BENIGN, f].median()),
        }

    metadata = {
        "app_version": APP_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "random_seed": seed,
        "test_size": test_size,
        "cv_folds": CV_FOLDS,
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "source": source,
            "path": "dataset/breast_cancer.csv",
            "raw_rows": raw_info["rows"],
            "clean_rows": clean_report["clean_rows"],
            "missing_values_raw": raw_info["missing_total"],
            "duplicates_removed": clean_report["duplicates_removed"],
            "cleaning_report": clean_report,
            "class_counts": clean_report["class_counts"],
            "train_counts": train_counts,
            "test_counts": test_counts,
            "n_features": len(FEATURE_NAMES),
        },
        "feature_names": list(FEATURE_NAMES),
        "label_map": {str(k): v for k, v in LABELS.items()},
        "positive_class_for_metrics": LABELS[MALIGNANT],
        "feature_stats": feature_stats,
        "correlated_pairs": corr_pairs,
        "selection": {
            "strategy": (
                "Stratified 5-fold CV on the training split. Score = 0.40*sensitivity + 0.20*specificity "
                "+ 0.20*F1(malignant) + 0.20*ROC-AUC. Models with CV sensitivity < 0.90 are ineligible "
                "while at least one model reaches it. Accuracy is reported but not used for selection. "
                "The test split is used only for the final report."
            ),
            "weights": SELECTION_WEIGHTS,
            "min_cv_sensitivity": MIN_CV_SENSITIVITY,
            "selected_model": selected,
            "ranking": [{"model": n, "score": results[n]["selection_score"]} for n in ranking],
        },
        "selected_model": selected,
        "models": results,
        "feature_importance": importance,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    from modules.report_generator import export_model_comparison

    export_model_comparison(comparison_path, metadata)
    log(f"  Model    -> {model_path}")
    log(f"  Metadata -> {metadata_path}")
    log(f"  Compare  -> {comparison_path}")
    log("\nDone. Start the application with:  python app.py")
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the breast cancer classification model.")
    parser.add_argument("--refresh-dataset", action="store_true", help="re-create dataset/breast_cancer.csv")
    parser.add_argument("--no-download", action="store_true", help="never use the network (uses local/fallback data)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        run_training(
            allow_download=not args.no_download,
            refresh_dataset=args.refresh_dataset,
            test_size=args.test_size,
            seed=args.seed,
        )
    except Exception as exc:  # show a readable message instead of a bare traceback
        print(f"\nTraining failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
