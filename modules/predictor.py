"""Loads the trained model and turns validated measurements into a classification."""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from .config import (
    BENIGN,
    FEATURE_INFO,
    FEATURE_NAMES,
    LABELS,
    MALIGNANT,
    METADATA_PATH,
    MODEL_PATH,
)


class ModelNotFoundError(RuntimeError):
    """The trained model / metadata files do not exist yet."""


class ModelLoadError(RuntimeError):
    """The model exists but cannot be used (corrupt, incompatible version, wrong schema)."""


# --------------------------------------------------------------------------- #
# Input validation (no model needed, so it is easy to test)
# --------------------------------------------------------------------------- #
def validate_one(feature: str, text) -> tuple[float | None, str | None]:
    """Validate one raw text value. Returns (value, None) or (None, error message)."""
    info = FEATURE_INFO[feature]
    name = info["display"]
    text = "" if text is None else str(text).strip()
    if not text:
        return None, f"{name}: a value is required"
    candidate = text
    if candidate.count(",") == 1 and "." not in candidate:  # accept "12,5" as 12.5
        candidate = candidate.replace(",", ".")
    try:
        value = float(candidate)
    except ValueError:
        return None, f"{name}: '{text}' is not a valid number"
    if not math.isfinite(value):
        return None, f"{name}: '{text}' is not a finite number"
    if info["positive_required"] and value <= 0:
        return None, f"{name}: must be greater than 0"
    if value < 0:
        return None, f"{name}: cannot be negative"
    return value, None


@dataclass
class ValidationResult:
    values: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)  # feature -> message
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_inputs(raw: Mapping[str, object], feature_stats: Mapping | None = None) -> ValidationResult:
    """Validate all 30 inputs; also produce soft warnings (never blocking)."""
    result = ValidationResult()
    for feature in FEATURE_NAMES:
        value, error = validate_one(feature, raw.get(feature))
        if error:
            result.errors[feature] = error
        else:
            result.values[feature] = value
    if result.errors:
        return result

    if feature_stats:
        for feature, value in result.values.items():
            stats = feature_stats.get(feature)
            if stats and not (stats["min"] <= value <= stats["max"]):
                result.warnings.append(
                    f"{FEATURE_INFO[feature]['display']} = {value:g} is outside the range seen in the "
                    f"training data ({stats['min']:.4g} to {stats['max']:.4g}); the prediction may be unreliable."
                )
    for key in {FEATURE_INFO[f]["key"] for f in FEATURE_NAMES}:
        mean_v, worst_v = result.values[f"{key}_mean"], result.values[f"{key}_worst"]
        if worst_v < mean_v:
            result.warnings.append(
                f"{FEATURE_INFO[key + '_worst']['display']} ({worst_v:g}) is smaller than "
                f"{FEATURE_INFO[key + '_mean']['display']} ({mean_v:g}); 'worst' is normally >= 'mean'."
            )
    return result


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    label: str
    class_code: int
    probability: float | None  # probability of the predicted class
    prob_malignant: float | None
    prob_benign: float | None
    timestamp: datetime
    values: dict
    model_name: str

    @property
    def is_malignant(self) -> bool:
        return self.class_code == MALIGNANT

    def to_record(self) -> dict:
        return {
            "created_at": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "result": self.label,
            "class_code": self.class_code,
            "probability": self.probability,
            "prob_malignant": self.prob_malignant,
            "model_name": self.model_name,
            "values": dict(self.values),
        }


class Predictor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        if not model_path.exists() or not metadata_path.exists():
            raise ModelNotFoundError(
                "The trained model was not found.\n\nTrain it first by running:\n\n    python train_model.py"
            )
        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.model = joblib.load(model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"The saved model could not be loaded ({type(exc).__name__}: {exc}).\n\n"
                "Re-create it with:\n\n    python train_model.py"
            ) from exc
        if any("version" in str(w.message).lower() for w in caught):
            raise ModelLoadError(
                "The saved model was created with a different scikit-learn version "
                f"(trained with {self.metadata.get('sklearn_version', '?')}).\n\n"
                "Re-create it for your environment with:\n\n    python train_model.py"
            )
        self._verify()

    def _verify(self) -> None:
        if self.metadata.get("feature_names") != FEATURE_NAMES:
            raise ModelLoadError("Feature names in the model metadata do not match the application. Re-train the model.")
        if list(self.model.classes_) != [MALIGNANT, BENIGN]:
            raise ModelLoadError(f"Unexpected model classes {list(self.model.classes_)}. Re-train the model.")
        if self.metadata.get("label_map") != {str(k): v for k, v in LABELS.items()}:
            raise ModelLoadError("Label mapping in the metadata does not match the application. Re-train the model.")

    # ------------------------------------------------------------------ API
    @property
    def model_name(self) -> str:
        return self.metadata["selected_model"]

    @property
    def feature_stats(self) -> dict:
        return self.metadata["feature_stats"]

    def validate(self, raw: Mapping[str, object]) -> ValidationResult:
        return validate_inputs(raw, self.feature_stats)

    def predict(self, values: Mapping[str, float]) -> PredictionResult:
        """Classify one set of 30 validated measurements with the real trained pipeline."""
        frame = pd.DataFrame([[float(values[f]) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        classes = list(self.model.classes_)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(frame)[0]
            code = int(classes[int(np.argmax(proba))])  # same rule as training (ties -> Malignant)
            p_mal = float(proba[classes.index(MALIGNANT)])
            p_ben = float(proba[classes.index(BENIGN)])
            p_pred = p_mal if code == MALIGNANT else p_ben
        else:  # pragma: no cover - every candidate model supports predict_proba
            code = int(self.model.predict(frame)[0])
            p_mal = p_ben = p_pred = None
        return PredictionResult(
            label=LABELS[code],
            class_code=code,
            probability=p_pred,
            prob_malignant=p_mal,
            prob_benign=p_ben,
            timestamp=datetime.now(),
            values={f: float(values[f]) for f in FEATURE_NAMES},
            model_name=self.model_name,
        )

    def example_values(self, label: str) -> dict:
        """Class-median measurements from the dataset, used by the 'Load example' buttons."""
        key = "median_malignant" if label == "Malignant" else "median_benign"
        return {f: self.feature_stats[f][key] for f in FEATURE_NAMES}
