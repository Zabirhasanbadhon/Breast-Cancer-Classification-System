"""Backend tests (no GUI needed).  Run with:  python -m unittest discover -s tests -v

They use the real trained model in models/ - run `python train_model.py` first.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from modules import analytics  # noqa: E402
from modules.config import BENIGN, DATASET_PATH, FEATURE_GROUPS, FEATURE_NAMES, MALIGNANT  # noqa: E402
from modules.database import Database  # noqa: E402
from modules.predictor import Predictor, validate_inputs, validate_one  # noqa: E402
from modules.preprocessing import clean_dataset, load_dataset, split_data  # noqa: E402
from modules.report_generator import (  # noqa: E402
    export_model_comparison,
    generate_record_report,
    generate_summary_report,
)


def as_text(values: dict) -> dict:
    return {k: str(v) for k, v in values.items()}


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()
        cls.metadata = cls.predictor.metadata

    def test_feature_schema(self):
        self.assertEqual(len(FEATURE_NAMES), 30)
        self.assertEqual(self.metadata["feature_names"], FEATURE_NAMES)
        self.assertEqual([f for _, fs in FEATURE_GROUPS for f in fs], FEATURE_NAMES)
        self.assertEqual(list(self.predictor.model.classes_), [MALIGNANT, BENIGN])

    def test_example_values_are_classified_correctly(self):
        mal = self.predictor.predict(self.predictor.validate(as_text(self.predictor.example_values("Malignant"))).values)
        ben = self.predictor.predict(self.predictor.validate(as_text(self.predictor.example_values("Benign"))).values)
        self.assertEqual(mal.label, "Malignant")
        self.assertEqual(ben.label, "Benign")
        self.assertGreater(mal.prob_malignant, 0.5)
        self.assertLess(ben.prob_malignant, 0.5)
        self.assertAlmostEqual(mal.prob_malignant + mal.prob_benign, 1.0, places=6)

    def test_real_dataset_rows_match_their_labels(self):
        """Label mapping end-to-end: every row goes through validate -> predict exactly like the GUI."""
        clean, _ = clean_dataset(load_dataset(DATASET_PATH), log=lambda *_: None)
        correct = 0
        for _, row in clean.iterrows():
            values = self.predictor.validate(as_text({f: row[f] for f in FEATURE_NAMES}))
            self.assertTrue(values.ok, values.errors)
            correct += self.predictor.predict(values.values).class_code == int(row["target"])
        accuracy = correct / len(clean)
        self.assertGreater(accuracy, 0.95, f"accuracy on dataset rows only {accuracy:.3f} (labels reversed?)")

    def test_gui_path_equals_training_path(self):
        """predict() on one row must equal the pipeline's batch output on the same row."""
        clean, _ = clean_dataset(load_dataset(DATASET_PATH), log=lambda *_: None)
        sample = clean.sample(25, random_state=1)
        batch = self.predictor.model.predict_proba(sample[FEATURE_NAMES])[:, 0]
        for p_batch, (_, row) in zip(batch, sample.iterrows()):
            result = self.predictor.predict({f: row[f] for f in FEATURE_NAMES})
            self.assertAlmostEqual(result.prob_malignant, p_batch, places=9)

    def test_metrics_in_metadata_are_sane(self):
        sel = self.metadata["models"][self.metadata["selected_model"]]
        for key in ("accuracy", "precision_malignant", "recall_malignant", "specificity", "f1_malignant", "roc_auc"):
            self.assertTrue(0.8 < sel["test"][key] <= 1.0, key)
        cm = sel["confusion_counts"]
        self.assertEqual(cm["TP"] + cm["FN"], self.metadata["dataset"]["test_counts"]["Malignant"])
        self.assertEqual(cm["TN"] + cm["FP"], self.metadata["dataset"]["test_counts"]["Benign"])
        self.assertEqual(len(self.metadata["models"]), 7)


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_values(self):
        for bad in ("", "   ", "abc", "nan", "inf", "-1", "-0.001"):
            value, error = validate_one("radius_mean", bad)
            self.assertIsNone(value, bad)
            self.assertTrue(error, bad)
        self.assertIsNotNone(validate_one("radius_mean", "0")[1])  # size must be > 0
        self.assertIsNone(validate_one("concavity_mean", "0")[1])  # zero concavity is valid
        self.assertEqual(validate_one("radius_mean", "12,5")[0], 12.5)
        self.assertEqual(validate_one("radius_mean", " 14.2 ")[0], 14.2)

    def test_all_inputs_required(self):
        result = validate_inputs({})
        self.assertEqual(len(result.errors), 30)
        self.assertFalse(result.ok)

    def test_warnings_do_not_block(self):
        predictor = Predictor()
        values = as_text(predictor.example_values("Benign"))
        values["radius_mean"] = "500"  # far outside the training range
        result = predictor.validate(values)
        self.assertTrue(result.ok)
        self.assertTrue(any("outside the range" in w for w in result.warnings))


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "sub" / "history.db")
        self.predictor = Predictor()

    def tearDown(self):
        self.tmp.cleanup()

    def _add(self, label):
        values = self.predictor.validate(as_text(self.predictor.example_values(label))).values
        return self.db.add_prediction(self.predictor.predict(values).to_record())

    def test_roundtrip_search_filter_delete_export(self):
        ids = [self._add("Benign"), self._add("Malignant"), self._add("Malignant")]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(i.startswith("BC-") for i in ids))
        stats = self.db.stats()
        self.assertEqual((stats["total"], stats["benign"], stats["malignant"]), (3, 1, 2))
        self.assertEqual(len(self.db.get_records(result_filter="Malignant")), 2)
        self.assertEqual(len(self.db.get_records(search=ids[0])), 1)
        self.assertEqual(len(self.db.get_records(search="%")), 0)  # wildcard is escaped, not a match-all
        self.assertEqual(len(self.db.get_records(search="'; DROP TABLE predictions;--")), 0)
        self.assertEqual(self.db.count(), 3)
        self.assertEqual(sum(b + m for _, b, m in self.db.timeline()), 3)

        record = self.db.get_records(search=ids[0])[0]
        self.assertEqual(record["result"], "Benign")
        for f in FEATURE_NAMES:
            self.assertAlmostEqual(record[f], self.predictor.example_values("Benign")[f])

        out = Path(self.tmp.name) / "export" / "history.csv"
        self.assertEqual(self.db.export_csv(out), 3)
        with open(out, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(FEATURE_NAMES) - set(rows[0]), set())

        self.assertEqual(self.db.delete_records([record["id"]]), 1)
        self.assertEqual(self.db.count(), 2)
        self.assertIsNone(self.db.get_record(record["id"]))

    def test_reports(self):
        rid = self._add("Malignant")
        summary = generate_summary_report(Path(self.tmp.name) / "summary.txt", self.db, self.predictor.metadata)
        text = summary.read_text(encoding="utf-8")
        self.assertIn(rid, text)
        self.assertIn("MEDICAL DISCLAIMER", text)
        detail = generate_record_report(Path(self.tmp.name) / "d.txt", self.db.get_records()[0])
        self.assertIn("Radius Mean", detail.read_text(encoding="utf-8"))
        comparison = export_model_comparison(Path(self.tmp.name) / "cmp.csv", self.predictor.metadata)
        self.assertEqual(len(pd.read_csv(comparison)), 7)


class AnalyticsTests(unittest.TestCase):
    def test_every_chart_builds_with_and_without_data(self):
        metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
        stats = {"total": 3, "benign": 2, "malignant": 1, "latest": None}
        empty = {"total": 0, "benign": 0, "malignant": 0, "latest": None}
        figures = [
            analytics.fig_prediction_distribution(stats),
            analytics.fig_prediction_distribution(empty),
            analytics.fig_history_over_time([("2026-01-01", 1, 2)]),
            analytics.fig_history_over_time([]),
            analytics.fig_model_comparison(metadata),
            analytics.fig_dataset_distribution(metadata),
            analytics.fig_confusion_matrix(metadata),
            analytics.fig_roc_curve(metadata),
            analytics.fig_feature_importance(metadata),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for i, fig in enumerate(figures):
                fig.savefig(Path(tmp) / f"{i}.png")
                self.assertTrue((Path(tmp) / f"{i}.png").stat().st_size > 1000)


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_handles_bad_data(self):
        df = load_dataset(DATASET_PATH).head(20).copy()
        df["area_mean"] = df["area_mean"].astype(object)  # a CSV with text in a numeric column loads like this
        df.loc[0, "radius_mean"] = -5          # invalid negative
        df.loc[1, "area_mean"] = "oops"        # non-numeric
        df.loc[2, "diagnosis"] = "?"           # unreadable label
        df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # exact duplicate
        clean, report = clean_dataset(df, log=lambda *_: None)
        self.assertEqual(report["negative_values_nullified"], 1)
        self.assertEqual(report["non_numeric_values_nullified"], 1)
        self.assertEqual(report["invalid_label_rows_removed"], 1)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertEqual(len(clean), 20 - 1)
        self.assertTrue(set(clean["target"]) <= {MALIGNANT, BENIGN})

    def test_stratified_split_keeps_class_ratio(self):
        clean, _ = clean_dataset(load_dataset(DATASET_PATH), log=lambda *_: None)
        X_train, X_test, y_train, y_test = split_data(clean)
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), delta=0.02)
        self.assertEqual(len(X_train) + len(X_test), len(clean))
        self.assertFalse(set(X_train.index) & set(X_test.index))


if __name__ == "__main__":
    unittest.main()
