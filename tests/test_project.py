"""Data, feature, split, model, and generated-output contract tests."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.pipeline import (
    CATEGORICAL_FEATURES,
    MODEL_NAMES,
    NUMERIC_FEATURES,
    build_grouped_splits,
    build_model,
    engineer_features,
    passenger_groups,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "raw data"
OUTPUT_DIR = ROOT / "outputs"


class ProjectDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = pd.read_csv(DATA_DIR / "train.csv")
        cls.test = pd.read_csv(DATA_DIR / "test.csv")
        cls.sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    def test_official_file_shapes_and_columns(self) -> None:
        self.assertEqual(self.train.shape, (8693, 14))
        self.assertEqual(self.test.shape, (4277, 13))
        self.assertEqual(list(self.sample.columns), ["PassengerId", "Transported"])
        self.assertTrue(self.sample["PassengerId"].equals(self.test["PassengerId"]))

    def test_feature_engineering_preserves_rows(self) -> None:
        train_features, test_features = engineer_features(
            self.train.drop(columns="Transported"), self.test
        )
        self.assertEqual(len(train_features), len(self.train))
        self.assertEqual(len(test_features), len(self.test))
        self.assertEqual(
            list(train_features.columns), CATEGORICAL_FEATURES + NUMERIC_FEATURES
        )
        self.assertFalse(train_features[NUMERIC_FEATURES].isna().any().any())

    def test_grouped_folds_have_no_group_overlap(self) -> None:
        features = self.train.drop(columns="Transported")
        target = self.train["Transported"].astype(bool).to_numpy()
        groups = passenger_groups(self.train["PassengerId"])
        splits = build_grouped_splits(features, target, groups, folds=5, seed=42)
        validation_rows = []
        for fit_index, validation_index in splits:
            self.assertFalse(set(groups[fit_index]).intersection(groups[validation_index]))
            validation_rows.extend(validation_index.tolist())
        self.assertEqual(sorted(validation_rows), list(range(len(self.train))))

    def test_required_model_lineup_can_be_built(self) -> None:
        self.assertEqual(
            MODEL_NAMES,
            ["logistic_regression", "random_forest", "xgboost", "catboost"],
        )
        for model_name in MODEL_NAMES:
            estimator = build_model(model_name, seed=42, quick=True)
            self.assertTrue(hasattr(estimator, "fit"))
            self.assertTrue(hasattr(estimator, "predict_proba"))

    def test_generated_submission_contract(self) -> None:
        submission_path = OUTPUT_DIR / "submission.csv"
        if not submission_path.exists():
            self.skipTest("Run the training pipeline to generate outputs/submission.csv")
        submission = pd.read_csv(submission_path)
        self.assertEqual(list(submission.columns), ["PassengerId", "Transported"])
        self.assertTrue(submission["PassengerId"].equals(self.test["PassengerId"]))
        self.assertEqual(len(submission), len(self.test))
        self.assertTrue(submission["Transported"].isin([True, False]).all())

    def test_generated_oof_and_fold_contracts(self) -> None:
        oof_path = OUTPUT_DIR / "oof_predictions.csv"
        folds_path = OUTPUT_DIR / "fold_assignments.csv"
        if not oof_path.exists() or not folds_path.exists():
            self.skipTest("Run the training pipeline to generate OOF outputs")
        oof = pd.read_csv(oof_path)
        folds = pd.read_csv(folds_path)
        self.assertEqual(len(oof), len(self.train))
        self.assertEqual(len(folds), len(self.train))
        self.assertTrue(oof["PassengerId"].equals(self.train["PassengerId"]))
        self.assertTrue(np.isfinite(oof.filter(like="probability")).all().all())
        self.assertTrue(
            ((oof.filter(like="probability") >= 0) & (oof.filter(like="probability") <= 1))
            .all()
            .all()
        )
        self.assertEqual(set(folds["Fold"]), {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
