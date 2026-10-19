"""Small data-contract tests for the course project."""

from pathlib import Path
import unittest

import pandas as pd

from src.train import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "raw data"


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


if __name__ == "__main__":
    unittest.main()

