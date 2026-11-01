"""Testes da comparação PCA sem sinais PPG ou dados reais de participantes."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_pca import _pipeline, _prepare_train, pca_model_table


def _dataset():
    rows = []
    for participant in range(1, 19):
        for recording in range(2):
            row = {
                "participant_id": f"participant_{participant}",
                "GLC": 85.0 + 3.0 * participant + recording,
            }
            for i in range(72):
                row[f"ppg_{i}"] = participant * 0.1 + i * 0.02 + recording * 0.01
            rows.append(row)
    return pd.DataFrame(rows)


class TestPCA(unittest.TestCase):
    def test_pca_inside_pipeline(self):
        pipeline = _pipeline(KNeighborsRegressor(), 8)
        self.assertEqual(
            list(pipeline.named_steps), ["imputer", "scaler", "pca", "model"]
        )
        self.assertIsInstance(pipeline.named_steps["pca"], PCA)
        self.assertEqual(pipeline.named_steps["pca"].n_components, 8)

    def test_only_original_ppg_columns(self):
        df = _dataset()
        df["repr_extra"] = np.arange(len(df))
        X, y, groups = _prepare_train(df)
        self.assertEqual(X.shape[1], 72)
        self.assertEqual(len(y), len(df))
        self.assertEqual(len(np.unique(groups)), 18)

    def test_cv_comparison_and_reference(self):
        df = _dataset()
        fast = {"KNN": (KNeighborsRegressor(), {"model__n_neighbors": [3]})}
        with patch("evaluate_pca._models", return_value=fast):
            table = pca_model_table(df, components=(4, 8))
        self.assertEqual(table["features"].tolist(), [
            "nenhuma (média)", "PCA 4 componentes", "PCA 8 componentes"
        ])
        self.assertTrue(np.isfinite(table["mae_medio"]).all())
        self.assertTrue(np.isfinite(table["rmse_medio"]).all())

    def test_invalid_components(self):
        with self.assertRaisesRegex(ValueError, "entre 1 e 72"):
            pca_model_table(_dataset(), components=(73,))


if __name__ == "__main__":
    unittest.main()
