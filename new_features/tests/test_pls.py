"""Testes do PLS supervisionado, sem utilizar dados reais de participantes."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_pls import PLSFeatureExtractor, _pipeline, pls_model_table
from evaluate_pca import _prepare_train
from evaluate_formulas import _feature_sets, independent_test, model_table
from feature_representation import apply_representation


def _dataset():
    rng = np.random.default_rng(16)
    data = []
    for participant in range(1, 21):
        for recording in range(2):
            row = {
                "participant_id": f"p{participant:03d}",
                "GLC": 75 + 3 * participant + rng.normal(0, 2),
            }
            for k in range(72):
                row[f"ppg_{k}"] = (0.1 * participant * np.cos(k / 7)
                                   + rng.normal(0, 1))
            data.append(row)
    return pd.DataFrame(data)


class TestPLS(unittest.TestCase):
    def test_transformer_returns_only_x_components(self):
        df = _dataset()
        X, y, _ = _prepare_train(df)
        transform = PLSFeatureExtractor(n_components=4)
        scores = transform.fit_transform(X, y)
        self.assertEqual(scores.shape, (len(df), 4))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(transform.transform(X[:2]).shape, (2, 4))

    def test_pipeline_and_supervised_training(self):
        pipeline = _pipeline(SVR(), n_components=4)
        self.assertEqual(
            list(pipeline.named_steps), ["imputer", "scaler", "pls", "model"]
        )
        df = _dataset()
        X, y, _ = _prepare_train(df)
        pipeline.fit(X, y)
        self.assertEqual(pipeline.predict(X[:2]).shape, (2,))

    def test_pls_cv_and_rmse_columns(self):
        fast = {"KNN": (
            KNeighborsRegressor(), {"model__n_neighbors": [3]}
        )}
        with patch("evaluate_pls._models", return_value=fast):
            table = pls_model_table(_dataset(), components=(2, 4))
        self.assertEqual(table["features"].tolist(), [
            "nenhuma (média)", "PLS 2 componentes", "PLS 4 componentes"
        ])
        for name in ("mae_medio", "rmse_medio", "desvio_entre_folds",
                     "desvio_rmse_entre_folds"):
            self.assertTrue(np.isfinite(table[name]).all(), name)

    def test_invalid_components_and_without_target(self):
        with self.assertRaisesRegex(ValueError, "glicose"):
            PLSFeatureExtractor(2).fit(np.ones((8, 3)))
        with self.assertRaisesRegex(ValueError, "entre 1 e 72"):
            pls_model_table(_dataset(), components=(73,))

    def test_feature_selection_is_not_an_option(self):
        # O avaliador das representações manuais mantém somente o baseline
        # de 72 originais e as duas representações com colunas construídas.
        df = _dataset()
        df["duration"] = 2000
        df["ppg_b2_N_BEATS"] = 35
        df["ppg_b2_width_median"] = 0.2
        df["ppg_b2_T_median"] = 0.5
        df["ppg_b2_Tr_median"] = 0.1
        df["ppg_p1_peak_interval_std"] = 0.04
        df["ppg_p1_hr_mean"] = 65
        transformed, _ = apply_representation(df, sampling_rate=500)
        names = _feature_sets(transformed)
        self.assertEqual(len(names), 3)
        self.assertFalse(any("ingredientes" in name for name in names))
        # Nos casos explicitamente avaliados no teste, seleciona-se entre
        # representações construídas, nunca o baseline de 72 originais.
        dummy_rows = pd.DataFrame([
            {"features": "72 originais", "modelo": "KNN", "rmse_medio": 1.0, "mae_medio": 1.0},
            {"features": "só as 4 novas", "modelo": "KNN", "rmse_medio": 3.0, "mae_medio": 2.0},
        ])
        test = transformed.iloc[0:4].copy()
        train = transformed.iloc[4:].copy()
        with patch("evaluate_formulas._models", return_value={"KNN": (
            KNeighborsRegressor(), {"model__n_neighbors": [3]})}):
            final = independent_test(train, test, dummy_rows)
        self.assertEqual(final.iloc[0]["features"], "só as 4 novas")


if __name__ == "__main__":
    unittest.main()
