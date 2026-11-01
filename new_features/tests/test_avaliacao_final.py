"""Testes da avaliação final (nested CV + teste), sem os CSVs reais do All-PPGs."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_final
from evaluate_formulas import EXPECTED_REPR


def _data(participants: range, seed: int, recordings: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for participant in participants:
        for recording in range(recordings):
            signal = participant / 10 + rng.normal(scale=0.1)
            row = {
                "participant_id": f"p{participant:03d}",
                "GLC": 100 + 5 * participant + recording,
            }
            row.update({f"ppg_f{i:02d}": signal * (i % 3) + rng.normal() for i in range(72)})
            row.update({name: signal + rng.normal(scale=0.2) for name in EXPECTED_REPR})
            rows.append(row)
    return pd.DataFrame(rows)


def _small_models():
    return {
        "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [3, 5]}),
        "SVM": (SVR(kernel="linear"), {"model__C": [1.0]}),
    }


@patch.object(evaluate_final, "N_REPEATS", 1)
@patch.object(evaluate_final, "COMPONENTS", (2, 4))
@patch.object(evaluate_final, "_models", _small_models)
class AvaliacaoFinalTest(unittest.TestCase):
    def setUp(self):
        self.train = _data(range(0, 20), seed=0)
        self.test = _data(range(20, 26), seed=1)

    def test_divisoes_externas_separam_participantes(self):
        groups = self.train["participant_id"].to_numpy()
        for fit_idx, val_idx in evaluate_final.outer_splits(groups):
            self.assertFalse(set(groups[fit_idx]) & set(groups[val_idx]))

    def test_escolha_nao_depende_do_teste(self):
        chosen = evaluate_final.choose_models(evaluate_final.nested_cv_table(self.train))
        self.assertEqual(
            list(chosen["representacao"]),
            [evaluate_final.DUMMY, *evaluate_final.FAMILIES],
        )
        shifted = self.test.assign(GLC=self.test["GLC"] + 1000)
        first = evaluate_final.test_table(self.train, self.test, chosen)
        second = evaluate_final.test_table(self.train, shifted, chosen)
        # Same choices and hyperparameters; only the test error changes.
        pd.testing.assert_series_equal(first["modelo"], second["modelo"])
        pd.testing.assert_series_equal(first["hiperparametros"], second["hiperparametros"])
        self.assertTrue((second["mae_teste"] > first["mae_teste"]).all())

    def test_componentes_escolhidos_na_divisao_interna(self):
        chosen = evaluate_final.choose_models(evaluate_final.nested_cv_table(self.train))
        final = evaluate_final.test_table(self.train, self.test, chosen)
        for family in ("PCA", "PLS"):
            params = final.loc[final["representacao"] == family, "hiperparametros"].iloc[0]
            self.assertIn("extract__n_components", params)


if __name__ == "__main__":
    unittest.main()
