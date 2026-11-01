"""Testes para garantir rastreabilidade da comparação de destaque."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_cv_results import build_comparison, main


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = self.path / "destaques.json"
        self.config.write_text(json.dumps([
            {"representacao": "PLS 2", "arquivo": "pls.csv",
             "features": "PLS 2 componentes", "modelo": "SVM"},
            {"representacao": "4 fórmulas", "arquivo": "manual.csv",
             "features": "só as 4 novas", "modelo": "Regressão linear"},
        ], ensure_ascii=False), encoding="utf-8")
        pd.DataFrame([
            {"features": "PLS 2 componentes", "modelo": "SVM", "mae_medio": 42.4, "rmse_medio": 66.8},
            {"features": "PLS 2 componentes", "modelo": "KNN", "mae_medio": 49.3, "rmse_medio": 66.6},
        ]).to_csv(self.path / "pls.csv", index=False)
        pd.DataFrame([
            {"features": "só as 4 novas", "modelo": "Regressão linear", "mae_medio": 49.0, "rmse_medio": 66.7},
            {"features": "só as 4 novas", "modelo": "SVM", "mae_medio": 48.4, "rmse_medio": 74.2},
        ]).to_csv(self.path / "manual.csv", index=False)

    def test_picks_configured_rows_not_best_mae_or_rmse(self):
        result = build_comparison(self.path, self.config)
        self.assertEqual(result["modelo"].to_list(), ["SVM", "Regressão linear"])
        self.assertEqual(result["mae_mg_dl"].to_list(), [42.4, 49.0])
        self.assertEqual(result["rmse_mg_dl"].to_list(), [66.8, 66.7])

    def test_updates_when_csv_changes(self):
        df = pd.read_csv(self.path / "pls.csv")
        df.loc[df["modelo"] == "SVM", "mae_medio"] = 40.2
        df.to_csv(self.path / "pls.csv", index=False)
        result = build_comparison(self.path, self.config)
        self.assertEqual(result.iloc[0]["mae_mg_dl"], 40.2)

    def test_missing_csv_fails_instead_of_fabricating(self):
        (self.path / "manual.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            build_comparison(self.path, self.config)

    def test_missing_combination_fails(self):
        df = pd.read_csv(self.path / "manual.csv")
        df["modelo"] = ["KNN", "SVM"]
        df.to_csv(self.path / "manual.csv", index=False)
        with self.assertRaises(ValueError):
            build_comparison(self.path, self.config)

    def test_generates_two_graphics_and_csv(self):
        main(["--results-dir", str(self.path), "--config", str(self.config)])
        self.assertEqual(len(pd.read_csv(self.path / "comparacao_representacoes.csv")), 2)
        for name in ("01_comparacao_mae.png", "02_comparacao_rmse.png"):
            self.assertGreater((self.path / name).stat().st_size, 2000)


if __name__ == "__main__":
    unittest.main()
