"""Testes rápidos, sem precisar dos sinais nem dos CSVs reais do All-PPGs."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_formulas import _check_data, independent_test, model_table
from split_integrity import resolve_official_overlap
from feature_representation import RATIO_RULES, _safe_ratio, apply_representation
from visualize_representation import _by_participant


def _data(participants: range, recordings: int = 2) -> pd.DataFrame:
    rows = []
    for participant in participants:
        for recording in range(recordings):
            x = participant * 0.015 + recording * 0.003
            rows.append(
                {
                    "participant_id": f"p{participant:03d}",
                    "session_id": recording,
                    "GLC": 100 + 2 * participant + recording,
                    "duration": 10000,
                    "ppg_b2_N_BEATS": 100 + participant,
                    "ppg_b2_width_median": 0.3 + x,
                    "ppg_b2_T_median": 0.9 + x,
                    "ppg_b2_Tr_median": 0.2 + x,
                    "ppg_p1_peak_interval_std": 0.02 + x / 10,
                    "ppg_p1_hr_mean": 65 + participant,
                }
            )
    source = pd.DataFrame(rows)
    # Colunas adicionais apenas para o teste: manter o baseline com 72 PPGs.
    for index in range(66):
        source[f"ppg_mock_{index:02d}"] = [
            float(index + p * 0.1) for p in range(len(source))
        ]
    transformed, report = apply_representation(source, sampling_rate=500)
    assert len(report["new_columns"]) == len(RATIO_RULES)
    return transformed


class TestFeatureExtraction(unittest.TestCase):
    def test_four_formulas_preserved(self):
        data = _data(range(1, 2), recordings=1)
        row = data.iloc[0]
        self.assertAlmostEqual(row["repr_ppg_beat_rate_hz"], 101 * 500 / 10000)
        self.assertAlmostEqual(
            row["repr_ppg_b2_width_over_period"],
            row["ppg_b2_width_median"] / row["ppg_b2_T_median"],
        )
        self.assertAlmostEqual(
            row["repr_ppg_b2_crest_time_over_period"],
            row["ppg_b2_Tr_median"] / row["ppg_b2_T_median"],
        )
        self.assertAlmostEqual(
            row["repr_ppg_p1_rr_cv"],
            row["ppg_p1_peak_interval_std"] * row["ppg_p1_hr_mean"] / 60,
        )

    def test_no_global_median_in_ratio(self):
        num = pd.Series([1.0, 1.0], dtype=float)
        den = pd.Series([1e-5, 1000.0], dtype=float)
        original = _safe_ratio(num, den).iloc[0]
        # Adding unrelated participants must not change the first ratio.
        extended = _safe_ratio(pd.concat([num, pd.Series([1.0])], ignore_index=True),
                               pd.concat([den, pd.Series([1e8])], ignore_index=True))
        self.assertAlmostEqual(extended.iloc[0], original)
        self.assertTrue(np.isnan(_safe_ratio(pd.Series([1]), pd.Series([0])).iloc[0]))

    def test_grouped_splits_and_reserved_test(self):
        train, test = _data(range(1, 21)), _data(range(21, 26))
        _check_data(train, test)
        groups = train["participant_id"].to_numpy()
        for outer_train, outer_val in GroupKFold(5).split(train, groups=groups):
            self.assertFalse(set(groups[outer_train]) & set(groups[outer_val]))
            for inner_train, inner_val in GroupKFold(3).split(
                train.iloc[outer_train], groups=groups[outer_train]
            ):
                self.assertFalse(
                    set(groups[outer_train][inner_train]) & set(groups[outer_train][inner_val])
                )
        altered = test.copy()
        altered.iloc[0, altered.columns.get_loc("participant_id")] = train.iloc[0]["participant_id"]
        with self.assertRaises(ValueError):
            _check_data(train, altered)

    def test_documented_upstream_overlap_is_removed_from_training_only(self):
        train = _data(range(1, 20))
        test = _data(range(21, 25))
        train.loc[0:1, "participant_id"] = "ieb_01"
        test.loc[0:1, "participant_id"] = "ieb_01"
        test_before = test.copy(deep=True)
        cleaned, excluded, n_rows = resolve_official_overlap(train, test)
        self.assertEqual(excluded, ["ieb_01"])
        self.assertEqual(n_rows, 2)
        self.assertEqual(len(cleaned), len(train) - 2)
        self.assertNotIn("ieb_01", set(cleaned["participant_id"]))
        pd.testing.assert_frame_equal(test, test_before)
        _check_data(cleaned, test)

    def test_other_overlap_is_not_silently_removed(self):
        train = _data(range(1, 20))
        test = _data(range(21, 25))
        test.loc[0, "participant_id"] = train.loc[0, "participant_id"]
        with self.assertRaisesRegex(ValueError, "Sobreposição inesperada"):
            resolve_official_overlap(train, test)

    def test_one_point_per_participant(self):
        data = _data(range(1, 8), recordings=3)
        summary = _by_participant(data)
        self.assertEqual(len(summary), 7)
        self.assertEqual(summary["participant_id"].nunique(), 7)

    def test_cv_and_final_test_only_after_selection(self):
        train, test = _data(range(1, 21)), _data(range(21, 26))
        # Smaller grid only for the fast unit test; production uses all four models.
        fast_models = {"Árvore de decisão": (
            DecisionTreeRegressor(random_state=0), {"model__max_depth": [2]}
        )}
        with patch("evaluate_formulas._models", return_value=fast_models):
            result = model_table(train)
            self.assertIn("rmse_medio", result.columns)
            self.assertEqual(result["modelo"].tolist()[0], "Dummy")
            # Changing only the test target must not affect nested CV.
            tampered_test = test.copy()
            tampered_test["GLC"] += 300
            result_again = model_table(train)
            pd.testing.assert_frame_equal(result, result_again)
            final = independent_test(train, test, result)
            final_tampered = independent_test(train, tampered_test, result)
            self.assertEqual(final["rmse_nested_cv"].iloc[0],
                             final_tampered["rmse_nested_cv"].iloc[0])
            self.assertNotEqual(final["rmse_teste"].iloc[0],
                                final_tampered["rmse_teste"].iloc[0])


if __name__ == "__main__":
    unittest.main()
