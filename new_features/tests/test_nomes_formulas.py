"""Garante consistência dos nomes da avaliação das fórmulas entre código e dados."""
import json
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from visualize_cv_results import build_comparison


class FormulaNamingTests(unittest.TestCase):
    def test_module_and_output_exist_under_new_names(self):
        self.assertTrue((BASE / 'evaluate_formulas.py').is_file())
        self.assertFalse((BASE / 'evaluate_representation.py').exists())
        self.assertTrue((BASE / 'results' / 'avaliacao_formulas.csv').is_file())
        self.assertFalse((BASE / 'results' / 'avaliacao_modelos.csv').exists())

    def test_config_and_results_are_consistent(self):
        config = BASE / 'config' / 'comparacao_destaques.json'
        entries = json.loads(config.read_text(encoding='utf-8'))
        refs = [entry['arquivo'] for entry in entries]
        self.assertEqual(refs.count('avaliacao_formulas.csv'), 2)
        df = build_comparison(BASE / 'results', config)
        self.assertEqual(len(df), 4)
        self.assertEqual(df.iloc[2]['origem'], 'avaliacao_formulas.csv')
        self.assertEqual(df.iloc[3]['origem'], 'avaliacao_formulas.csv')


if __name__ == '__main__':
    unittest.main()
