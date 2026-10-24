"""
Avaliar as quatro features novas: redundância e desempenho em regressão.

Gera as duas tabelas da seção 3.4 do README a partir de
``results/features0_repr.csv`` (ou do arquivo indicado com ``--input``):

redundancia.csv
    Para cada feature nova, a feature original mais parecida (Spearman).
avaliacao_modelos.csv
    MAE da glicose para cada conjunto de features e cada modelo. Segue o
    protocolo de ``src/machinelearning/regression_evaluation.py``: validação
    cruzada aninhada (5 partes externas para medir o erro, 3 internas para
    escolher os hiperparâmetros), padronização dentro do pipeline e a média
    como referência. A divisão externa é repetida com sorteios diferentes, e
    todos os conjuntos usam os mesmos sorteios, então a comparação com a
    média é pareada (fold a fold).

Uso, a partir da raiz do repositório::

    python new_features/evaluate_representation.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from feature_representation import RATIO_RULES, REPR_PREFIX, TARGET_COLUMN

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"

# Same outer/inner sizes as src/machinelearning/regression_evaluation.py.
# There is one row per participant, so a plain KFold is already grouped.
N_SPLITS = 5
N_INNER_SPLITS = 3
N_REPEATS = 20
RANDOM_STATE = 0


def redundancy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Feature original mais parecida com cada feature nova (|Spearman|)."""
    original = [c for c in df.columns if c.startswith("ppg_")]
    rows = []
    for name in [c for c in df.columns if c.startswith(REPR_PREFIX)]:
        rho = df[original].apply(lambda col: spearmanr(col, df[name])[0])
        best = rho.abs().idxmax()
        rows.append(
            {
                "feature_nova": name,
                "original_mais_parecida": best,
                "spearman_rho": round(float(rho[best]), 3),
            }
        )
    return pd.DataFrame(rows)


def _models() -> dict[str, tuple[object, dict]]:
    """Modelos vistos na disciplina, cada um com a grade de hiperparâmetros."""
    return {
        "Regressão linear": (LinearRegression(), {}),
        "KNN": (
            KNeighborsRegressor(),
            {"model__n_neighbors": [3, 5, 7, 9, 15], "model__p": [1, 2]},
        ),
        # max_depth = 1 is a decision stump.
        "Árvore de decisão": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"model__max_depth": [1, 2, 3, 4], "model__min_samples_leaf": [3, 5, 10]},
        ),
        "SVM": (
            SVR(),
            {
                "model__kernel": ["linear", "rbf"],
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__epsilon": [1.0, 10.0],
            },
        ),
    }


def _fold_maes(model, params: dict, X: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    """MAE em cada divisão externa; os hiperparâmetros saem da divisão interna."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    inner_cv = KFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    maes = []
    for train, test in splits:
        if params:
            estimator = GridSearchCV(
                pipeline, params, cv=inner_cv,
                scoring="neg_root_mean_squared_error", n_jobs=-1,
            )
        else:
            estimator = pipeline
        estimator.fit(X[train], y[train])
        maes.append(mean_absolute_error(y[test], estimator.predict(X[test])))
    return np.array(maes)


def model_table(df: pd.DataFrame) -> pd.DataFrame:
    """MAE por conjunto de features, pareado com o baseline (média)."""
    original = [c for c in df.columns if c.startswith("ppg_")]
    new = [c for c in df.columns if c.startswith(REPR_PREFIX)]
    # The original columns that enter the four ratios: tells apart the effect
    # of the ratios from the effect of simply using few columns.
    ingredients = sorted(
        {
            col
            for rule in RATIO_RULES
            for col in rule.numerator + rule.denominator
            if col in original
        }
    )
    feature_sets = {
        "72 originais": original,
        "72 originais + 4 novas": original + new,
        f"só os {len(ingredients)} ingredientes originais": ingredients,
        "só as 4 novas": new,
    }
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    splits = list(
        RepeatedKFold(
            n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
        ).split(df)
    )
    baseline = _fold_maes(DummyRegressor(), {}, np.zeros((len(y), 1)), y, splits)

    rows = [
        {
            "features": "nenhuma (média)",
            "modelo": "Dummy",
            "mae_medio": baseline.mean(),
            "desvio_entre_folds": baseline.std(ddof=1),
            "ganho_vs_media": 0.0,
            "pct_folds_melhor_que_media": np.nan,
        }
    ]
    for set_name, columns in feature_sets.items():
        X = df[columns].to_numpy(dtype=float)
        for model_name, (model, params) in _models().items():
            print(f"  {set_name} / {model_name}")
            maes = _fold_maes(model, params, X, y, splits)
            rows.append(
                {
                    "features": set_name,
                    "modelo": model_name,
                    "mae_medio": maes.mean(),
                    "desvio_entre_folds": maes.std(ddof=1),
                    "ganho_vs_media": (baseline - maes).mean(),
                    "pct_folds_melhor_que_media": 100.0 * (maes < baseline).mean(),
                }
            )
    return pd.DataFrame(rows).round(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--input",
        default=str(RESULTS_FOLDER / "features0_repr.csv"),
        help="CSV com as features originais e as colunas repr_.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    df = pd.read_csv(args.input)
    print(f"lendo: {args.input} ({len(df)} linhas)")

    redundancy = redundancy_table(df)
    redundancy.to_csv(RESULTS_FOLDER / "redundancia.csv", index=False)
    print("\nfeature original mais parecida com cada feature nova:")
    print(redundancy.to_string(index=False))

    models = model_table(df)
    models.to_csv(RESULTS_FOLDER / "avaliacao_modelos.csv", index=False)
    print(
        f"\nMAE da glicose (mg/dL), {N_SPLITS} partes x {N_REPEATS} repetições:"
    )
    print(models.to_string(index=False))
    print(f"\ntabelas salvas em: {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
