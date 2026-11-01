"""
Comparar representações PLS de 2, 4, 8 e 12 componentes com nested CV agrupada.

A nova representação é construída usando as 72 features PPG e a glicose
somente dos folds de treinamento. O teste independente não é utilizado.

Uso, a partir de src/:

    PYTHONPATH=. python ../new_features/evaluate_pls.py \
        ../config_experiments/input_all_ppgs/exp0.json

Os resultados são gravados em new_features/results/avaliacao_pls_cv.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_formulas import (
    N_INNER_SPLITS,
    N_SPLITS,
    RESULTS_FOLDER,
    _metrics,
    _models,
)
from evaluate_pca import _prepare_train

DEFAULT_COMPONENTS = (2, 4, 8, 12)


class PLSFeatureExtractor(TransformerMixin, BaseEstimator):
    """Extrai componentes PLS supervisionados, sem transformar o alvo em saída.

    PLSRegression.fit_transform(X, y) pode devolver (X_scores, y_scores),
    incompatível com um transformador intermediário do Pipeline. Aqui o fit
    usa y apenas para aprender os componentes e transform devolve somente X.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        if y is None:
            raise ValueError("PLS precisa dos valores de glicose no treino.")
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        scores = self.pls_.transform(X)
        # Este adaptador só entrega componentes de X ao regressor seguinte.
        if isinstance(scores, tuple):
            scores = scores[0]
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != self.n_components:
            raise ValueError("A transformação PLS não produziu a forma esperada.")
        return scores


def _pipeline(model, n_components: int) -> Pipeline:
    """Imputação, escala e PLS ajustados dentro de cada fold de treinamento."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("pls", PLSFeatureExtractor(n_components=n_components)),
            ("model", clone(model)),
        ]
    )


def _nested_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_components: int,
    model,
    params: dict,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    maes, rmses = [], []
    for train_idx, val_idx in splits:
        pipeline = _pipeline(model, n_components)
        if params:
            estimator = GridSearchCV(
                pipeline,
                params,
                cv=GroupKFold(n_splits=N_INNER_SPLITS),
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
                refit=True,
            )
            estimator.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        else:
            estimator = pipeline.fit(X[train_idx], y[train_idx])
        mae, rmse = _metrics(y[val_idx], estimator.predict(X[val_idx]))
        maes.append(mae)
        rmses.append(rmse)
    return np.asarray(maes), np.asarray(rmses)


def pls_model_table(
    df: pd.DataFrame,
    components: tuple[int, ...] = DEFAULT_COMPONENTS,
    model_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compara quantidades de componentes nos mesmos folds agrupados."""
    X, y, groups = _prepare_train(df)
    if any(n < 1 or n > X.shape[1] for n in components):
        raise ValueError("Cada quantidade de componentes deve estar entre 1 e 72.")
    if len(set(components)) != len(components):
        raise ValueError("Não repita quantidades de componentes.")

    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    baseline_mae, baseline_rmse = [], []
    for train_idx, val_idx in splits:
        estimator = DummyRegressor(strategy="mean").fit(
            np.zeros((len(train_idx), 1)), y[train_idx]
        )
        mae, rmse = _metrics(
            y[val_idx], estimator.predict(np.zeros((len(val_idx), 1)))
        )
        baseline_mae.append(mae)
        baseline_rmse.append(rmse)
    baseline_mae = np.asarray(baseline_mae)
    baseline_rmse = np.asarray(baseline_rmse)
    rows = [
        {
            "features": "nenhuma (média)",
            "modelo": "Dummy",
            "mae_medio": baseline_mae.mean(),
            "rmse_medio": baseline_rmse.mean(),
            "desvio_entre_folds": baseline_mae.std(ddof=1),
            "desvio_rmse_entre_folds": baseline_rmse.std(ddof=1),
            "ganho_vs_media": 0.0,
            "pct_folds_melhor_que_media": np.nan,
        }
    ]
    available_models = _models()
    if model_names is not None:
        unknown = set(model_names) - set(available_models)
        if unknown:
            raise ValueError(f"Modelos desconhecidos: {sorted(unknown)}")
        available_models = {name: available_models[name] for name in model_names}

    for n in components:
        for name, (model, params) in available_models.items():
            print(f"  PLS {n} componentes / {name}", flush=True)
            mae, rmse = _nested_scores(X, y, groups, n, model, params, splits)
            rows.append(
                {
                    "features": f"PLS {n} componentes",
                    "modelo": name,
                    "mae_medio": mae.mean(),
                    "rmse_medio": rmse.mean(),
                    "desvio_entre_folds": mae.std(ddof=1),
                    "desvio_rmse_entre_folds": rmse.std(ddof=1),
                    "ganho_vs_media": (baseline_mae - mae).mean(),
                    "pct_folds_melhor_que_media": 100.0 * (mae < baseline_mae).mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("json_file", help="JSON All-PPGs/PPG do experimento.")
    parser.add_argument(
        "--components", nargs="+", type=int, default=list(DEFAULT_COMPONENTS),
        help="Quantidades de componentes PLS testadas separadamente (padrão: 2 4 8 12).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Teste inicial apenas com KNN e SVM; não substitui a avaliação completa.",
    )
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("DATASET_NAME") != "all_ppgs" or config.get("MODALITIES") != "ppg":
        raise ValueError("Esta etapa usa apenas DATASET_NAME=all_ppgs e MODALITIES=ppg.")

    output_path = Path(config["SIMULATIONS_OUTPUT_PATH"].replace("\\", "/")) / "features"
    train_file = output_path / f"{config['FEATURES_FILE_PREFIX']}_train_repr.csv"
    if not train_file.is_file():
        raise FileNotFoundError(
            f"Treino não encontrado: {train_file}\n"
            "Execute primeiro run_feature_representation.py --source train_test."
        )
    train = pd.read_csv(train_file)
    print(
        f"treino: {len(train)} linhas / {train['participant_id'].nunique()} participantes"
    )
    print("PLS ajustado somente dentro dos folds; teste independente não utilizado.")
    table = pls_model_table(
        train,
        components=tuple(args.components),
        model_names=("KNN", "SVM") if args.quick else None,
    )
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    suffix = "_rapida" if args.quick else ""
    destination = RESULTS_FOLDER / f"avaliacao_pls_cv{suffix}.csv"
    table.to_csv(destination, index=False, float_format="%.3f")
    print(f"\nNested CV agrupada: {N_SPLITS} partes externas x {N_INNER_SPLITS} internas")
    print(table.round(2).to_string(index=False))
    print(f"\ntabela salva em: {destination}")
    print("Comparar MAE e RMSE com a representação manual e PCA; não consultar o teste.")


if __name__ == "__main__":
    main()
