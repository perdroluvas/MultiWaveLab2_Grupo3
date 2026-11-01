"""
Comparar representações PCA de 4, 8 e 12 componentes com nested CV agrupada.

O PCA substitui as 72 features originais por componentes calculados em cada
fold de treinamento. Não utiliza o teste independente nesta etapa.

Uso, a partir de src/:

    PYTHONPATH=. python ../new_features/evaluate_pca.py \\
        ../config_experiments/input_all_ppgs/exp0.json

A comparação com as quatro features manuais já está em
new_features/results/avaliacao_formulas.csv, gerado por evaluate_formulas.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA
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
from feature_representation import TARGET_COLUMN

DEFAULT_COMPONENTS = (4, 8, 12)


def _pipeline(model, n_components: int) -> Pipeline:
    """Imputação, escala e PCA ajustados exclusivamente dentro de cada fold."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="full")),
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
            estimator.fit(
                X[train_idx], y[train_idx], groups=groups[train_idx]
            )
        else:
            estimator = pipeline.fit(X[train_idx], y[train_idx])
        mae, rmse = _metrics(y[val_idx], estimator.predict(X[val_idx]))
        maes.append(mae)
        rmses.append(rmse)
    return np.asarray(maes), np.asarray(rmses)


def _prepare_train(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Usa apenas as features PPG e a glicose no conjunto de desenvolvimento."""
    required = {"participant_id", TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    if df.empty or df["participant_id"].isna().any():
        raise ValueError("O treino está vazio ou tem identificadores ausentes.")
    features = [name for name in df if name.startswith("ppg_")]
    if len(features) != 72:
        raise ValueError(
            f"Esperadas 72 features PPG originais do exp0; encontradas {len(features)}."
        )
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("GLC possui valores inválidos no treino.")
    X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    groups = df["participant_id"].astype(str).to_numpy()
    if len(np.unique(groups)) < N_SPLITS:
        raise ValueError("Não há participantes suficientes para a nested CV.")
    return X, y, groups


def pca_model_table(
    df: pd.DataFrame,
    components: tuple[int, ...] = DEFAULT_COMPONENTS,
    model_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Avalia cada dimensão separadamente, com os mesmos folds externos."""
    X, y, groups = _prepare_train(df)
    if any(n < 1 or n > X.shape[1] for n in components):
        raise ValueError("Cada quantidade de componentes deve estar entre 1 e 72.")
    if len(set(components)) != len(components):
        raise ValueError("Não repita quantidades de componentes.")

    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    # Não comparar com a média calculada no conjunto inteiro: refazer em cada fold.
    dummy_mae, dummy_rmse = [], []
    for train_idx, val_idx in splits:
        estimator = DummyRegressor(strategy="mean").fit(
            np.zeros((len(train_idx), 1)), y[train_idx]
        )
        mae, rmse = _metrics(
            y[val_idx], estimator.predict(np.zeros((len(val_idx), 1)))
        )
        dummy_mae.append(mae)
        dummy_rmse.append(rmse)
    dummy_mae = np.asarray(dummy_mae)
    dummy_rmse = np.asarray(dummy_rmse)
    rows = [
        {
            "features": "nenhuma (média)",
            "modelo": "Dummy",
            "mae_medio": dummy_mae.mean(),
            "rmse_medio": dummy_rmse.mean(),
            "desvio_entre_folds": dummy_mae.std(ddof=1),
            "desvio_rmse_entre_folds": dummy_rmse.std(ddof=1),
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
            print(f"  PCA {n} componentes / {name}", flush=True)
            mae, rmse = _nested_scores(X, y, groups, n, model, params, splits)
            rows.append(
                {
                    "features": f"PCA {n} componentes",
                    "modelo": name,
                    "mae_medio": mae.mean(),
                    "rmse_medio": rmse.mean(),
                    "desvio_entre_folds": mae.std(ddof=1),
                    "desvio_rmse_entre_folds": rmse.std(ddof=1),
                    "ganho_vs_media": (dummy_mae - mae).mean(),
                    "pct_folds_melhor_que_media": 100.0 * (mae < dummy_mae).mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("json_file", help="JSON All-PPGs/PPG do experimento.")
    parser.add_argument(
        "--components", nargs="+", type=int, default=list(DEFAULT_COMPONENTS),
        help="Quantidades de componentes PCA avaliadas separadamente (padrão: 4 8 12).",
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
    print("PCA ajustado somente dentro dos folds; teste independente não utilizado.")
    table = pca_model_table(
        train,
        components=tuple(args.components),
        model_names=("KNN", "SVM") if args.quick else None,
    )
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    suffix = "_rapida" if args.quick else ""
    destination = RESULTS_FOLDER / f"avaliacao_pca_cv{suffix}.csv"
    table.to_csv(destination, index=False, float_format="%.3f")
    print(f"\nNested CV agrupada: {N_SPLITS} partes externas x {N_INNER_SPLITS} internas")
    print(table.round(2).to_string(index=False))
    print(f"\ntabela salva em: {destination}")
    print("Comparar com results/avaliacao_formulas.csv; não usar o teste para escolher PCA.")


if __name__ == "__main__":
    main()
