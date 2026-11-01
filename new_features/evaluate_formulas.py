"""
Avaliar quatro fórmulas, 72 features PPG originais e sua combinação.

Gera as tabelas da seção 5 do README a partir dos arquivos de treino e teste
produzidos pelo MultiWaveLab2 e por ``run_feature_representation.py``:

redundancia.csv
    Para cada feature nova, a feature original mais parecida (Spearman),
    calculada apenas no conjunto de desenvolvimento.
avaliacao_formulas.csv
    MAE e RMSE para somente as fórmulas, somente as 72 originais e 72 + 4.
    Segue o protocolo de ``src/machinelearning/regression_evaluation.py``:
    validação cruzada aninhada por participante (5 partes externas para medir
    o erro, 3 internas para escolher os hiperparâmetros) e padronização
    dentro do pipeline. Todos os conjuntos usam os mesmos folds externos.
avaliacao_teste_exploratoria.csv
    MAE e RMSE no teste independente, somente com --evaluate-test. O teste
    já foi consultado anteriormente e não deve orientar novas escolhas.

Uso, a partir da pasta ``src`` do repositório::

    PYTHONPATH=. python ../new_features/evaluate_formulas.py \\
        --train ../../multiwavelab_outputs/output_all_ppgs/features/features0_train_repr.csv \\
        --test ../../multiwavelab_outputs/output_all_ppgs/features/features0_test_repr.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from feature_representation import RATIO_RULES, REPR_PREFIX, TARGET_COLUMN

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"

# Same outer/inner sizes as src/machinelearning/regression_evaluation.py.
# All folds are grouped: several recordings may belong to one participant.
N_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_STATE = 0
EXPECTED_REPR = tuple(REPR_PREFIX + rule.name for rule in RATIO_RULES)


def _check_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check the split and fail rather than evaluate mixed participants."""
    for name, df in (("treino", train), ("teste", test)):
        if df.empty:
            raise ValueError(f"O conjunto de {name} está vazio.")
        required = ["participant_id", TARGET_COLUMN, *EXPECTED_REPR]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no {name}: {missing}")
        if df["participant_id"].isna().any():
            raise ValueError(f"Há participant_id ausente no {name}.")
        target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"Há valores de GLC inválidos no {name}.")
    if train["participant_id"].nunique() < N_SPLITS:
        raise ValueError(f"Treino precisa de pelo menos {N_SPLITS} participantes.")
    repeated = set(train["participant_id"]) & set(test["participant_id"])
    if repeated:
        raise ValueError(
            f"Participantes presentes no treino e no teste: {sorted(repeated)}"
        )
    original_train = {c for c in train if c.startswith("ppg_")}
    original_test = {c for c in test if c.startswith("ppg_")}
    if not original_train or original_train != original_test:
        raise ValueError("As features PPG originais precisam coincidir em treino e teste.")


def redundancy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Feature original mais parecida com cada feature nova (|Spearman|)."""
    original = [c for c in df.columns if c.startswith("ppg_")]
    rows = []
    for name in EXPECTED_REPR:
        # Spearman ignores missing pairs; constant columns cannot be compared.
        rho = df[original].apply(lambda col: col.corr(df[name], method="spearman"))
        rho = rho.dropna()
        if rho.empty:
            best, value = "", np.nan
        else:
            best = rho.abs().idxmax()
            value = round(float(rho[best]), 3)
        rows.append(
            {
                "feature_nova": name,
                "original_mais_parecida": best,
                "spearman_rho": value,
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


def _pipeline(model) -> Pipeline:
    """Impute and standardize inside each training fold only."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", clone(model)),
        ]
    )


def _tuned_estimator(model, params: dict):
    """Build the inner, participant-grouped hyperparameter search."""
    pipeline = _pipeline(model)
    if not params:
        return pipeline
    return GridSearchCV(
        pipeline,
        params,
        cv=GroupKFold(n_splits=N_INNER_SPLITS),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # avoids exhausting memory on an 8 GB notebook
        refit=True,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def _fold_metrics(
    model, params: dict, X: np.ndarray, y: np.ndarray,
    groups: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """MAE/RMSE in outer folds; the inner folds tune hyperparameters."""
    maes, rmses = [], []
    for train, validation in splits:
        estimator = _tuned_estimator(model, params)
        if params:
            estimator.fit(X[train], y[train], groups=groups[train])
        else:
            estimator.fit(X[train], y[train])
        mae, rmse = _metrics(y[validation], estimator.predict(X[validation]))
        maes.append(mae)
        rmses.append(rmse)
    return np.asarray(maes), np.asarray(rmses)


def _feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    original = [c for c in df.columns if c.startswith("ppg_")]
    new = list(EXPECTED_REPR)
    return {
        f"{len(original)} originais": original,
        f"{len(original)} originais + {len(new)} novas": original + new,
        f"só as {len(new)} novas": new,
    }


def model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Nested CV no treino, pareada com o baseline (média)."""
    feature_sets = _feature_sets(df)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    groups = df["participant_id"].to_numpy()
    splits = list(GroupKFold(n_splits=N_SPLITS).split(df, y, groups))
    # The Dummy predictor is fit in each outer fold using training GLC only.
    baseline_mae, baseline_rmse = _fold_metrics(
        DummyRegressor(), {}, np.zeros((len(y), 1)), y, groups, splits
    )
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
    for set_name, columns in feature_sets.items():
        X = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X[~np.isfinite(X)] = np.nan
        for model_name, (model, params) in _models().items():
            print(f"  {set_name} / {model_name}")
            maes, rmses = _fold_metrics(model, params, X, y, groups, splits)
            rows.append(
                {
                    "features": set_name,
                    "modelo": model_name,
                    "mae_medio": maes.mean(),
                    "rmse_medio": rmses.mean(),
                    "desvio_entre_folds": maes.std(ddof=1),
                    "desvio_rmse_entre_folds": rmses.std(ddof=1),
                    "ganho_vs_media": (baseline_mae - maes).mean(),
                    "pct_folds_melhor_que_media": 100.0 * (maes < baseline_mae).mean(),
                }
            )
    # Keep full precision for choosing the winner; round only when saving.
    return pd.DataFrame(rows)


def independent_test(
    train: pd.DataFrame, test: pd.DataFrame, models: pd.DataFrame,
) -> pd.DataFrame:
    """Escolhe entre representações construídas por RMSE e avalia no teste."""
    # Somente representações construídas participam da seleção final.
    # As 72 características originais e o Dummy são baselines de comparação.
    candidates = models.loc[
        models["features"].isin(
            {"72 originais + 4 novas", "só as 4 novas"}
        )
    ]
    if candidates.empty:
        raise ValueError("Nenhuma representação de Feature Extraction foi avaliada.")
    selected = candidates.sort_values("rmse_medio", kind="stable").iloc[0]
    set_name, model_name = selected["features"], selected["modelo"]
    feature_sets = _feature_sets(train)
    if model_name == "Dummy":
        columns = []
        model, params = DummyRegressor(), {}
    else:
        columns = feature_sets[set_name]
        model, params = _models()[model_name]

    if columns:
        X_train = train[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        X_test = test[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        X_train = np.zeros((len(train), 1))
        X_test = np.zeros((len(test), 1))
    X_train[~np.isfinite(X_train)] = np.nan
    X_test[~np.isfinite(X_test)] = np.nan
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)

    estimator = _tuned_estimator(model, params)
    if params:
        estimator.fit(X_train, y_train, groups=train["participant_id"].to_numpy())
        best_params = estimator.best_params_
    else:
        estimator.fit(X_train, y_train)
        best_params = {}
    mae, rmse = _metrics(y_test, estimator.predict(X_test))
    return pd.DataFrame(
        [
            {
                "features": set_name,
                "modelo": model_name,
                "mae_nested_cv": round(float(selected["mae_medio"]), 1),
                "rmse_nested_cv": round(float(selected["rmse_medio"]), 1),
                "mae_teste": round(mae, 1),
                "rmse_teste": round(rmse, 1),
                "n_participantes_treino": train["participant_id"].nunique(),
                "n_participantes_teste": test["participant_id"].nunique(),
                "hiperparametros": str(best_params),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "json_file", metavar="JSON_FILE",
        help="JSON All-PPGs/PPG do experimento, usado para localizar treino e teste.",
    )
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Avalia explicitamente no teste reservado (não usar para explorar alternativas).",
    )
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("DATASET_NAME") != "all_ppgs" or config.get("MODALITIES") != "ppg":
        raise ValueError("Esta etapa usa apenas DATASET_NAME=all_ppgs e MODALITIES=ppg.")
    # The config uses paths relative to src/, as in the original pipeline.
    output_path = Path(config["SIMULATIONS_OUTPUT_PATH"].replace("\\", "/")) / "features"
    prefix = config["FEATURES_FILE_PREFIX"]
    train_file = output_path / f"{prefix}_train_repr.csv"
    test_file = output_path / f"{prefix}_test_repr.csv"
    for file in (train_file, test_file):
        if not file.is_file():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {file}\n"
                "Rode primeiro run_feature_representation.py com --source train_test."
            )

    splits_folder = Path(config["SIMULATIONS_INPUT_PATH"].replace("\\", "/")) / "train_test_splits"
    split_prefix = config["TRAIN_TEST_SPLIT_PREFIX"]
    official = {}
    for label in ("train", "test"):
        path = splits_folder / f"{split_prefix}_participants_{label}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Lista oficial de participantes não encontrada: {path}")
        official[label] = set(pd.read_csv(path)["participant_id"].astype(str))
    overlap_official = official["train"] & official["test"]
    if overlap_official - {"ieb_01"}:
        raise ValueError(
            f"Sobreposição inesperada nas listas oficiais: {sorted(overlap_official)}"
        )
    if overlap_official:
        print(
            "Aviso: as listas oficiais All-PPGs incluem ieb_01 nos dois lados. "
            "Nos CSVs de representação do Grupo 3, esse participante fica "
            "somente no teste independente."
        )

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    _check_data(train, test)
    for label, data in (("train", train), ("test", test)):
        unexpected = set(data["participant_id"].astype(str)) - official[label]
        if unexpected:
            raise ValueError(
                f"Há participantes fora da lista oficial de {label}: {sorted(unexpected)}"
            )
    print(
        f"treino: {len(train)} linhas / {train['participant_id'].nunique()} participantes; "
        f"teste reservado: {len(test)} linhas / {test['participant_id'].nunique()} participantes"
    )

    redundancy = redundancy_table(train)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    redundancy.to_csv(RESULTS_FOLDER / "redundancia.csv", index=False)
    print("\nfeature original mais parecida com cada feature nova (somente treino):")
    print(redundancy.to_string(index=False))

    models = model_table(train)
    models.to_csv(RESULTS_FOLDER / "avaliacao_formulas.csv", index=False, float_format="%.1f")
    print(f"\nNested CV por participante: {N_SPLITS} partes externas x {N_INNER_SPLITS} internas:")
    print(models.round(1).to_string(index=False))

    if args.evaluate_test:
        print(
            "\nAviso: o teste independente já foi consultado na exploração anterior. "
            "Esta execução deve ser documentada como análise adicional, "
            "não como teste inédito."
        )
        final = independent_test(train, test, models)
        final.to_csv(RESULTS_FOLDER / "avaliacao_teste_exploratoria.csv", index=False)
        print("\nAvaliação exploratória no teste reservado:")
        print(final.to_string(index=False))
    else:
        print("\nApenas nested CV: o teste independente não foi reavaliado.")
    print(f"\ntabelas salvas em: {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
