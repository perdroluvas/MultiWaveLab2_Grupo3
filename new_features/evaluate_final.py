"""
Avaliação final: nested CV no treino para escolher e teste reservado para confirmar.

Segue as duas premissas do professor:

* somente o dataset All-PPGs e somente a modalidade PPG;
* nested CV **e** um conjunto de teste.

Protocolo (fixado antes de olhar os resultados)
----------------------------------------------
Representações comparadas:

    72 originais   as features do MultiWaveLab2, sem transformação (referência)
    4 fórmulas     as colunas repr_ criadas por feature_representation.py
    PCA            componentes das 72 originais (não usa a glicose)
    PLS            componentes das 72 originais orientados pela glicose

Modelos: regressão linear, KNN, árvore de decisão e SVM, com as grades de
``evaluate_formulas._models``. O Dummy (prevê a média do treino) é o piso.

1. **Nested CV, só no treino.** Divisão externa em 5 partes por
   participante, repetida ``N_REPEATS`` vezes com sorteios diferentes; divisão
   interna em 3 partes para escolher os hiperparâmetros. No PCA e no PLS, o
   número de componentes também é escolhido na divisão interna. Imputação,
   padronização e PCA/PLS são ajustados só no treino de cada fold.
2. **Escolha.** Em cada representação, o modelo com menor RMSE médio na
   divisão externa (mesma métrica de ``src/machinelearning/regression_evaluation.py``).
3. **Teste, uma vez só.** O modelo escolhido de cada representação é
   reajustado em todo o treino (hiperparâmetros pela divisão interna) e prevê
   o conjunto de teste oficial. Nenhuma escolha usa o teste.

Saídas em ``new_features/results/``:

avaliacao_final_nested_cv.csv   todas as combinações representação × modelo
avaliacao_final.csv             o modelo escolhido de cada representação,
                                com o erro na nested CV e no teste
03_avaliacao_final.png          MAE de cada representação ao lado do Dummy

Uso, a partir de ``src/``::

    PYTHONPATH=. python ../new_features/evaluate_final.py \\
        ../config_experiments/input_all_ppgs/exp0.json
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import ensure_import_paths

ensure_import_paths()

from evaluate_formulas import (
    EXPECTED_REPR,
    N_INNER_SPLITS,
    N_SPLITS,
    RESULTS_FOLDER,
    _metrics,
    _models,
    load_train_test,
)
from evaluate_pls import PLSFeatureExtractor
from feature_representation import TARGET_COLUMN

N_REPEATS = 5
COMPONENTS = (2, 4, 8, 12)
DUMMY = "nenhuma (média)"
FAMILIES = ("72 originais", "4 fórmulas", "PCA", "PLS")


def _columns(df: pd.DataFrame, family: str) -> list[str]:
    """Colunas de entrada de cada representação."""
    if family == "4 fórmulas":
        return list(EXPECTED_REPR)
    original = [c for c in df.columns if c.startswith("ppg_")]
    if len(original) != 72:
        raise ValueError(f"Esperadas 72 features PPG originais; encontradas {len(original)}.")
    return original


def _extractor(family: str):
    """Transformação ajustada dentro do fold (None para as colunas prontas)."""
    if family == "PCA":
        return PCA(svd_solver="full")
    if family == "PLS":
        return PLSFeatureExtractor()
    return None


def build_estimator(family: str, model, params: dict):
    """Pipeline da representação + busca interna dos hiperparâmetros."""
    steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    extractor = _extractor(family)
    grid = dict(params)
    if extractor is not None:
        steps.append(("extract", extractor))
        grid["extract__n_components"] = list(COMPONENTS)
    steps.append(("model", clone(model)))
    pipeline = Pipeline(steps)
    if not grid:
        return pipeline
    return GridSearchCV(
        pipeline,
        grid,
        cv=GroupKFold(n_splits=N_INNER_SPLITS),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )


def _fit(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    if isinstance(estimator, GridSearchCV):
        return estimator.fit(X, y, groups=groups)
    return estimator.fit(X, y)


def _matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X


def outer_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Divisões externas por participante, repetidas com sorteios diferentes."""
    splits = []
    for repeat in range(N_REPEATS):
        cv = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=repeat)
        splits.extend(cv.split(groups, groups=groups))
    return splits


def nested_cv_table(train: pd.DataFrame) -> pd.DataFrame:
    """Nested CV de cada representação × modelo, nos mesmos folds externos."""
    y = train[TARGET_COLUMN].to_numpy(dtype=float)
    groups = train["participant_id"].astype(str).to_numpy()
    splits = outer_splits(groups)

    def scores(family, model, params, X):
        maes, rmses = [], []
        for fit_idx, val_idx in splits:
            estimator = _fit(
                build_estimator(family, model, params),
                X[fit_idx], y[fit_idx], groups[fit_idx],
            )
            mae, rmse = _metrics(y[val_idx], estimator.predict(X[val_idx]))
            maes.append(mae)
            rmses.append(rmse)
        return np.asarray(maes), np.asarray(rmses)

    base_mae, base_rmse = scores("72 originais", DummyRegressor(), {}, np.zeros((len(y), 1)))
    rows = [_row(DUMMY, "Dummy", base_mae, base_rmse, base_mae, base_rmse)]
    for family in FAMILIES:
        X = _matrix(train, _columns(train, family))
        for name, (model, params) in _models().items():
            print(f"  nested CV: {family} / {name}", flush=True)
            mae, rmse = scores(family, model, params, X)
            rows.append(_row(family, name, mae, rmse, base_mae, base_rmse))
    return pd.DataFrame(rows)


def _row(family, model, mae, rmse, base_mae, base_rmse) -> dict:
    # Gains are paired fold by fold: positive = better than predicting the mean.
    gain_mae, gain_rmse = base_mae - mae, base_rmse - rmse
    return {
        "representacao": family,
        "modelo": model,
        "mae_nested_cv": mae.mean(),
        "desvio_mae": mae.std(ddof=1),
        "rmse_nested_cv": rmse.mean(),
        "desvio_rmse": rmse.std(ddof=1),
        "ganho_mae_nested_cv": gain_mae.mean(),
        "desvio_ganho_mae": gain_mae.std(ddof=1),
        "ganho_rmse_nested_cv": gain_rmse.mean(),
        "desvio_ganho_rmse": gain_rmse.std(ddof=1),
        "pct_folds_rmse_melhor_que_media": (
            np.nan if model == "Dummy" else 100.0 * (rmse < base_rmse).mean()
        ),
        "n_folds_externos": len(mae),
    }


def choose_models(cv_table: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por representação: o modelo de menor RMSE na nested CV."""
    chosen = (
        cv_table.sort_values("rmse_nested_cv", kind="stable")
        .groupby("representacao", sort=False)
        .head(1)
    )
    order = {name: i for i, name in enumerate((DUMMY, *FAMILIES))}
    return chosen.sort_values("representacao", key=lambda s: s.map(order)).reset_index(drop=True)


def test_table(train: pd.DataFrame, test: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """Reajusta cada escolha em todo o treino e avalia uma vez no teste."""
    models = {**_models(), "Dummy": (DummyRegressor(), {})}
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)
    groups = train["participant_id"].astype(str).to_numpy()
    rows = []
    for _, row in chosen.iterrows():
        family, name = row["representacao"], row["modelo"]
        model, params = models[name]
        if name == "Dummy":
            X_train, X_test = np.zeros((len(train), 1)), np.zeros((len(test), 1))
            family_for_pipeline = "72 originais"
        else:
            columns = _columns(train, family)
            X_train, X_test = _matrix(train, columns), _matrix(test, columns)
            family_for_pipeline = family
        estimator = _fit(
            build_estimator(family_for_pipeline, model, params), X_train, y_train, groups
        )
        mae, rmse = _metrics(y_test, estimator.predict(X_test))
        best = estimator.best_params_ if isinstance(estimator, GridSearchCV) else {}
        rows.append(
            {
                **row.to_dict(),
                "mae_teste": mae,
                "rmse_teste": rmse,
                "hiperparametros": str(best),
                "glc_media_treino": float(y_train.mean()),
                "n_participantes_treino": train["participant_id"].nunique(),
                "n_participantes_teste": test["participant_id"].nunique(),
            }
        )
    table = pd.DataFrame(rows)
    dummy = table.loc[table["representacao"] == DUMMY].iloc[0]
    for metric in ("mae", "rmse"):
        table[f"ganho_{metric}_teste"] = dummy[f"{metric}_teste"] - table[f"{metric}_teste"]
    return table


def plot_final(final: pd.DataFrame, path: Path) -> None:
    """MAE de cada representação ao lado do Dummy: nested CV e teste.

    O Dummy (chutar a média) é uma barra como as outras, em cinza, para que a
    comparação com ele se veja direto: a menor barra é a que erra menos.
    """
    blue, grey = "#2a78d6", "#898781"
    ink, muted, grid = "#0b0b0b", "#52514e", "#e1e0d9"
    order = (DUMMY, *FAMILIES)
    rows = final.set_index("representacao").loc[list(order)].reset_index()
    mean_glc = float(rows["glc_media_treino"].iloc[0])
    names = [f"sempre {mean_glc:.0f} mg/dL\n(média do treino)" if r == DUMMY else f"{r}\n({m})"
             for r, m in zip(rows["representacao"], rows["modelo"])]
    colors = [grey if r == DUMMY else blue for r in rows["representacao"]]
    x = np.arange(len(rows))
    n_train = int(rows["n_participantes_treino"].iloc[0])
    n_test = int(rows["n_participantes_teste"].iloc[0])
    panels = (
        ("mae_nested_cv", f"Nested CV no treino ({n_train} pessoas)"),
        ("mae_teste", f"Teste ({n_test} pessoas, uma avaliação)"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, (column, title) in zip(axes, panels):
        values = rows[column].to_numpy()
        axis.bar(x, values, 0.62, color=colors)
        axis.axhline(values[0], color=grey, lw=1, ls=(0, (4, 3)))
        best = int(np.argmin(values))
        for xi, value in zip(x, values):
            axis.text(xi, value + 0.8, f"{value:.1f}".replace(".", ","), ha="center",
                      va="bottom", fontsize=9, color=ink,
                      fontweight="bold" if xi == best else "normal")
        axis.set_xticks(x)
        axis.set_xticklabels(names, fontsize=9, color=ink)
        axis.set_title(title, fontsize=11, color=ink)
        axis.grid(axis="y", color=grid, lw=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=muted, labelsize=8)
    spread = rows.loc[rows["representacao"] != DUMMY, "desvio_ganho_mae"]
    axes[0].text(
        0.5, 0.97,
        f"a diferença para o cinza varia ±{spread.min():.0f} a ±{spread.max():.0f} mg/dL entre folds".replace(".", ","),
        transform=axes[0].transAxes, ha="center", va="top", fontsize=8, color=muted,
    )
    axes[0].set_ylabel("erro médio da glicose, MAE (mg/dL)", fontsize=9, color=muted)
    axes[0].set_ylim(0, rows[["mae_nested_cv", "mae_teste"]].to_numpy().max() * 1.22)
    figure.suptitle(
        f"Comparado a prever sempre {mean_glc:.0f} mg/dL: na nested CV as diferenças são pequenas; "
        "no teste, ninguém erra menos — All-PPGs, PPG",
        fontsize=12, color=ink,
    )
    figure.text(
        0.5, 0.005,
        f"Cinza = Dummy: ignora o PPG e prevê para todos a glicose média do treino ({mean_glc:.0f} mg/dL). "
        "Menor barra = menor erro (em negrito). Modelo de cada representação escolhido por nested CV.",
        ha="center", fontsize=8, color=muted,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("json_file", help="JSON All-PPGs/PPG do experimento.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    train, test = load_train_test(args.json_file)

    cv_table = nested_cv_table(train)
    chosen = choose_models(cv_table)
    final = test_table(train, test, chosen)

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(RESULTS_FOLDER / "avaliacao_final_nested_cv.csv", index=False, float_format="%.3f")
    final.to_csv(RESULTS_FOLDER / "avaliacao_final.csv", index=False, float_format="%.3f")
    plot_final(final, RESULTS_FOLDER / "03_avaliacao_final.png")

    columns = ["representacao", "modelo", "mae_nested_cv", "desvio_mae",
               "rmse_nested_cv", "mae_teste", "rmse_teste", "hiperparametros"]
    print(f"\nNested CV: {N_SPLITS} partes externas x {N_REPEATS} repetições, "
          f"{N_INNER_SPLITS} internas. Teste avaliado uma vez por representação.")
    print(final[columns].round(1).to_string(index=False))
    print(f"\nresultados em: {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
