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

Regressor: um só, o SVM (``MODEL_NAME``), igual para todas as
representações. O trabalho compara representações, não regressores: escolher
o melhor regressor é o tema de outro grupo. O Dummy (prevê a glicose média do
treino) é a referência.

1. **Nested CV, só no treino.** Divisão externa em 5 partes por
   participante, repetida ``N_REPEATS`` vezes com sorteios diferentes; divisão
   interna em 3 partes para escolher os ajustes do SVM e, no PCA e no PLS, o número
   de componentes. Imputação, padronização e PCA/PLS são ajustados só no
   treino de cada fold. A escolha interna usa RMSE, como em
   ``src/machinelearning/regression_evaluation.py``.
2. **Teste, uma vez só.** Cada representação é reajustada em todo o treino
   (ajustes e componentes pela divisão interna) e prevê o conjunto de teste
   oficial. Nenhuma escolha usa o teste.

Saídas em ``new_features/results/``:

avaliacao_final_nested_cv.csv   nested CV de cada representação
avaliacao_final.csv             nested CV e teste de cada representação
03_avaliacao_final.png          RMSE de cada representação ao lado da baseline (72 originais)
04_colunas_vs_erro.png          colunas de cada representação × RMSE

Uso, a partir de ``src/``::

    PYTHONPATH=. python ../new_features/evaluate_final.py \\
        ../config_experiments/input_all_ppgs/exp0.json
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker

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
MODEL_NAME = "SVM"
COMPONENTS = (2, 4, 8, 12, 16, 24)
DUMMY = "nenhuma (média)"
FAMILIES = ("72 originais", "4 fórmulas", "PCA", "PLS")
BASELINE = "72 originais"


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
    folds = {DUMMY: (base_mae, base_rmse)}
    model, params = _models()[MODEL_NAME]
    for family in FAMILIES:
        print(f"  nested CV: {family} / {MODEL_NAME}", flush=True)
        X = _matrix(train, _columns(train, family))
        mae, rmse = scores(family, model, params, X)
        folds[family] = (mae, rmse)
        rows.append(_row(family, MODEL_NAME, mae, rmse, base_mae, base_rmse))
    # The baseline of the theme is the 72 original features: each
    # representation minus the baseline, fold by fold (negative = lower error).
    ref_mae, ref_rmse = folds[BASELINE]
    for row in rows:
        mae, rmse = folds[row["representacao"]]
        row["dif_mae_vs_baseline"] = (mae - ref_mae).mean()
        row["desvio_dif_mae_vs_baseline"] = (mae - ref_mae).std(ddof=1)
        row["dif_rmse_vs_baseline"] = (rmse - ref_rmse).mean()
        row["desvio_dif_rmse_vs_baseline"] = (rmse - ref_rmse).std(ddof=1)
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


def test_table(train: pd.DataFrame, test: pd.DataFrame, cv_table: pd.DataFrame) -> pd.DataFrame:
    """Reajusta cada representação em todo o treino e avalia uma vez no teste."""
    models = {**_models(), "Dummy": (DummyRegressor(), {})}
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)
    groups = train["participant_id"].astype(str).to_numpy()
    rows = []
    for _, row in cv_table.iterrows():
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
    baseline = table.loc[table["representacao"] == BASELINE].iloc[0]
    for metric in ("mae", "rmse"):
        table[f"ganho_{metric}_teste"] = dummy[f"{metric}_teste"] - table[f"{metric}_teste"]
        table[f"dif_{metric}_teste_vs_baseline"] = table[f"{metric}_teste"] - baseline[f"{metric}_teste"]
    return table


def plot_final(final: pd.DataFrame, path: Path) -> None:
    """RMSE de cada representação nova ao lado da baseline (as 72 originais).

    A baseline do tema são as 72 features originais (cinza); as representações
    novas ficam em azul. O Dummy (prever sempre a glicose média do treino) é só
    uma linha de contexto: o erro de quem não usa o PPG.
    """
    blue, grey, line = "#2a78d6", "#898781", "#52514e"
    ink, muted, grid = "#0b0b0b", "#52514e", "#e1e0d9"
    order = FAMILIES
    rows = final.set_index("representacao").loc[list(order)].reset_index()
    dummy = final.loc[final["representacao"] == DUMMY].iloc[0]
    mean_glc = float(dummy["glc_media_treino"])
    names = [f"{r}\n(baseline)" if r == BASELINE else r for r in rows["representacao"]]
    colors = [grey if r == BASELINE else blue for r in rows["representacao"]]
    x = np.arange(len(rows))
    n_train = int(dummy["n_participantes_treino"])
    n_test = int(dummy["n_participantes_teste"])
    panels = (
        ("rmse_nested_cv", f"Nested CV no treino ({n_train} pessoas)"),
        ("rmse_teste", f"Teste ({n_test} pessoas, uma avaliação)"),
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, (column, title) in zip(axes, panels):
        values = rows[column].to_numpy()
        axis.bar(x, values, 0.62, color=colors)
        axis.axhline(dummy[column], color=line, lw=1, ls=(0, (4, 3)))
        for xi, value in zip(x, values):
            axis.text(xi, value / 2, f"{value:.1f}".replace(".", ","), ha="center",
                      va="center", fontsize=9, color="white")
        axis.set_xticks(x)
        axis.set_xticklabels(names, fontsize=9, color=ink)
        axis.set_title(title, fontsize=11, color=ink, pad=18)
        axis.text(0.5, 1.0, f"tracejado = sem PPG (sempre {mean_glc:.0f} mg/dL): "
                  f"{dummy[column]:.1f}".replace(".", ","), transform=axis.transAxes,
                  ha="center", va="bottom", fontsize=8, color=muted)
        axis.grid(axis="y", color=grid, lw=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=muted, labelsize=8)
    spread = rows.loc[rows["representacao"] != BASELINE, "desvio_dif_rmse_vs_baseline"]
    axes[0].text(
        0.5, 0.97,
        f"a diferença para a baseline varia ±{spread.min():.0f} a ±{spread.max():.0f} mg/dL entre folds".replace(".", ","),
        transform=axes[0].transAxes, ha="center", va="top", fontsize=8, color=muted,
    )
    axes[0].set_ylabel("erro da glicose, RMSE (mg/dL)", fontsize=9, color=muted)
    axes[0].set_ylim(0, max(rows[["rmse_nested_cv", "rmse_teste"]].to_numpy().max(),
                            dummy["rmse_nested_cv"]) * 1.22)
    figure.suptitle(
        "Representações novas ao lado da baseline (as 72 features originais) — All-PPGs, PPG",
        fontsize=12, color=ink,
    )
    figure.text(
        0.5, 0.005,
        "Cinza = baseline (72 features originais); azul = representações novas construídas a partir delas. "
        f"Todas com o mesmo regressor ({MODEL_NAME}), só como instrumento de medida.",
        ha="center", fontsize=8, color=muted,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(figure)


def plot_compactness(final: pd.DataFrame, path: Path) -> None:
    """Quantas colunas cada representação usa × quanto ela erra (RMSE, nested CV).

    É o resumo da *nova representação*: o eixo x é o tamanho da representação
    (colunas que o modelo recebe) e o eixo y, o erro. A linha cinza é a
    baseline (as 72 originais).
    """
    blue, grey = "#2a78d6", "#898781"
    ink, muted, grid = "#0b0b0b", "#52514e", "#e1e0d9"
    baseline = final.loc[final["representacao"] == BASELINE].iloc[0]
    rows = final.loc[final["representacao"] != DUMMY].reset_index(drop=True)

    def n_columns(row) -> int:
        params = row["hiperparametros"]
        if "extract__n_components" in params:
            return int(params.split("extract__n_components': ")[1].split(",")[0])
        return len(EXPECTED_REPR) if row["representacao"] == "4 fórmulas" else 72

    sizes = rows.apply(n_columns, axis=1).to_numpy()
    errors = rows["rmse_nested_cv"].to_numpy()
    spread = rows["desvio_dif_rmse_vs_baseline"].to_numpy()

    figure, axis = plt.subplots(figsize=(8.5, 4.6))
    axis.axhline(baseline["rmse_nested_cv"], color=grey, lw=1.2, ls=(0, (4, 3)))
    axis.text(40, baseline["rmse_nested_cv"] + 0.25,
              f"baseline (72 originais): {baseline['rmse_nested_cv']:.1f}".replace(".", ","),
              ha="right", va="bottom", fontsize=8, color=muted)
    axis.errorbar(sizes, errors, yerr=spread, fmt="none", ecolor=muted, elinewidth=1, capsize=3)
    point_colors = [grey if r == BASELINE else blue for r in rows["representacao"]]
    axis.scatter(sizes, errors, s=70, color=point_colors, zorder=3,
                 edgecolor="#fcfcfb", linewidth=1.5)
    for size, error, name in zip(sizes, errors, rows["representacao"]):
        label = f"{name}\n{size} colunas · {error:.1f}".replace(".", ",")
        axis.annotate(label, (size, error), xytext=(8, -2), textcoords="offset points",
                      fontsize=8.5, color=ink, va="top",
                      bbox={"facecolor": "#fcfcfb", "edgecolor": "none", "pad": 1.5})
    axis.set_xscale("log")
    axis.set_xticks([2, 4, 8, 16, 32, 72])
    axis.set_xticklabels(["2", "4", "8", "16", "32", "72"])
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.set_xlim(1.5, 110)
    axis.set_xlabel("colunas que o modelo recebe (escala logarítmica)", fontsize=9, color=muted)
    axis.set_ylabel("RMSE na nested CV (mg/dL)", fontsize=9, color=muted)
    axis.set_title(
        "Quantas colunas cada representação usa e quanto ela erra",
        fontsize=11, color=ink,
    )
    axis.grid(color=grid, lw=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=muted, labelsize=8)
    figure.text(
        0.5, 0.005,
        "Barras: ± desvio da diferença para a baseline entre os 25 folds. "
        f"Componentes do PCA/PLS: os escolhidos no ajuste final. Todas com {MODEL_NAME}.",
        ha="center", fontsize=7.5, color=muted,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("json_file", help="JSON All-PPGs/PPG do experimento.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    train, test = load_train_test(args.json_file)

    cv_table = nested_cv_table(train)
    final = test_table(train, test, cv_table)

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(RESULTS_FOLDER / "avaliacao_final_nested_cv.csv", index=False, float_format="%.3f")
    final.to_csv(RESULTS_FOLDER / "avaliacao_final.csv", index=False, float_format="%.3f")
    plot_final(final, RESULTS_FOLDER / "03_avaliacao_final.png")
    plot_compactness(final, RESULTS_FOLDER / "04_colunas_vs_erro.png")

    columns = ["representacao", "modelo", "mae_nested_cv", "desvio_mae",
               "rmse_nested_cv", "mae_teste", "rmse_teste", "hiperparametros"]
    print(f"\nNested CV: {N_SPLITS} partes externas x {N_REPEATS} repetições, "
          f"{N_INNER_SPLITS} internas. Teste avaliado uma vez por representação.")
    print(final[columns].round(1).to_string(index=False))
    print(f"\nresultados em: {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
