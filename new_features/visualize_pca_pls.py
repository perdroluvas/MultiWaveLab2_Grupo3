"""
Gráficos que explicam as representações PCA e PLS (sem comparar modelos).

05_pca_variancia.png
    Quanto da variação das 72 features originais o PCA guarda com 1, 2, ...
    componentes; marca 2, 4, 8 e 12, os tamanhos testados na avaliação.
06_pca_pls_glicose.png
    O mesmo treino visto pelos 2 primeiros componentes do PCA (construídos sem
    a glicose) e do PLS (construídos com a glicose), cada gravação colorida
    pela glicose.

Tudo é ajustado só no treino; o teste não é usado. Uso, a partir de ``src/``::

    PYTHONPATH=. python ../new_features/visualize_pca_pls.py \\
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from _bootstrap import ensure_import_paths

ensure_import_paths()

from evaluate_final import COMPONENTS, _columns, _matrix
from evaluate_formulas import RESULTS_FOLDER, load_train_test
from evaluate_pls import PLSFeatureExtractor
from feature_representation import TARGET_COLUMN

INK, MUTED, GRID, BLUE, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6", "#fcfcfb"


def _standardized_train(train) -> np.ndarray:
    X = _matrix(train, _columns(train, "72 originais"))
    return StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))


def _style(axis) -> None:
    axis.grid(color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=8)


def plot_pca_variance(X: np.ndarray, path: Path) -> None:
    """Variância acumulada das 72 features guardada por k componentes."""
    cumulative = np.cumsum(PCA(svd_solver="full").fit(X).explained_variance_ratio_) * 100
    ks = np.arange(1, 31)

    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    axis.plot(ks, cumulative[:30], color=BLUE, lw=2)
    for k in COMPONENTS:
        value = cumulative[k - 1]
        axis.scatter([k], [value], s=50, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        axis.annotate(f"{k} componentes: {value:.0f}%", (k, value), xytext=(8, -14),
                      textcoords="offset points", fontsize=8.5, color=INK)
    axis.set_xlim(0, 31)
    axis.set_ylim(0, 100)
    axis.set_xlabel("componentes do PCA (colunas da nova representação)", fontsize=9, color=MUTED)
    axis.set_ylabel("variação das 72 features guardada (%)", fontsize=9, color=MUTED)
    axis.set_title("Poucos componentes do PCA guardam a maior parte das 72 features",
                   fontsize=11, color=INK)
    _style(axis)
    figure.text(0.5, 0.005, "PCA ajustado no treino do All-PPGs (130 gravações), depois de "
                "preencher faltantes e padronizar as 72 colunas.",
                ha="center", fontsize=7.5, color=MUTED)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_pca_pls_glucose(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """2 primeiros componentes do PCA e do PLS, coloridos pela glicose."""
    pca = PCA(n_components=2, svd_solver="full").fit_transform(X)
    pls = PLSFeatureExtractor(n_components=2).fit(X, y).transform(X)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = (
        (axes[0], pca, "PCA: componentes construídos sem a glicose"),
        (axes[1], pls, "PLS: componentes construídos com a glicose do treino"),
    )
    for axis, scores, title in panels:
        points = axis.scatter(scores[:, 0], scores[:, 1], c=y, cmap="Blues", vmin=40,
                              vmax=y.max(), s=34, edgecolor=MUTED, linewidth=0.4)
        axis.set_title(title, fontsize=10.5, color=INK)
        axis.set_xlabel("componente 1", fontsize=9, color=MUTED)
        axis.set_ylabel("componente 2", fontsize=9, color=MUTED)
        _style(axis)
    figure.subplots_adjust(left=0.06, right=0.88, bottom=0.14, top=0.86, wspace=0.22)
    bar = figure.colorbar(points, cax=figure.add_axes([0.905, 0.16, 0.015, 0.68]))
    bar.set_label("glicose (mg/dL)", fontsize=9, color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    figure.suptitle("As mesmas 130 gravações em duas representações novas de 2 colunas",
                    fontsize=12, color=INK, x=0.45)
    figure.text(0.45, 0.005, "Cada ponto é uma gravação do treino; quanto mais escuro, maior a "
                "glicose. No PLS, a glicose entra na construção dos componentes.",
                ha="center", fontsize=7.5, color=MUTED)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("json_file", help="JSON All-PPGs/PPG do experimento.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    train, _ = load_train_test(args.json_file)
    X = _standardized_train(train)
    y = train[TARGET_COLUMN].to_numpy(dtype=float)

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plot_pca_variance(X, RESULTS_FOLDER / "05_pca_variancia.png")
    plot_pca_pls_glucose(X, y, RESULTS_FOLDER / "06_pca_pls_glicose.png")
    print(f"figuras em: {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
