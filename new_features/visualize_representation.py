"""
Gerar as visualizações da representação construída a partir das features originais.

As figuras são gravadas em ``features0_stats/figures/<sufixo>/`` (ou na pasta
indicada com ``--output-dir``). O script lê o arquivo ``features0_repr.csv``
gerado por ``run_feature_representation.py``.

Uso, a partir da pasta ``src``::

    python ../new_features/visualize_representation.py \
        ../config_experiments/input_ieb1/exp0.json

Figuras geradas
---------------
01_as_quatro_features.png
    As quatro features novas juntas (2x2), cada uma contra a GLC, com a
    correlação de Spearman e a reta de tendência. Usa todos os participantes.
02_mapa_correlacao.png
    Mapa de calor da correlação (Spearman) entre as quatro features e a GLC.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _bootstrap import ensure_import_paths

ensure_import_paths()

from datasets_util.naming_conventions import DatasetConfig
from feature_representation import REPR_PREFIX, RATIO_RULES, TARGET_COLUMN

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
FIGURE_NAMES = ("01_as_quatro_features.png", "02_mapa_correlacao.png")

EXPECTED_COLUMNS = tuple(REPR_PREFIX + rule.name for rule in RATIO_RULES)

TARGET_LABEL = "Glicose"

FEATURE_LABELS = {
    "repr_ppg_beat_rate_hz": "Ritmo do coração",
    "repr_ppg_b2_width_over_period": "Largura do pulso",
    "repr_ppg_b2_crest_time_over_period": "Subida do pulso",
    "repr_ppg_p1_rr_cv": "Irregularidade do ritmo",
}

FEATURE_DESCRIPTIONS = {
    "repr_ppg_beat_rate_hz": "batimentos por segundo",
    "repr_ppg_b2_width_over_period": "parte do batimento com a onda acima da metade da altura",
    "repr_ppg_b2_crest_time_over_period": "parte do batimento até a onda chegar ao pico",
    "repr_ppg_p1_rr_cv": "quanto o intervalo entre batimentos varia",
}

FEATURE_AXIS_LABELS = {
    "repr_ppg_beat_rate_hz": "batimentos por segundo (1,2 = 72 por minuto)",
    "repr_ppg_b2_width_over_period": "fração do batimento (0 a 1)",
    "repr_ppg_b2_crest_time_over_period": "fração do batimento (0 a 1)",
    "repr_ppg_p1_rr_cv": "variação relativa (0 = ritmo constante)",
}


def _significance(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def _expected_columns(df: pd.DataFrame) -> list[str]:
    missing = [name for name in EXPECTED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(
            f"Features novas ausentes: {missing}. "
            "Rode o run_feature_representation.py primeiro."
        )
    return list(EXPECTED_COLUMNS)


def _n_participants(df: pd.DataFrame) -> int:
    if "participant_id" in df.columns:
        return int(df["participant_id"].nunique())
    return len(df)


def plot_four_features(df: pd.DataFrame, output: Path) -> None:
    """As quatro features juntas, cada uma contra a GLC."""
    if TARGET_COLUMN not in df.columns:
        print("dados insuficientes para a figura 01")
        return
    columns = _expected_columns(df)
    if len(columns) != 4:
        raise ValueError(f"A figura espera 4 features, encontrei {len(columns)}.")

    figure, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for axis, name in zip(axes.ravel(), columns):
        heading = (
            f"{FEATURE_LABELS.get(name, name)}\n"
            f"({FEATURE_DESCRIPTIONS.get(name, '')})"
        )
        x = pd.to_numeric(df[name], errors="coerce")
        y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        mask = x.notna() & y.notna()
        axis.scatter(
            x[mask], y[mask], s=30, alpha=0.75, color="#4C72B0",
            edgecolor="white", linewidth=0.5,
        )
        if mask.sum() >= 2 and x[mask].nunique() > 1:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            grid = np.linspace(x[mask].min(), x[mask].max(), 50)
            axis.plot(grid, slope * grid + intercept, color="#C44E52", linewidth=1.6)
            rho, pvalue = spearmanr(x[mask], y[mask])
            stats = (
                f"correlação com a glicose: ρ = {rho:+.2f}  (p = {pvalue:.3f}) "
                .replace(".", ",")
                + _significance(pvalue)
            )
            axis.set_title(f"{heading}\n{stats}", fontsize=11)
        else:
            axis.set_title(heading, fontsize=11)
        axis.set_xlabel(FEATURE_AXIS_LABELS.get(name, ""), fontsize=9)
        axis.set_ylabel("glicose (mg/dL)", fontsize=9)
        axis.tick_params(labelsize=9)
        axis.grid(alpha=0.3)

    figure.suptitle(
        f"As quatro features novas × glicose "
        f"({_n_participants(df)} participantes, um ponto por pessoa)",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "ρ = correlação de Spearman (−1 a +1; 0 = sem relação)   "
        "* p < 0,05   ** p < 0,01   *** p < 0,001   n.s. = não significativo",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    path = output / "01_as_quatro_features.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"figura salva em: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, output: Path) -> None:
    """Correlação de Spearman entre as features novas e a GLC."""
    if TARGET_COLUMN not in df.columns:
        print("dados insuficientes para a figura 02")
        return
    columns = _expected_columns(df)

    subset = df[columns + [TARGET_COLUMN]].apply(pd.to_numeric, errors="coerce")
    matrix = subset.corr(method="spearman")

    values = matrix.to_numpy().copy()
    np.fill_diagonal(values, np.nan)

    figure, axis = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    image = axis.imshow(values, cmap=cmap, vmin=-1, vmax=1)
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_yticks(range(len(matrix.columns)))
    labels = [
        TARGET_LABEL if c == TARGET_COLUMN
        else FEATURE_LABELS.get(c, c.replace(REPR_PREFIX, ""))
        for c in matrix.columns
    ]
    axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    axis.set_yticklabels(labels, fontsize=9)

    for row in range(len(matrix)):
        for col in range(len(matrix)):
            if row == col:
                continue
            value = matrix.iat[row, col]
            axis.text(
                col, row, f"{value:.2f}".replace(".", ","),
                ha="center", va="center", fontsize=9,
                color="white" if abs(value) > 0.5 else "black",
            )

    figure.colorbar(
        image, ax=axis,
        label="correlação (−1 = inversa, 0 = nenhuma, +1 = direta)",
    )
    axis.set_title(
        "Quanto as 4 features e a glicose andam juntas\n"
        f"(correlação de Spearman, {_n_participants(df)} participantes)",
        fontsize=11,
    )
    figure.tight_layout()
    path = output / "02_mapa_correlacao.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"figura salva em: {path}")


def write_result_tables(df: pd.DataFrame, folder: Path) -> None:
    """Grava as tabelas de resultados que acompanham a apresentação."""
    if TARGET_COLUMN not in df.columns:
        print("dados insuficientes para as tabelas de resultados")
        return
    columns = _expected_columns(df)

    id_columns = [c for c in ("participant_id", TARGET_COLUMN) if c in df.columns]
    df[id_columns + columns].to_csv(
        folder / "dados_por_participante.csv", index=False
    )

    rows = []
    for name in columns:
        x = pd.to_numeric(df[name], errors="coerce")
        y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 2 and x[mask].nunique() > 1:
            rho, pvalue = spearmanr(x[mask], y[mask])
        else:
            rho, pvalue = float("nan"), float("nan")
        rows.append(
            {
                "feature": name,
                "media": round(x.mean(), 4),
                "desvio_padrao": round(x.std(), 4),
                "minimo": round(x.min(), 4),
                "maximo": round(x.max(), 4),
                "spearman_rho_glc": round(rho, 3),
                "p_valor": round(pvalue, 4),
                "n": int(mask.sum()),
            }
        )
    pd.DataFrame(rows).to_csv(folder / "resumo_estatistico.csv", index=False)
    print(f"tabelas salvas em: {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizações da nova representação.")
    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="JSON do experimento (o mesmo usado no run_feature_representation.py).",
    )
    parser.add_argument(
        "--suffix",
        default="_repr",
        help="Sufixo do arquivo de features com a representação (padrão: _repr).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pasta das figuras (padrão: <statistics>/figures/<sufixo>).",
    )
    args = parser.parse_args()

    dataset_config = DatasetConfig(args.json_file)
    features_folder = Path(dataset_config.get_dataset_features_path())
    stem = Path(dataset_config.get_features_file_name()).stem
    input_file = features_folder / f"{stem}{args.suffix}.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {input_file}\n"
            "Rode primeiro o run_feature_representation.py para gerar a nova representação."
        )

    output = Path(args.output_dir) if args.output_dir else (
        Path(dataset_config.get_statistics_output_folder())
        / "figures"
        / args.suffix.strip("_")
    )
    output.mkdir(parents=True, exist_ok=True)

    print(f"lendo: {input_file}")
    df = pd.read_csv(input_file)
    print(
        f"linhas: {len(df)} | participantes: {_n_participants(df)} | "
        f"colunas repr_: {sum(c.startswith(REPR_PREFIX) for c in df.columns)}"
    )

    plot_four_features(df, output)
    plot_correlation_heatmap(df, output)

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    write_result_tables(df, RESULTS_FOLDER)
    shutil.copy2(input_file, RESULTS_FOLDER / input_file.name)
    report_file = (
        Path(dataset_config.get_statistics_output_folder())
        / "feature_representation_report.json"
    )
    if report_file.exists():
        shutil.copy2(report_file, RESULTS_FOLDER / report_file.name)
    if output.resolve() != RESULTS_FOLDER.resolve():
        for name in FIGURE_NAMES:
            source = output / name
            if source.exists():
                shutil.copy2(source, RESULTS_FOLDER / name)

    print(f"\nfiguras em: {output}")
    print(f"resultados em: {RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
