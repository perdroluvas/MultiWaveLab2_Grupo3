"""Atualiza a comparação *selecionada* a partir dos CSVs da nested CV.

Executar da raiz do MultiWaveLab2::

    python new_features/visualize_cv_results.py

As quatro combinações mostradas são definidas em
``config/comparacao_destaques.json``. Elas NÃO são escolhidas automaticamente
pelo menor MAE ou RMSE: são uma seleção explícita para apresentação.

Fontes: results/avaliacao_formulas.csv, avaliacao_pca_cv.csv,
avaliacao_pls_cv.csv. A pasta histórica IEB-1 não é consultada.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_CONFIG = ROOT / "config" / "comparacao_destaques.json"
REQUIRED_COLUMNS = {"features", "modelo", "mae_medio", "rmse_medio"}
OUTPUT_COLUMNS = ["representacao", "modelo", "mae_mg_dl", "rmse_mg_dl", "origem"]


def build_comparison(results_dir: Path, config_path: Path) -> pd.DataFrame:
    """Busca as linhas exatas dos destaques, falhando em caso de CSV ausente/ambíguo."""
    entries = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("A configuração deve conter uma lista não vazia de destaques.")

    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for item in entries:
        for key in ("representacao", "arquivo", "features", "modelo"):
            if key not in item or not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"Destaque com campo ausente/inválido: {key}")
        source = item["arquivo"]
        if Path(source).name != source or not source.endswith(".csv"):
            raise ValueError(f"Nome de arquivo inválido na configuração: {source}")
        if source not in cache:
            path = results_dir / source
            if not path.is_file():
                raise FileNotFoundError(f"CSV necessário não encontrado: {path}")
            table = pd.read_csv(path)
            missing = REQUIRED_COLUMNS - set(table.columns)
            if missing:
                raise ValueError(f"{source} sem colunas obrigatórias: {sorted(missing)}")
            cache[source] = table
        key = (source, item["features"], item["modelo"])
        if key in seen:
            raise ValueError(f"Combinação repetida na configuração: {key}")
        seen.add(key)
        table = cache[source]
        matched = table.loc[(table["features"] == item["features"]) &
                            (table["modelo"] == item["modelo"])]
        if len(matched) != 1:
            raise ValueError(
                f"Esperada uma única linha em {source} para "
                f"features={item['features']!r}, modelo={item['modelo']!r}; "
                f"encontradas {len(matched)}. Verifique os CSVs e a configuração."
            )
        match = matched.iloc[0]
        mae = float(match["mae_medio"])
        rmse = float(match["rmse_medio"])
        if not np.isfinite(mae) or not np.isfinite(rmse) or mae < 0 or rmse < 0:
            raise ValueError(f"Erro não numérico ou inválido em {source}: {key}")
        rows.append({
            "representacao": item["representacao"],
            "modelo": item["modelo"],
            "mae_mg_dl": mae,
            "rmse_mg_dl": rmse,
            "origem": source,
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def plot_comparison(compared: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    vals = compared[metric].astype(float).to_numpy()
    labels = [f"{label}  ·  {model}" for label, model in
              zip(compared["representacao"], compared["modelo"])]
    ypos = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(10.8, 5), layout="constrained")
    # Cores distintas para distinguir as quatro famílias, com eixos comuns.
    palette = ["#16796a", "#487cae", "#b38a40", "#777e89", "#946bb5"]
    ax.barh(ypos, vals, height=0.64, color=[palette[i % len(palette)] for i in range(len(vals))])
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.21)
    ax.set_xlabel("Erro médio (mg/dL)  ·  menor é melhor")
    ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    for y, val in zip(ypos, vals):
        ax.text(val + max(vals) * .018, y, f"{val:.2f}".replace(".", ","),
                va="center", fontsize=10, fontweight="medium")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Diretório com os três CSVs de avaliação (padrão: new_features/results).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON com as combinações a destacar.")
    args = parser.parse_args(argv)
    result_dir = args.results_dir
    comparison = build_comparison(result_dir, args.config)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "comparacao_representacoes.csv"
    comparison.to_csv(csv_path, index=False, encoding="utf-8")
    plot_comparison(comparison, "mae_mg_dl", "MAE — configurações selecionadas",
                    result_dir / "01_comparacao_mae.png")
    plot_comparison(comparison, "rmse_mg_dl", "RMSE — configurações selecionadas",
                    result_dir / "02_comparacao_rmse.png")
    print("Comparação atualizada a partir dos CSVs:")
    print(comparison.to_string(index=False))
    print(f"Arquivos gerados: {csv_path.name}, 01_comparacao_mae.png e 02_comparacao_rmse.png")
    print(f"Pasta: {result_dir}")


if __name__ == "__main__":
    main()
