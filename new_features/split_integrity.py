"""Validação dos conjuntos de treino e teste para a tarefa de Feature Extraction."""

from __future__ import annotations

import pandas as pd


def resolve_official_overlap(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], int]:
    """Keep the official reserved test and exclude its shared participant from development.

    The upstream All-PPGs manual train/test lists both contain ``ieb_01``.
    This adjustment changes only the derived Group 3 representation, not the
    official participant lists or original features CSVs.
    """
    shared = sorted(
        set(df_train["participant_id"].astype(str))
        & set(df_test["participant_id"].astype(str))
    )
    if not shared:
        return df_train, [], 0
    if shared != ["ieb_01"]:
        raise ValueError(
            f"Sobreposição inesperada entre treino e teste: {shared}. "
            "Apenas ieb_01 está documentado nas listas oficiais All-PPGs."
        )
    cleaned = df_train.loc[
        ~df_train["participant_id"].astype(str).isin(shared)
    ].copy()
    removed = len(df_train) - len(cleaned)
    if cleaned.empty:
        raise ValueError("O conjunto de treino ficou vazio após resolver a sobreposição.")
    print(
        "Divisão oficial All-PPGs: ieb_01 aparece em treino e teste. "
        f"Removendo {removed} linha(s) de ieb_01 somente da representação "
        "de treino; o teste oficial foi preservado."
    )
    return cleaned, shared, removed


