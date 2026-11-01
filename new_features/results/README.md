# Resultados — Grupo 3 / All-PPGs / PPG

Os arquivos desta pasta registram os resultados agregados da **nested CV com GroupKFold por participante** (5 folds externos × 3 internos). Os resultados a apresentar são os três arquivos `avaliacao_final*` e `03_avaliacao_final.png`; os demais são da análise exploratória anterior, só com nested CV e gerados com a lista de k antiga do KNN (3 a 15).

| Arquivo | Significado |
|---|---|
| `avaliacao_final.csv` | **Resultado final** (nested CV + teste): modelo escolhido de cada representação, erro na nested CV e no teste |
| `avaliacao_final_nested_cv.csv` | Nested CV da avaliação final, todas as combinações representação × modelo |
| `03_avaliacao_final.png` | Figura da avaliação final |
| `avaliacao_formulas.csv` | (exploratório) Quatro fórmulas, 72 originais e 72 originais + quatro novas (4 regressores e Dummy) |
| `avaliacao_pca_cv.csv` | (exploratório) PCA com 4, 8 e 12 componentes × 4 regressores e Dummy |
| `avaliacao_pls_cv.csv` | (exploratório) PLS com 2, 4, 8 e 12 componentes × 4 regressores e Dummy |
| `redundancia.csv` | Correlação de Spearman de cada fórmula nova com a original mais associada, no treino All-PPGs |
| `comparacao_representacoes.csv` | (exploratório) **Quatro linhas** selecionadas, extraídas automaticamente dos três CSVs anteriores |
| `01_comparacao_mae.png` | (exploratório) Gráfico do MAE das quatro combinações selecionadas |
| `02_comparacao_rmse.png` | (exploratório) Gráfico do RMSE das quatro combinações selecionadas |

**A seleção das quatro linhas** corresponde à tabela apresentada ao grupo e está declarada em `../config/comparacao_destaques.json`. Não equivale ao menor MAE ou RMSE de todos os métodos, pois as métricas podem apontar para modelos diferentes.

**Atualizar os gráficos e o CSV-resumo**, da raiz do MultiWaveLab2, após novas execuções dos scripts de avaliação:

```bash
python new_features/visualize_cv_results.py
```

A geração de imagens não utiliza dados individuais, arquivos de teste ou figuras históricas. Se alguma combinação configurada não estiver disponível, a execução **falha com um erro**, sem reutilizar valores antigos. Consulte `../README.md` para o protocolo, métodos e comandos completos de reprodução.

**Observação de precisão:** `avaliacao_formulas.csv` foi gravado com uma casa decimal; os CSVs de PCA/PLS têm três casas. No gráfico mostramos duas casas apenas para padronizar a apresentação.

**Compartilhamento:** os CSVs desta pasta trazem identificadores de participantes e valores de glicose. Antes de publicar externamente, verificar as regras aplicáveis aos dados originais.
