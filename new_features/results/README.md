# Resultados — Grupo 3 / All-PPGs / PPG

Os arquivos desta pasta registram os resultados agregados da **nested CV com GroupKFold por participante** (5 folds externos × 3 internos). **A subpasta `ieb1_legacy/` é histórica** e NÃO é lida pelo script de gráficos agregados.

| Arquivo | Significado |
|---|---|
| `avaliacao_formulas.csv` | Quatro fórmulas, 72 originais e 72 originais + quatro novas (4 regressores e Dummy) |
| `avaliacao_pca_cv.csv` | PCA com 4, 8 e 12 componentes × 4 regressores e Dummy |
| `avaliacao_pls_cv.csv` | PLS com 2, 4, 8 e 12 componentes × 4 regressores e Dummy |
| `redundancia.csv` | Correlação de Spearman de cada fórmula nova com a original mais associada, no treino All-PPGs |
| `comparacao_representacoes.csv` | **Quatro linhas** selecionadas, extraídas automaticamente dos três CSVs anteriores |
| `01_comparacao_mae.png` | Gráfico do MAE das quatro combinações selecionadas |
| `02_comparacao_rmse.png` | Gráfico do RMSE das quatro combinações selecionadas |
| `ieb1_legacy/` | Histórico IEB-1 (preservado, não comparado numericamente ao All-PPGs) |

**A seleção das quatro linhas** corresponde à tabela apresentada ao grupo e está declarada em `../config/comparacao_destaques.json`. Não equivale ao menor MAE ou RMSE de todos os métodos, pois as métricas podem apontar para modelos diferentes.

**Atualizar os gráficos e o CSV-resumo**, da raiz do MultiWaveLab2, após novas execuções dos scripts de avaliação:

```bash
python new_features/visualize_cv_results.py
```

A geração de imagens não utiliza dados individuais, arquivos de teste ou figuras históricas. Se alguma combinação configurada não estiver disponível, a execução **falha com um erro**, sem reutilizar valores antigos. Consulte `../README.md` para o protocolo, métodos e comandos completos de reprodução.

**Observação de precisão:** `avaliacao_formulas.csv` foi gravado com uma casa decimal; os CSVs de PCA/PLS têm três casas. No gráfico mostramos duas casas apenas para padronizar a apresentação.

**Compartilhamento:** a subpasta IEB-1 contém arquivos por participante. Antes de publicar este ZIP externamente, verificar as regras aplicáveis aos dados originais.
