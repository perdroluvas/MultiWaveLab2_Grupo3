# Resultados — Grupo 3 / All-PPGs / PPG

Resultados da avaliação das representações (72 originais, 4 fórmulas, PCA e PLS), todas com o mesmo regressor (SVM), comparadas com a baseline (as 72 originais): **nested CV com `GroupKFold` por participante** no treino (5 folds externos × 5 repetições, 3 internos) e **uma avaliação no teste oficial**.

| Arquivo | Significado |
|---|---|
| `01_as_quatro_features.png` | As 4 fórmulas contra a glicose (uma média por pessoa, treino) |
| `02_mapa_correlacao.png` | Correlação entre as 4 fórmulas e a glicose |
| `03_avaliacao_final.png` | RMSE de cada representação nova ao lado da baseline (72 originais), na nested CV e no teste |
| `04_colunas_vs_erro.png` | Quantas colunas cada representação usa × RMSE na nested CV |
| `05_pca_variancia.png` | Quanto da variação das 72 features o PCA guarda com cada número de componentes |
| `06_pca_pls_glicose.png` | As mesmas gravações do treino nos 2 primeiros componentes do PCA e do PLS, coloridas pela glicose |
| `avaliacao_final.csv` | Nested CV e teste de cada representação, com a diferença para a baseline |
| `avaliacao_final_nested_cv.csv` | Nested CV de cada representação (média e desvio entre os 25 folds externos) |
| `resumo_estatistico.csv` | Média, desvio, mínimo, máximo e correlação com a glicose das 4 fórmulas |
| `redundancia.csv` | Correlação de Spearman de cada fórmula nova com a característica original mais parecida, no treino do All-PPGs |

**Atualizar**, a partir de `src/`, depois de gerar as colunas `repr_` (ver `../README.md`, seção 7):

```bash
PYTHONPATH=. python ../new_features/evaluate_final.py ../config_experiments/input_all_ppgs/exp0.json
PYTHONPATH=. python ../new_features/visualize_pca_pls.py ../config_experiments/input_all_ppgs/exp0.json
PYTHONPATH=. python ../new_features/visualize_representation.py ../config_experiments/input_all_ppgs/exp0.json
```

**Compartilhamento:** os CSVs desta pasta trazem identificadores de participantes e valores de glicose. Antes de publicar externamente, verificar as regras aplicáveis aos dados originais.
