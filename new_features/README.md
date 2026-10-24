# Nova representação a partir das features originais

Tema 5 do Grupo 3. Este documento tem quatro partes: **(1)** a extração de
features que já existe no MultiWaveLab, **(2)** como ela funciona no código
original, **(3)** as **quatro features novas** que criamos — e, no fim, **(4)**
os comandos para rodar tudo e ver a figura.

Antes de tudo, a distinção que importa:

```
sinal PPG ──(extração, JÁ EXISTE no projeto)──► 72 features por segmento
                                                        │
                                          (NOSSA TAREFA) └──► 4 colunas novas
```

- **Extrair features** = transformar o sinal em números. Já está pronto.
- **Construir uma nova representação** = a partir desses números, criar
  *outras* colunas que o modelo enxerga melhor. É o que fizemos.

---

## 1. A extração de features já existe no MultiWaveLab

O projeto transforma cada segmento de sinal em uma linha de **72 números**.
As features de cada modalidade ficam em `features0_ppg.csv`; a fusão gera o
`features0.csv`, uma linha por segmento, com 72 features:

| participant_id | GLC | ppg_b2_N_BEATS | duration | ppg_b2_T_median | ppg_h4_PRV_HRV_RMSSD | ... |
|---|---|---|---|---|---|---|
| hu_01 | 181 | 353 | 18271 | 0.867 | 26.36 | ... (72 features) |

As etapas que produzem esse arquivo:

```
src/signal_processing/ppg.py            ← processa o sinal (batidas, PSD, HRV...)
src/segments/create_segment_files.py    ← recorta os trechos bons do sinal
src/features/extract_features_for_all_modalities.py  ← aplica as features
src/features/fusion_of_multimodality_features.py     ← junta as modalidades
src/machinelearning/train_test_splits.py             ← separa treino/teste
```

Ou seja: **é só chamar o pipeline que a tabela de 72 features aparece.**

---

## 2. Como a extração funciona no código original

O caminho é sempre o mesmo: para cada linha da tabela de segmentos, carregar o
sinal, recortar o trecho e chamar o extrator da modalidade.

**`src/features/extract_features_for_all_modalities.py:177`**

```python
def extract_features_for_segment(row, fs_dict, datasetConfig):
    signal = load_signal(path)                      # lê o sinal do arquivo
    signal = signal[start_sample:start_sample+duration]   # recorta o segmento

    if modality == "ppg":
        features = extract_ppg_features(datasetConfig, signal, fs_dict["ppg"])
    return features
```

O extrator de PPG (`src/signal_processing/ppg.py:1035`) usa o NeuroKit2 para
achar as batidas e, a partir delas, calcula as features em grupos:

```python
signals, info = nk.ppg_process(ppg, sampling_rate=fs)   # acha os picos
peaks = np.sort(info["PPG_Peaks"])

features1_peaks   = extract_features_group1_peaks(...)     # p1: HR, amplitude
features2_beats   = extract_features_group2_beats(...)     # b2: forma da batida
features3_spectral= extract_features_group3_spectral(...)  # s3: espectro (PSD)
features4_hrv     = extract_features_group4_prv(...)       # h4: variabilidade
features6_imf     = extract_features_group6_imf(...)       # i6: EMD/IMF

return {**features1_peaks, **features2_beats, ...}
```

Cada grupo vira colunas com um prefixo, por exemplo:

| grupo | exemplo de coluna | o que mede |
|---|---|---|
| `p1` | `ppg_p1_hr_mean` | frequência cardíaca média |
| `b2` | `ppg_b2_T_median` | período médio entre batidas (s) |
| `b2` | `ppg_b2_width_median` | largura da sístole |
| `s3` | `ppg_s3_MAX_POWER_1st` | potência do 1º harmônico |
| `h4` | `ppg_h4_PRV_HRV_RMSSD` | variabilidade batida-a-batida |
| `i6` | `ppg_i6_IMF_1_dom_freq` | frequência dominante do 1º IMF |

Depois de extrair, o projeto limpa e imputa valores (`clean_features_dataframe.py`)
e grava o CSV:

```python
df_ppg = cleaning_and_imputation(df_ppg, datasetConfig, modality="ppg")
df_ppg.to_csv(output_file, index=False)   # features0_ppg.csv
```

**Resumo:** a extração está pronta e entrega, por segmento, uma linha com 72
números + metadados (participante, sessão, qualidade, `GLC`).

---

## 3. As quatro features novas

A regra é: **usar somente as 72 colunas originais**. Não voltamos ao sinal.
É aritmética entre colunas. O resultado são exatamente **quatro colunas**
novas, todas com o prefixo `repr_`.

### 3.1 Um exemplo real (participante `hu_01`, GLC = 181)

Valores originais usados:

| coluna original | valor |
|---|---|
| `ppg_b2_N_BEATS` | 353 |
| `duration` | 18271 **amostras** |
| `ppg_b2_T_median` | 0.867 s |
| `ppg_b2_width_median` | 0.283 s |
| `ppg_b2_Tr_median` | 0.250 s |
| `ppg_p1_peak_interval_std` | 0.0151 s |
| `ppg_p1_hr_mean` | 69.76 bpm |

As quatro features novas:

| # | coluna nova | conta | resultado | para que serve |
|---|---|---|---|---|
| 1 | `repr_ppg_beat_rate_hz` | `353 × 60 / 18271` | 1.159 Hz (69,5 bpm) | frequência cardíaca real, sem depender da duração do segmento |
| 2 | `repr_ppg_b2_width_over_period` | `0.283 / 0.867` | 0.327 | fração do ciclo gasta na sístole (adimensional) |
| 3 | `repr_ppg_b2_crest_time_over_period` | `0.250 / 0.867` | 0.288 | fração do ciclo até o pico sistólico (tempo de subida) |
| 4 | `repr_ppg_p1_rr_cv` | `0.0151 / 69.76 × 60` | 0.013 (1,3 %) | variabilidade batida-a-batida, relativa à FC |

No arquivo final ficam as 72 originais **+ 4 `repr_`** (fora os metadados).

> `hu_01` é só a linha de exemplo. O pipeline calcula as 4 features para
> **todos os 92 participantes** do IEB-1.

Resumo nos 92 participantes (uma medição por pessoa):

| feature | média | desvio | mín | máx | Spearman com GLC |
|---|---|---|---|---|---|
| `repr_ppg_beat_rate_hz` | 1.195 Hz | 0.195 | 0.779 | 1.734 | +0.33 (p = 0.001) |
| `repr_ppg_b2_width_over_period` | 0.379 | 0.052 | 0.295 | 0.529 | −0.19 (p = 0.070) |
| `repr_ppg_b2_crest_time_over_period` | 0.314 | 0.040 | 0.243 | 0.482 | +0.23 (p = 0.027) |
| `repr_ppg_p1_rr_cv` | 0.067 | 0.064 | 0.009 | 0.358 | −0.28 (p = 0.007) |

### 3.2 Por que essas quatro

| feature | o que ela captura | por que é útil |
|---|---|---|
| `beat_rate_hz` | **frequência** | `N_BEATS` sozinho depende do tamanho do segmento; dividir pela duração dá uma taxa comparável entre pessoas e gravações |
| `width_over_period` | **morfologia** (sístole) | vira uma fração adimensional: não depende do ganho do sensor nem da perfusão |
| `crest_time_over_period` | **morfologia** (subida) | tempo até o pico normalizado; relacionado à rigidez arterial |
| `rr_cv` | **variabilidade** | desvio-padrão dos intervalos RR dividido pela média (coeficiente de variação): mede o quanto o ritmo oscila, sem unidade |

As quatro cobrem **frequência, forma (2×) e variabilidade** — uma história
curta e fácil de apresentar em 1 minuto.

### 3.3 Qual é o objetivo disso

- **Remover o "ruído de pessoa/sensor":** razões eliminam unidade, ganho do
  aparelho e tamanho do segmento, então o modelo tende a aprender fisiologia
  em vez de "quem é o participante".
- **Resumir features redundantes:** `N_BEATS`, `N_PEAKS` e `F_MAX_1st` medem
  quase a mesma coisa (frequência cardíaca). Uma razão bem escolhida substitui
  várias colunas.
- **Testar a hipótese do grupo:** comparar o erro do modelo (MAE/RMSE) com e
  sem as colunas `repr_*`.

> As colunas novas nunca entram em cima de metadados (`participant_id`,
> `session_id`, `datetime`, `file_id`, `quality_indicator`, `GLC`). Todas usam
> o prefixo `repr_`.

---

## 4. Como rodar no terminal

Todos os comandos abaixo partem da pasta `src` do repositório, com o ambiente
local `.venv` ativo (Python 3.10.13):

```bash
cd /home/plbsa/Projects/IC/MultiWaveLab2_Grupo3
source .venv/bin/activate
cd src
```

### 4.1 Gerar as features originais (só se ainda não existirem)

```bash
PYTHONPATH=. python signal_processing/ppg.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python segments/create_segment_files.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python features/extract_features_for_all_modalities.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python features/fusion_of_multimodality_features.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python machinelearning/train_test_splits.py ../config_experiments/input_ieb1/exp0.json
```

Isso produz `features0.csv`, `features0_train.csv` e `features0_test.csv`.

### 4.2 Construir as quatro features novas

```bash
# arquivo fundido (todas as linhas)
PYTHONPATH=. python ../new_features/run_feature_representation.py \
    ../config_experiments/input_ieb1/exp0.json --repr ratio

# divisões de treino/teste
PYTHONPATH=. python ../new_features/run_feature_representation.py \
    ../config_experiments/input_ieb1/exp0.json --source train_test --repr ratio
```

Opções úteis:

| opção | efeito |
|---|---|
| `--repr ratio` | a única representação exposta (as 4 features) |
| `--spec arquivo.json` | usa uma configuração pronta (`config/feature_representation.json`) |
| `--source fused\|train_test` | aplica no arquivo fundido ou nas divisões |
| `--no-original` | grava as colunas novas + metadados (ids e alvo), sem as features originais |
| `--describe` | só descreve as features originais, sem gerar nada |

### 4.3 Gerar a figura

Depois de gerar a representação (passo 4.2):

```bash
PYTHONPATH=. python ../new_features/visualize_representation.py \
    ../config_experiments/input_ieb1/exp0.json
```

As figuras são geradas na pasta de saída do projeto e **copiadas para
`new_features/results/`**, junto com as tabelas de resultado:

| figura | o que mostra |
|---|---|
| `01_as_quatro_features.png` | as 4 features juntas (2×2), cada uma contra a GLC, com ρ de Spearman e reta de tendência (n = 92) |
| `02_mapa_correlacao.png` | mapa de calor da correlação de Spearman entre as 4 features e a GLC |

### 4.4 Arquivos gerados

| arquivo | conteúdo |
|---|---|
| `results/dados_por_participante.csv` | `participant_id`, `GLC` e as 4 features (92 linhas) |
| `results/resumo_estatistico.csv` | média, desvio, mín, máx, ρ e p-valor das 4 features |
| `results/features0_repr.csv` | features originais + as 4 colunas `repr_*` |
| `results/feature_representation_report.json` | o que foi aplicado e as colunas criadas |
| `results/01_as_quatro_features.png` / `results/02_mapa_correlacao.png` | as duas figuras |

Além disso, o pipeline grava os arquivos completos em
`../../multiwavelab_outputs/output_ieb1/features/` (`features0_repr.csv`,
`features0_train_repr.csv`, `features0_test_repr.csv`).

---

## Arquivos deste módulo

```
new_features/
├── README.md                       este documento
├── _bootstrap.py                   deixa a pasta src/ importável nos scripts
├── feature_representation.py       funções que criam as 4 colunas novas
├── run_feature_representation.py   lê features0.csv e grava features0_repr.csv
├── visualize_representation.py     gera as figuras e as tabelas de resultado
├── results/                        resultados prontos (CSVs e figuras)
└── config/
    └── feature_representation.json  exemplo de configuração
```
