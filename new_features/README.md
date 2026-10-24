# Nova representação a partir das features originais

**Tema 5 do Grupo 3:** *Feature extraction — constructing a new feature
representation from the original features.*

Este documento é a fonte da verdade da apresentação: explica o que o
MultiWaveLab já fazia, o que nós acrescentamos, os números que encontramos e
como reproduzir tudo.

---

## Resumo em 30 segundos

- O MultiWaveLab **já extrai** features do sinal de PPG (o sensor óptico de
  pulso): cada participante vira uma linha com **72 números**.
- Nós **não voltamos ao sinal**. A partir dessas 72 colunas, criamos **4
  colunas novas** fazendo contas entre elas (divisões que tiram a unidade e o
  efeito da frequência cardíaca).
- As 4 colunas novas são fáceis de explicar e duas delas se correlacionam
  com a glicose (*Ritmo do coração* e *Subida do pulso*).
- Três das quatro reproduzem valores que o pipeline já calculava por outro
  caminho, o que confirma que as contas estão certas. A *Largura do pulso* é
  a única que traz informação nova.
- Para prever a glicose, os modelos erram o mesmo ou um pouco menos com as
  4 colunas novas do que com as 72 originais. O melhor (SVM com as 4 novas)
  erra em média 53,8 mg/dL, contra 61,1 mg/dL de quem chuta sempre a média:
  uma melhora pequena, que não basta para estimar a glicose.

---

## Roteiro dos 3 slides (1 minuto cada)

| slide | mensagem | o que mostrar |
|---|---|---|
| **1. O que já existia** | O MultiWaveLab transforma o sinal de pulso de cada pessoa em 72 números. Nossa tarefa começa depois disso: criar uma representação nova **a partir desses 72 números**, sem voltar ao sinal. | o diagrama da seção 1 |
| **2. As 4 features novas** | Cada feature nova é uma divisão entre colunas originais. Dividir tira a unidade e o efeito da frequência cardíaca, e assim pessoas diferentes ficam comparáveis. | a tabela da seção 3.1, com o exemplo do participante hu_01 |
| **3. O que encontramos** | O Ritmo do coração e a Subida do pulso se correlacionam com a glicose. Três features confirmam valores que já existiam (validação), e a Largura do pulso é nova. Com 4 colunas em vez de 72, os modelos erram o mesmo ou um pouco menos (melhor caso: 53,8 mg/dL contra 61,1 de chutar a média), mas nenhum estima bem a glicose. | `results/01_as_quatro_features.png` ou `results/02_mapa_correlacao.png` |

---

## 1. O que o repositório original já faz

O MultiWaveLab é o projeto de mestrado da Sofia P. Klautau (IEB-UFSC). Ele
tenta estimar a **glicose no sangue** a partir de sinais fisiológicos não
invasivos. No dataset IEB-1, que usamos aqui, o sinal é só o **PPG** (a onda
de pulso medida por um sensor de luz na pele), gravado a 60 amostras por
segundo, com uma medição de glicose por participante.

O pipeline original faz tudo, do sinal ao modelo:

```
sinal PPG bruto
   │  src/signal_processing/ppg.py           filtra o sinal e acha as batidas
   ▼
segmentos bons do sinal
   │  src/segments/create_segment_files.py   recorta os trechos de boa qualidade
   ▼
72 features por segmento    ◄── EXTRAÇÃO (já existia)
   │  src/features/extract_features_for_all_modalities.py
   │  src/features/fusion_of_multimodality_features.py
   ▼
features0.csv  (92 participantes × 72 features + glicose)
   │                                         ┌──────────────────────────────┐
   ├────────────────────────────────────────►│ NOSSA PARTE: new_features/   │
   │                                         │ 72 colunas → +4 colunas novas│
   │                                         └──────────────────────────────┘
   │  src/machinelearning/train_test_splits.py   separa treino e teste
   │  src/features/feature_selection.py          escolhe as melhores features
   ▼  src/machinelearning/regression_evaluation.py  treina modelos e mede o erro
estimativa de glicose
```

| | repositório original | o que nós acrescentamos |
|---|---|---|
| **Entrada** | sinal PPG bruto | a tabela de 72 features que o original produz |
| **Saída** | 72 features por participante, modelos de regressão | 4 colunas novas (`repr_*`), figuras e tabelas de análise |
| **Onde fica** | `src/` | `new_features/` (o código original não foi alterado) |
| **Como liga** | grava `features0.csv` | lê `features0.csv` e grava `features0_repr.csv`, que o resto do pipeline pode usar no lugar |

### 1.1 Como a extração funciona no código

Para cada segmento, o código carrega o sinal, recorta o trecho e chama o
extrator da modalidade
(**`src/features/extract_features_for_all_modalities.py:177`**):

```python
def extract_features_for_segment(row, fs_dict, datasetConfig):
    signal = load_signal(path)                            # lê o sinal do arquivo
    signal = signal[start_sample:start_sample+duration]   # recorta o segmento

    if modality == "ppg":
        features = extract_ppg_features(datasetConfig, signal, fs_dict["ppg"])
    return features
```

O extrator de PPG (**`src/signal_processing/ppg.py:1035`**) usa a biblioteca
NeuroKit2 para achar as batidas e calcula as features em cinco grupos:

```python
signals, info = nk.ppg_process(ppg, sampling_rate=fs)   # acha os picos
peaks = np.sort(info["PPG_Peaks"])

features1_peaks    = extract_features_group1_peaks(...)     # p1: frequência, amplitude
features2_beats    = extract_features_group2_beats(...)     # b2: forma de cada batida
features3_spectral = extract_features_group3_spectral(...)  # s3: espectro de frequências
features4_hrv      = extract_features_group4_prv(...)       # h4: variabilidade do ritmo
features6_imf      = extract_features_group6_imf(...)       # i6: decomposição do sinal (EMD)
```

Cada grupo vira colunas com um prefixo:

| grupo | exemplo de coluna | o que mede |
|---|---|---|
| `p1` | `ppg_p1_hr_mean` | frequência cardíaca média (batimentos por minuto) |
| `b2` | `ppg_b2_T_median` | duração típica de um batimento (s) |
| `b2` | `ppg_b2_width_median` | tempo em que a onda fica acima da metade da altura (s) |
| `b2` | `ppg_b2_Tr_median` | tempo do início do batimento até o pico (s) |
| `s3` | `ppg_s3_MAX_POWER_1st` | força da frequência principal do sinal |
| `h4` | `ppg_h4_PRV_HRV_RMSSD` | variação entre batimentos seguidos |
| `i6` | `ppg_i6_IMF_1_dom_freq` | frequência dominante da 1ª componente do sinal |

Depois, o projeto limpa e preenche valores faltantes
(`src/features/clean_features_dataframe.py`) e grava o CSV. O resultado é
uma linha por segmento com 72 features e metadados (participante, sessão,
qualidade do sinal e a glicose, na coluna `GLC`). No IEB-1 há **um segmento
por participante**, ou seja, 92 linhas.

---

## 2. A regra do nosso trabalho

> **Usar somente as 72 colunas originais. Não voltar ao sinal.**

"Construir uma nova representação" aqui quer dizer criar colunas novas que
são **contas entre colunas que já existem**. Escolhemos **divisões**, porque
dividir duas grandezas da mesma unidade gera um número sem unidade, que
permite comparar pessoas e gravações diferentes.

As colunas novas recebem o prefixo `repr_`, ficam ao lado das 72 originais e
nunca sobrescrevem metadados (`participant_id`, `GLC` etc.).

---

## 3. As quatro features novas

### 3.1 Um exemplo real: participante `hu_01` (glicose = 181 mg/dL)

Valores originais usados:

| coluna original | valor | o que é |
|---|---|---|
| `ppg_b2_N_BEATS` | 353 | número de batimentos no segmento |
| `duration` | 18 271 amostras | tamanho do segmento (18 271 ÷ 60 = 304,5 s) |
| `ppg_b2_T_median` | 0,867 s | duração típica de um batimento |
| `ppg_b2_width_median` | 0,283 s | tempo com a onda acima da metade da altura |
| `ppg_b2_Tr_median` | 0,250 s | tempo do início até o pico |
| `ppg_p1_peak_interval_std` | 0,0151 s | desvio-padrão do intervalo entre batimentos |
| `ppg_p1_hr_mean` | 69,76 bpm | frequência cardíaca média |

As quatro features novas:

| # | nome | o que significa | conta | resultado |
|---|---|---|---|---|
| 1 | **Ritmo do coração** | batimentos por segundo | 353 ÷ (18 271 ÷ 60) | 1,159 (= 69,5 por minuto) |
| 2 | **Largura do pulso** | fração do batimento com a onda acima da metade da altura | 0,283 ÷ 0,867 | 0,327 |
| 3 | **Subida do pulso** | fração do batimento até a onda chegar ao pico | 0,250 ÷ 0,867 | 0,288 |
| 4 | **Irregularidade do ritmo** | quanto o intervalo entre batimentos varia, em relação ao intervalo médio | 0,0151 ÷ (60 ÷ 69,76) | 0,0175 (1,75%) |

Nome das colunas no CSV: `repr_ppg_beat_rate_hz`,
`repr_ppg_b2_width_over_period`, `repr_ppg_b2_crest_time_over_period` e
`repr_ppg_p1_rr_cv`.

### 3.2 Por que essas quatro

| feature | o que ela captura | por que a divisão ajuda |
|---|---|---|
| **Ritmo do coração** | frequência | o número de batimentos sozinho depende do tamanho da gravação; dividindo pela duração, vira uma taxa comparável |
| **Largura do pulso** | forma da onda | em segundos, a largura cresce só porque o coração bate mais devagar; como fração do batimento, mede a forma, e não o ritmo |
| **Subida do pulso** | forma da onda | idem; o tempo de subida até o pico é associado na literatura à rigidez das artérias |
| **Irregularidade do ritmo** | variabilidade | o desvio-padrão sozinho é maior em quem tem batimentos mais lentos; dividindo pelo intervalo médio (coeficiente de variação), fica comparável |

Juntas, elas cobrem **frequência, forma (duas) e variabilidade**.

### 3.3 Números nos 92 participantes

| feature | média | desvio | mín | máx | correlação com a glicose (Spearman) |
|---|---|---|---|---|---|
| **Ritmo do coração** | 1,195 | 0,195 | 0,779 | 1,734 | +0,33 (p = 0,001) |
| **Largura do pulso** | 0,379 | 0,052 | 0,295 | 0,529 | −0,19 (p = 0,070, não significativa) |
| **Subida do pulso** | 0,314 | 0,040 | 0,243 | 0,482 | +0,23 (p = 0,027) |
| **Irregularidade do ritmo** | 0,084 | 0,064 | 0,018 | 0,328 | −0,17 (p = 0,103, não significativa) |

Como ler: a correlação de Spearman vai de −1 a +1 (0 = nenhuma relação). Um
valor de +0,33 é uma relação **fraca a moderada**: pessoas com ritmo mais
rápido tendem a ter glicose mais alta, mas com muita dispersão (veja a
figura `01_as_quatro_features.png`). Com 92 pessoas, |ρ| precisa passar de
~0,21 para ser significativo (p < 0,05).

---

## 4. O que encontramos

### 4.1 As features novas são novas mesmo?

Para cada feature nova, procuramos a coluna original que mais se parece com
ela (correlação de Spearman entre as duas; ±1 = praticamente a mesma
informação):

| feature nova | coluna original mais parecida | ρ | leitura |
|---|---|---|---|
| **Irregularidade do ritmo** | `ppg_h4_PRV_HRV_CVNN` | +1,00 | é a mesma medida, calculada por outro caminho |
| **Ritmo do coração** | `ppg_h4_PRV_HRV_MeanNN` | −0,999 | é o inverso do intervalo médio entre batimentos |
| **Subida do pulso** | `ppg_b2_Tr_norm_median` | +0,96 | quase a mesma coisa (o original divide batimento a batimento e tira a mediana; nós dividimos as medianas) |
| **Largura do pulso** | `ppg_s3_E2_norm` | −0,49 | **informação nova**: nenhuma coluna original se parece muito com ela |

Três das quatro reproduzem valores que o pipeline já calculava (as duas
primeiras vêm do NeuroKit2, a terceira do próprio `ppg.py`). Isso não é um
defeito: **confirma que nossas contas estão certas**, porque chegamos ao
mesmo resultado partindo de outras colunas.

Também vale notar o que a divisão faz com a correlação: a largura em
segundos tem ρ = −0,46 com a glicose, e a Largura do pulso cai para −0,19.
Ou seja, boa parte da relação da largura com a glicose vinha da frequência
cardíaca, e a divisão separa os dois efeitos.

### 4.2 Elas ajudam a prever a glicose?

**Como medimos.** Usamos o mesmo método de avaliação do repositório original
(`src/machinelearning/regression_evaluation.py`), com os modelos vistos na
disciplina: **regressão linear, KNN, árvore de decisão** (a de profundidade 1
é o *decision stump*) e **SVM**. O passo a passo:

1. Os 92 participantes são divididos em 5 partes. O modelo treina com 4
   partes e é testado na 5ª, que ele nunca viu.
2. Dentro das 4 partes de treino, uma segunda divisão (em 3) escolhe os
   ajustes de cada modelo, como o número de vizinhos do KNN ou a
   profundidade da árvore. Assim, a parte de teste não influencia nenhuma
   escolha.
3. As colunas são padronizadas (média 0, desvio 1) antes de entrar no
   modelo, como no original.
4. A divisão em 5 partes é repetida 20 vezes com sorteios diferentes (100
   testes no total), e todos os conjuntos de features usam os mesmos
   sorteios. A diferença para o original é só essa repetição: o original faz
   a divisão uma vez e guarda 6 participantes para um teste final, o que é
   pouco para comparar conjuntos de features.

A medida é o **erro médio absoluto (MAE)**: em média, quantos mg/dL a
previsão erra. A referência é **chutar sempre a média** da glicose do treino,
que erra **61,1 mg/dL**. Para comparar, a glicose dos participantes vai de 65
a 400 mg/dL (mediana 122).

| features usadas | regressão linear | KNN | árvore de decisão | SVM |
|---|---|---|---|---|
| 72 originais | 528,6 | 58,8 | 66,1 | 56,9 |
| 72 originais + 4 novas | 1442,9 | 58,7 | 65,9 | 56,5 |
| só as 6 colunas originais que entram nas contas | 59,1 | 59,5 | 61,1 | 55,6 |
| **só as 4 novas** | **57,0** | 58,8 | 62,1 | **53,8** |

Erro em mg/dL; quanto menor, melhor. Chutar a média erra 61,1.

O que isso mostra:

- **A regressão linear não funciona com as 72 colunas.** Cada parte de
  treino tem 73 ou 74 pessoas e o modelo recebe 72 colunas, ou seja, quase uma
  coluna por pessoa. Ele decora o treino e erra 529 mg/dL em média no teste.
  Com as 4 colunas novas, o mesmo modelo erra 57,0 mg/dL, 4,1 a menos que
  chutar a média.
- **O melhor resultado é o SVM com as 4 novas:** erra 53,8 mg/dL, 7,4 a
  menos que chutar a média, e ganha da média em 84 dos 100 testes. Com as 72
  originais, o SVM erra 56,9.
- **Somar as 4 novas às 72 não muda nada** nos modelos que funcionam (KNN,
  árvore e SVM variam no máximo 0,4 mg/dL).
- **Boa parte do ganho vem de usar poucas colunas.** As 6 colunas originais
  que entram nas contas, sem fazer as divisões, ficam perto das 4 novas (no
  SVM, 55,6 contra 53,8). A diferença de 1,8 mg/dL é pequena perto da
  variação de um teste para outro (~11 a 15 mg/dL).
- **A árvore de decisão é o pior modelo** e, com as 72 colunas, erra mais
  que chutar a média (66,1).

**Conclusão:** com 4 colunas em vez de 72, os modelos erram o mesmo ou um
pouco menos (a regressão linear só funciona com as 4). Mesmo o melhor
modelo ainda erra cerca de 54 mg/dL, apenas 7,4 mg/dL a menos que chutar a
média: com 92 participantes, nenhuma combinação estima bem a glicose.

---

## 5. Como rodar

### 5.1 Preparar o ambiente (uma vez)

O projeto precisa de **Python 3.10** (com o 3.14, o `numpy==2.2.6` não
instala). Usamos o `uv` instalado **dentro do projeto**, sem nada global. Da
raiz do repositório:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.uv/bin" UV_NO_MODIFY_PATH=1 sh
UV_PYTHON_INSTALL_DIR="$PWD/.uv/python" .uv/bin/uv venv --seed --python 3.10.13 .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyPPG --no-deps
```

Para desfazer tudo: `rm -rf .uv .venv`.

### 5.2 Gerar as features originais (só se ainda não existirem)

A partir da pasta `src`, com o `.venv` ativo:

```bash
cd src
PYTHONPATH=. python signal_processing/ppg.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python segments/create_segment_files.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python features/extract_features_for_all_modalities.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python features/fusion_of_multimodality_features.py ../config_experiments/input_ieb1/exp0.json
PYTHONPATH=. python machinelearning/train_test_splits.py ../config_experiments/input_ieb1/exp0.json
```

Isso grava `features0.csv`, `features0_train.csv` e `features0_test.csv` em
`../../multiwavelab_outputs/output_ieb1/features/`. (O
`src/run_scripts/run_all_pipelines.py` faz a mesma sequência, mas usa
caminhos no formato do Windows.)

### 5.3 Construir as quatro features novas

Ainda em `src`:

```bash
# arquivo com todos os participantes
PYTHONPATH=. python ../new_features/run_feature_representation.py \
    ../config_experiments/input_ieb1/exp0.json --repr ratio

# divisões de treino/teste
PYTHONPATH=. python ../new_features/run_feature_representation.py \
    ../config_experiments/input_ieb1/exp0.json --source train_test --repr ratio
```

| opção | efeito |
|---|---|
| `--repr ratio` | a representação com as 4 features (é a única) |
| `--spec arquivo.json` | lê a configuração de um arquivo (exemplo em `config/feature_representation.json`) |
| `--source fused\|train_test` | aplica no arquivo completo ou nas divisões de treino/teste |
| `--no-original` | grava só as colunas novas e os metadados, sem as 72 originais |
| `--describe` | só descreve as 72 features originais, sem gerar nada |

### 5.4 Gerar as figuras e tabelas

```bash
PYTHONPATH=. python ../new_features/visualize_representation.py \
    ../config_experiments/input_ieb1/exp0.json
```

As figuras são copiadas para `new_features/results/`:

| figura | o que mostra |
|---|---|
| `01_as_quatro_features.png` | cada feature nova contra a glicose (um ponto por pessoa), com a correlação e a reta de tendência |
| `02_mapa_correlacao.png` | mapa de cores com a correlação entre as 4 features e a glicose |

### 5.5 Reproduzir a seção 4

Da raiz do repositório (usa o `results/features0_repr.csv` já pronto):

```bash
python new_features/evaluate_representation.py
```

Leva de 1 a 2 minutos e grava `results/redundancia.csv` (seção 4.1) e
`results/avaliacao_modelos.csv` (seção 4.2).

### 5.6 Arquivos de resultado

| arquivo | conteúdo |
|---|---|
| `results/features0_repr.csv` | as 72 features originais + as 4 colunas novas (92 linhas) |
| `results/dados_por_participante.csv` | participante, glicose e as 4 features novas |
| `results/resumo_estatistico.csv` | média, desvio, mín, máx e correlação com a glicose (seção 3.3) |
| `results/redundancia.csv` | coluna original mais parecida com cada feature nova (seção 4.1) |
| `results/avaliacao_modelos.csv` | erro dos modelos com cada conjunto de features (seção 4.2) |
| `results/feature_representation_report.json` | o que foi aplicado e quais colunas foram criadas |
| `results/01_as_quatro_features.png`, `results/02_mapa_correlacao.png` | as duas figuras |

---

## Arquivos deste módulo

```
new_features/
├── README.md                       este documento
├── _bootstrap.py                   deixa a pasta src/ importável nos scripts
├── feature_representation.py       as regras que criam as 4 colunas novas
├── run_feature_representation.py   lê features0.csv e grava features0_repr.csv
├── visualize_representation.py     gera as figuras e as tabelas da seção 3
├── evaluate_representation.py      gera as tabelas da seção 4
├── results/                        resultados prontos (CSVs e figuras)
└── config/
    └── feature_representation.json  exemplo de configuração
```
