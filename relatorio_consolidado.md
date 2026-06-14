# Relatório Consolidado do Projeto — Guia para o Relatório Final
## Prevenção de Churn em Telecomunicações via Inferência Causal e *Unit Selection*

> **Propósito deste documento.** Servir de *blueprint* para uma LLM redigir o **relatório final**.
> Mantém a estrutura de seções do `Relatório_intermediário_Causal.pdf` (§1–§10) e **adiciona** as
> seções da entrega final (CATE, Refutação, Unit Selection, ROI, Qini/AUUC, Multi-tratamento).
> Cada seção indica: **[estágio do framework]**, o que escrever, os **resultados/números reais**
> (extraídos do `tpcausal_Lucas.ipynb` já corrigido e reexecutado) e **notas de atenção** (o que mudou
> em relação ao intermediário — para **não** repetir números antigos/errados).
>
> **Framework de inferência causal (Pearl / DoWhy):** *Query → Model → Identify → Estimate → Refute*,
> acrescido de **Unit Selection** (Nível 3 de Pearl) e **otimização de política**.

---

## 0. Mapa do Framework → Seções

| Estágio | Pergunta | Seções deste guia |
|---|---|---|
| **QUERY** | O que queremos saber? | §1 Problema · §2 Formulação · §5 Perguntas de Pesquisa · §7.0 Nível de Pearl |
| **MODEL** | Qual a estrutura causal? | §6 EDA + Prova de Confundimento · §7 DAG (domínio + descoberta PC/GES/FCI) |
| **IDENTIFY** | O efeito é estimável? | §10 Identificação por *backdoor* (DoWhy) |
| **ESTIMATE** | Qual a magnitude? | §11 ATE (IPW/DoWhy) · §12 CATE (meta-learners) · §14 Unit Selection · §15 ROI · §17 Multi-tratamento |
| **REFUTE** | É robusto? | §13 Refutação + sensibilidade · §16 Qini/AUUC (avaliação causal) |

**Tese central do projeto (frase-resumo para abertura e conclusão):** modelos preditivos dizem *quem*
vai cancelar, mas não *o que fazer*. Tratando churn como **estimação de efeito de tratamento** +
**Unit Selection**, mostramos que (i) o efeito observacional do TechSupport é inflado pelo viés de
seleção, (ii) o efeito causal real é heterogêneo entre clientes e (iii) **direcionar o incentivo pelos
*Compliers* (uplift causal) gera lucro, enquanto a heurística de "tratar quem tem alto risco" destrói valor.**

---

## 0.1 Checklist de validação (re-feito do zero — estado ATUAL do notebook)

Cruzando os 6 pilares de um pipeline causal rigoroso contra o `tpcausal_Lucas.ipynb` **atual**
(104 células, reexecutado sem erros):

| # | Pilar | Estado atual |
|---|---|---|
| 1 | **Modelagem causal e DAG** (C/X/Y separados; DAG + backdoor via DoWhy) | ✅ Completo. Ajuste = `{InternetService, tenure, Contract}`. |
| 2 | **Premissas (Overlap/Positividade)** (propensity plot; tratamento de violação) | ✅ Overlap plotado; PS escalados; *clipping* [0,05; 0,95]. |
| 3 | **CATE** (meta-learners EconML; efeito individual) | ✅ T/X-Learner calibrados, train/test, sem vazamento. |
| 4 | **Unit Selection** (função de benefício/ROI; estratos contrafactuais) | ✅ Estratos + benefício; 🟡 *bounds* formais de Li & Pearl pendentes (aproximação por uplift, monotonicidade empírica). |
| 5 | **Métricas causais** (Qini/AUUC; causal vs. ML preditivo) | ✅ Qini/AUUC por IPW; ROI causal vs. heurística. |
| 6 | **Refutação e robustez** (placebo; confundidor não-observado) | ✅ Placebo + Random Common Cause + Data Subset + **varredura** de sensibilidade. |

**Contraste com `revisao_tecnica_gap_analysis.md`:** aquele documento é o **registro da jornada**
(diagnóstico → correção dos Gaps #1–#6 em 3 rodadas). **Este** documento é a **fotografia final
coerente** do projeto, já com tudo corrigido, organizada para virar texto de relatório. Em caso de
divergência numérica com versões antigas (inclusive a Tabela 1 do intermediário), **valem os números
deste guia** (= saída atual do notebook).

> ⚠️ **Erros do intermediário que NÃO devem ser reproduzidos no final:** (a) Tabela 1 com PR-AUC
> 0,84/0,82 e "LogReg melhor calibrado" — **errado**; ver §8. (b) Qualquer ganho de ROI na casa de
> R$ 900 mil — era artefato de avaliação circular; ver §15. (c) Texto afirmando robustez total a
> confundidor — a sensibilidade real tem limiar ~0,1; ver §13.

---

# PARTE I — Seções herdadas do intermediário (atualizar números, manter narrativa)

## 1. Definição do Problema  *(QUERY)*
Manter o texto do intermediário. Pontos-chave:
- CAC ≫ custo de retenção; churn é crítico. Modelos preditivos **não prescrevem ação**.
- **Unit Selection (Li & Pearl):** classificar a população em 4 estratos contrafactuais —
  **Always-takers** (ficam de qualquer jeito), **Never-takers** (saem de qualquer jeito),
  **Defiers** (saem *por causa* do incentivo), **Compliers** (só ficam *se* tratados).
- Objetivo: oferecer o incentivo **apenas onde ele é a causa da permanência** (Compliers) → maximiza ROI.

## 2. Formulação do Problema de ML  *(QUERY)*
Manter. Variáveis:
- **C (covariáveis):** demografia (`gender`, `SeniorCitizen`), conta (`tenure`, `Contract`,
  `MonthlyCharges`, `TotalCharges`), serviços (`InternetService`, `OnlineSecurity`, …).
- **X (tratamentos):** `TechSupport`, `StreamingTV`, `StreamingMovies`.
- **Y (desfecho):** **Retenção** = inversão de `Churn` (Y=1 ficou, Y=0 cancelou).
- Alvo formal: $CATE_k(c)=\mathbb{E}[Y\mid do(X=x_k),C=c]-\mathbb{E}[Y\mid do(X=0),C=c]$ → política.

## 3. Paradigma de ML  *(QUERY)*
Manter. Aprendizado supervisionado **como motor** da inferência causal: estimadores-base (regressão/
árvores) dentro de **meta-learners** (T-/X-Learner) para prever desfechos contrafactuais e ajustar o
viés de seleção dos dados observacionais.

## 4. Datasets  *(MODEL)*
Manter. **Telco Customer Churn (IBM/Kaggle)**: **7.043 clientes × 21 atributos**, observacional (sem RCT).
- Desbalanceamento: **5.174 ficaram (73,5%) / 1.869 cancelaram (26,5%)**.
- Limitações: desbalanceamento; **viés de seleção** (premissa de desconfundimento); `TotalCharges`
  com 11 vazios (clientes `tenure=0`) → imputados com 0.
- Para a análise binária de cada tratamento, descarta-se "No internet service" → **5.517 clientes**
  (TechSupport: 2.044 com, 3.473 sem).

## 5. Perguntas de Pesquisa  *(QUERY)*
Manter as três; **agora todas têm resposta** (citar nas seções indicadas):
- **RQ1 — Efeito causal multi-tratamento:** TechSupport vs. StreamingTV vs. StreamingMovies. → **§17**.
- **RQ2 — Otimização de política (Unit Selection / ROI):** ganho de focar nos Compliers vs. heurística de risco. → **§15**.
- **RQ3 — Heterogeneidade e robustez:** papel de `tenure`/`Contract` e robustez ao desbalanceamento. → **§6, §12, §13**.

### 7.0 Nível na hierarquia de Pearl  *(QUERY)*
Manter: **Nível 3 (Contrafactual)** apoiado no Nível 2 (Intervenção). A classificação em estratos
(Compliers etc.) é raciocínio contrafactual cruzado ($Y_{x=1}, Y_{x=0}$). **Nota honesta:** a entrega
implementa a aproximação por *uplift* sob **monotonicidade** (empiricamente válida — Defiers ≈ 0,9%);
os **bounds** formais de Li & Pearl ficam como extensão (ver §14 e §18).

## 6. Análise Exploratória e Prova de Confundimento  *(MODEL)*
Manter a narrativa; **atualizar os números do viés de seleção** (a Fig. 4 do intermediário trocou os
rótulos dos grupos). Pontos:
- `gender` ~ sem efeito; `SeniorCitizen`, sem parceiro/dependentes → mais churn.
- Contratos *month-to-month* e *Fiber optic* → picos de cancelamento.
- **Correlações com Y (Pearson):** `Contract` 0,40 · `tenure` 0,35 · `OnlineSecurity` 0,29 ·
  `TechSupport` 0,28 · `TotalCharges` 0,20 · `MonthlyCharges` −0,19.
- **Prova do viés de seleção (a "ilusão observacional"):** quem tem TechSupport retém **84,8%** vs.
  **58,4%** sem (≈ **+26,4 p.p. ingênuos**). Mas os tratados já têm **tenure médio 44,8 meses vs. 25,8**
  (≈ **19 meses a mais**) → caminho de *backdoor* `X ← tenure → Y`. A associação mistura efeito + lealdade prévia.
  > ⚠️ Use **44,8 vs 25,8** (Yes vs No, com internet). Evite a confusão da Fig. 4 do intermediário.

## 7. Grafo Causal: do Domínio à Descoberta  *(MODEL)*
**Manter §7 do intermediário (DAG de domínio)** e **adicionar a descoberta causal** (era "próximos passos"):
- **DAG de domínio:** `C → X`, `C → Y`, `X → Y`; premissa de **Unconfoundedness**; *backdoor* `X ← C → Y`.
- **Descoberta causal (nova):** *Background knowledge* por **tiers temporais** (95 arestas proibidas)
  + **8 arestas requeridas** (`tenure`,`Contract` → cada tratamento e → `Y`). Algoritmos sobre dados
  **discretizados**: **PC (qui-quadrado)**, **GES (BDeu)**, **FCI (qui-quadrado)**.
  - **Consenso** (arestas em ≥2 algoritmos) + **scoring BIC/BDeu** (`pgmpy`) para escolher o DAG.
  - DAG vencedor por BIC: **GES** (≈ Consenso; ambos próximos). **Nota de reprodutibilidade:** a FCI
    tem leve não-determinismo, então o "vencedor" pode oscilar entre GES/Consenso — **mas o conjunto de
    ajuste resultante é estável**: `{InternetService, tenure, Contract}` (graças às arestas requeridas).
  - **FCI (qui-quadrado) NÃO acusa `TechSupport ↔ Y`** como confundidor latente; as bidirecionadas
    restantes são só entre covariáveis (ex. `tenure ↔ Contract`) — reforça a *unconfoundedness*.
  > ⚠️ Metodológico: usar **qui-quadrado/BDeu** (dados categóricos), não Fisher-Z. Foi o que corrigiu o
  > falso alerta `TechSupport↔Y` que aparecia com Fisher-Z.

## 8. Modelos Preditivos de Base (Baselines)  *(suporte ao ESTIMATE)*
Manter a ideia; **substituir a Tabela 1 pelos números corretos** e a conclusão de base learner.

| Modelo | PR-AUC (classe minoritária) | ROC-AUC | Brier Score |
|---|---|---|---|
| Regressão Logística | 0,610 | 0,832 | 0,1415 |
| **Random Forest** | **0,642** | **0,835** | **0,1388** |

- PR-AUC priorizada (desbalanceamento); Brier para **calibração** (erros em $P(Y|X,C)$ propagam ao CATE).
- **O RF é marginalmente melhor nas três métricas** → é o **base learner** do estágio causal,
  **calibrado** via `CalibratedClassifierCV` (isotônica).
  > ⚠️ Corrigir o intermediário: ele dizia PR-AUC 0,84/0,82 e "LogReg melhor calibrado" — **incorreto**.
  > Os ~0,83 eram, na prática, ROC-AUC; e o RF tem Brier menor. Texto, tabela e código agora batem.

## 9. Validação da Premissa de Positividade (Overlap)  *(MODEL/IDENTIFY)*
Manter. Modelo logístico (features **escaladas**) prevê $e(C)=P(\text{TechSupport}=1\mid C)$; KDE dos
*propensity scores* de tratados vs. controle mostra **boa sobreposição** → positividade plausível;
para cada tratado há um "gêmeo" no controle. Habilita o ajuste de *backdoor*/Unit Selection.

## 10. Identificação Causal (Backdoor via DoWhy)  *(IDENTIFY)*
Seção (parcialmente nova) — formalizar a identificação:
- `CausalModel` (DoWhy) recebe o DAG refinado (GML). `identify_effect()` retorna estimando **backdoor**:
  $\frac{d}{d[\text{TechSupport}]}\,\mathbb{E}[Y\mid \text{InternetService}, \text{tenure}, \text{Contract}]$.
- **Conjunto de ajuste = {InternetService, tenure, Contract}** — inclui os confundidores mais fortes (§6).
- `frontdoor` inexistente; `iv` candidato (instrumento sugerido) reportado mas **não usado**.
- Premissa explícita: *Unconfoundedness* condicional ao conjunto de ajuste.

---

# PARTE II — Seções novas (entrega final)

## 11. Estimação do Efeito Médio (ATE)  *(ESTIMATE — agregado)*
Manter o §10 do intermediário (IPW) e **adicionar a estimativa do DoWhy**:
- **IPW manual:** efeito **ingênuo +26,47 p.p.** → **ATE +10,72 p.p.** (retenção contrafactual
  **76,66%** tratado vs **65,94%** controle). **Viés removido: 15,75 p.p.** (o ML tradicional
  superestimava o TechSupport em mais do dobro).
- **DoWhy (Propensity Score Weighting):** **ATE = 0,086** (mesma ordem do IPW; conjunto de ajuste reforçado).
- Mensagem: o ajuste causal **confirma** o benefício do TechSupport, mas o "encolhe" para um patamar honesto.

## 12. Efeito Heterogêneo (CATE) via Meta-Learners  *(ESTIMATE — individual)*
Nova. Do efeito médio ao **efeito individual** $\hat\tau(x)$ — base do Unit Selection.
- **T-Learner** e **X-Learner** (EconML), base learner RF, **ajustados no treino**, avaliados no **teste**.
- Integração **DoWhy + EconML** (respeita o estimando do backdoor).
- Resultados (teste, out-of-sample): **T-Learner ATE +0,111** (faixa [−0,30; +0,48]);
  **X-Learner ATE +0,077**, **std 0,080** → **heterogeneidade real**. **83% dos clientes com $\hat\tau>0$.**
- Consistência: alinhado a IPW (+0,107) e DoWhy PSW (+0,086).
- **Conexão RQ3:** o efeito **varia** com o perfil (tenure/contrato) → justifica personalizar a decisão.
  > ⚠️ Mencione (boa prática) que a estimação usa **covariáveis limpas** (sem artefatos), **split
  > treino/teste** e **calibração** — pontos que diferenciam um CATE confiável de um enviesado.

## 13. Refutação e Análise de Sensibilidade  *(REFUTE)*
Nova/expandida. Bateria de robustez sobre o ATE (DoWhy):
- **Placebo Treatment** (permuta o tratamento): efeito colapsa **0,086 → −0,25** (p=0). ✅
- **Random Common Cause** (adiciona covariável aleatória): efeito **inalterado** (0,0862, p=1,0). ✅
- **Data Subset** (80% da amostra): estável (**0,087**). ✅
- **Confundidor não-observado — varredura** (vários níveis de força, não um ponto): o ATE é robusto a
  confundimento **fraco**, mas **seria anulado por um confundidor moderado (força ~0,1)**.
- **FCI (qui-quadrado)** não acusa `TechSupport ↔ Y`.
- **Conclusão honesta (escrever assim):** as evidências sustentam o efeito, mas *unconfoundedness*
  é uma **suposição**; dado o efeito modesto após ajustar tenure/Contract, um confundidor latente
  **moderado** ainda poderia explicá-lo — limitação a declarar, não a esconder. *(Atende RQ3.)*

## 14. Unit Selection — Estratos Causais  *(ESTIMATE → decisão; Nível 3)*
Nova. De probabilidades contrafactuais à classificação (Li & Pearl, sob monotonicidade):
- Dois classificadores calibrados estimam $y_1=P(Y{=}1\mid do(X{=}1))$ e $y_0=P(Y{=}1\mid do(X{=}0))$.
- Estratos: Complier $=\max(0,y_1-y_0)$, Always $=\min(y_0,y_1)$, Never $=1-\max(y_0,y_1)$, Defier $=\max(0,y_0-y_1)$.
- **Médias:** Complier **0,081** · Always **0,654** · Never **0,256** · **Defier 0,009 ≈ 0**
  → **monotonicidade empiricamente válida** (legitima a aproximação).
- **Função de benefício** (pesos de negócio): $v=1000$ (lucro retido), $c=50$ (custo da ação):
  $B = P_{compl}(v-c) - c\,(P_{alw}+P_{nev}) - (v+c)P_{def}$. Regra: **tratar se $B>0$** (limiar de uplift $c/v=0,05$).
  → **52,1%** da base selecionada.
  > ⚠️ Honestidade de Nível: é **aproximação por uplift sob monotonicidade**, não os *bounds* de
  > Li & Pearl. Apresente assim e cite os bounds como trabalho futuro.

## 15. Otimização de Política e ROI — Causal vs. Heurística  *(ESTIMATE → política; RQ2)*
Nova. Prova de valor com **avaliação honesta** (não circular):
- **Avaliação:** *policy value* por **IPW sobre desfechos observados do teste** (independente do modelo
  que ranqueia) — corrige a circularidade que inflava o ganho.
- **Política causal** (ordena por uplift) ótima: trata **589** clientes → **+R$ 206.955**.
- **Heurística de churn (>70% risco):** trata **125** → **−R$ 33.383** (destrói valor!).
- **Ganho da seleção causal: +R$ 240.338.**
- **Leitura (RQ2):** tratar por **risco** ≠ tratar quem **responde**; a heurística da indústria pode
  dar prejuízo porque mira Always/Never-takers.
  > ⚠️ NÃO citar o antigo "~R$ 919 mil" (avaliação circular, descartado).

## 16. Métricas de Avaliação Causal (Qini / AUUC)  *(REFUTE/avaliação)*
Nova. Qualidade do **ranqueamento causal** (sem ground truth contrafactual real):
- Métrica por **IPW** sobre o *outcome transformado* $z=\frac{TY}{e}-\frac{(1-T)Y}{1-e}$, no teste.
- **AUUC:** Causal **188,5** > Aleatório **143,0** > Heurística **54,4** (oráculo 1118).
- **Qini coefficient:** Causal **+45,5** vs. Heurística **−88,7** (Qini normalizado 0,047 vs −0,091).
- **Leitura forte (RQ1):** o uplift causal **supera o aleatório**, enquanto a **heurística de risco
  fica ABAIXO do aleatório** para selecionar respondedores.
  > Nota: o Qini normalizado é baixo porque o "oráculo" IPW é um teto inflado pela variância do IPW;
  > o que importa é a ordenação causal > aleatório > heurística.

## 17. Multi-Tratamento — Comparação de Intervenções  *(ESTIMATE; RQ1)*
Nova. Generaliza CATE→Qini→ROI para os 3 tratamentos (mesmo split, avaliação IPW no teste):

| Tratamento | ATE_IPW | % $\hat\tau>0$ | AUUC | Qini | ROI ótimo (R$) |
|---|---|---|---|---|---|
| **TechSupport** | **+0,173** | 76,7% | **188,5** | **+45,5** | **206.955** |
| StreamingTV | −0,048 | 29,4% | −47,0 | −7,5 | 0 |
| StreamingMovies | −0,054 | 19,1% | −31,3 | +13,1 | 1.457 |

- **Resposta RQ1:** **somente o TechSupport** tem efeito causal positivo de retenção; Streaming não
  retém (efeito ~nulo/negativo). É o melhor em ATE, Qini e ROI.
- **Política personalizada (melhor incentivo por cliente):** TechSupport → **2.795**, Nenhum → **2.573**,
  StreamingTV → 99, StreamingMovies → 50 (53,4% tratados). Ganho esperado (modelo) da personalização
  **R$ 276.828** vs. melhor único (TechSupport) R$ 271.438 (**+R$ 5.390**).
  > ⚠️ Os valores por-tratamento (ATE/AUUC/Qini/ROI) são **IPW honestos**; o ganho da política
  > personalizada é **estimativa do modelo** (rotule assim).

## 18. Conclusão  *(síntese)*
Escrever fechando o ciclo **Q→M→I→E→R + Unit Selection**:
1. **Model/Identify:** descoberta (PC/GES/FCI) + background knowledge → ajuste estável `{tenure, Contract, InternetService}`.
2. **Estimate:** o viés de seleção infla o efeito (26,5 → 10,7 p.p.); o CATE revela **heterogeneidade real**.
3. **Refute:** placebo/random/subset passam; sensibilidade declara o limiar (~0,1) — honestidade.
4. **Unit Selection/ROI (RQ2):** focar nos Compliers **gera lucro** (+R$ 240k vs. heurística); risco ≠ resposta.
5. **Multi-tratamento (RQ1):** só TechSupport retém; política personalizada o confirma.
- **Limitações:** *unconfoundedness* (efeito frágil a confundidor moderado); aproximação por uplift
  (bounds de Li & Pearl como trabalho futuro = Nível 3 pleno); positividade validada principalmente
  para TechSupport.

## 19. Referências
- IBM Watson Analytics. *Telco Customer Churn dataset.* Kaggle, 2015.
- Ang Li, Judea Pearl. *Unit selection based on counterfactual logic.* IJCAI 2019, 1793–1799.
- Ang Li, Judea Pearl. *Unit selection with causal diagram.* AAAI 2022, 36:5765–5772.
- Amit Sharma, Emre Kiciman. *DoWhy: an end-to-end library for causal inference.* arXiv:2011.04216, 2020.
- (Software) **EconML** (meta-learners/CATE), **causal-learn** (PC/GES/FCI), **pgmpy** (scoring BIC/BDeu).

---

## Apêndice A — Tabela mestre de resultados (para citar no texto)

| Item | Valor | Onde |
|---|---|---|
| Clientes / atributos | 7.043 / 21 | §4 |
| Retenção vs. churn | 73,5% / 26,5% | §4 |
| Amostra binária (TechSupport) | 5.517 (2.044/3.473) | §10 |
| Viés: tenure (Yes vs No) | 44,8 vs 25,8 meses | §6 |
| Retenção bruta (Yes vs No) | 84,8% vs 58,4% (+26,4 p.p.) | §6 |
| Baseline PR-AUC / ROC-AUC / Brier (LR) | 0,610 / 0,832 / 0,1415 | §8 |
| Baseline PR-AUC / ROC-AUC / Brier (RF) | 0,642 / 0,835 / 0,1388 | §8 |
| ATE IPW (efeito ingênuo → causal) | +26,47 → **+10,72 p.p.** | §11 |
| ATE DoWhy (PSW) | **0,086** | §11 |
| Conjunto de ajuste (backdoor) | {InternetService, tenure, Contract} | §10 |
| CATE T-Learner / X-Learner (teste) | +0,111 / +0,077 (std 0,080) | §12 |
| Clientes com $\hat\tau>0$ | 83% | §12 |
| Placebo (efeito → novo) | 0,086 → −0,25 (p=0) | §13 |
| Sensibilidade: limiar de anulação | força ~0,1 | §13 |
| Estratos (Compl/Alw/Nev/Def) | 0,081 / 0,654 / 0,256 / **0,009** | §14 |
| Selecionados p/ tratar (TechSupport) | 52,1% | §14 |
| ROI causal vs. heurística (teste, IPW) | **+R$ 206.955 vs −R$ 33.383** (Δ +240.338) | §15 |
| AUUC / Qini (causal vs heurística) | 188,5/+45,5 vs 54,4/−88,7 | §16 |
| RQ1 — efeito por tratamento (ATE_IPW) | TechSupport +0,173 · TV −0,048 · Movies −0,054 | §17 |
| Política personalizada (ganho modelo) | R$ 276.828 (+5.390 vs único) | §17 |

## Apêndice B — Dicas de redação para a LLM
- **Tom:** rigoroso e **honesto** — declarar limitações é ponto forte, não fraqueza.
- **Para cada efeito causal, contraste sempre com o observacional** (a "ilusão observacional" é a espinha dorsal).
- **Não** reaproveitar números antigos do PDF intermediário sem checar nesta tabela mestre.
- Amarrar cada seção ao estágio Q→M→I→E→R (ver §0) e a uma RQ quando aplicável.
- Figuras a referenciar (já existem no notebook): distribuição do alvo; boxplot tenure×TechSupport;
  matriz de correlação; PR-curves; overlap de PS; DAGs (domínio, PC/GES/FCI, consenso, final);
  distribuição do CATE; curva de sensibilidade; curva de lucro acumulado; curva Qini; barras AUUC/Qini
  por tratamento; atribuição de tratamento por cliente.
