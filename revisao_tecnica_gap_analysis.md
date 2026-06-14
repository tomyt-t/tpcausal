# Revisão Técnica — Gap Analysis do Pipeline Causal

**Revisor:** Pesquisador sênior em Causal ML / Inferência Causal (Pearl L2–L3)
**Data:** 2026-06-14
**Artefatos cruzados:**
- `CML___TP.pdf` (proposta)
- `Relatório_intermediário_Causal.pdf` (progresso)
- `tpcausal_Lucas.ipynb` (implementação — 103 células, §1 a §17)

**Legenda de status:**
✅ Implementado e robusto · 🟡 Implementado, porém superficial / com ressalvas · 🔴 Implementado **com erro grave que invalida o resultado** · ❌ Ausente

---

## 0. Veredito em uma frase

O notebook tem **cobertura de escopo excelente** (todas as etapas do framework *Query→Model→Identify→Estimate→Refute* + Unit Selection + ROI existem em código). Os **gaps críticos #1–#6 foram corrigidos** (ver §3 e as notas de status abaixo); o **único item da proposta ainda pendente** é os **bounds formais de Li & Pearl** (opcional, para o Nível 3 pleno) — a Unit Selection hoje é uma aproximação por uplift sob monotonicidade já empiricamente válida (Defiers ≈ 0).

> **🔧 Status de correção — 2026-06-14 (rodada 1):** Gaps Críticos **#1 (vazamento), #2 (CATE degenerado) e #3 (ROI circular)** corrigidos; notebook reexecutado sem erros. T-Learner ATE(teste) **+0,111**, X-Learner **+0,077** (std **0,080**); Defiers **0,009 ≈ 0**.
>
> **🔧 Status de correção — 2026-06-14 (rodada 2):** adicionados **Qini/AUUC (§15.3)** e **multi-tratamento (§16, RQ1)**; aplicadas as melhorias de código **4.1, 4.2, 4.3, 4.4, 4.5 e 4.9**. Reexecução completa **sem erros (103 células)**. Destaques: o backdoor do DoWhy agora ajusta **{InternetService, Contract, tenure}** (4.3) e o ATE(DoWhy)=**0,086**; ROI honesto (§15.2) **+R$ 206.955** vs. heurística **−R$ 33.383**; **RQ1**: apenas **TechSupport** tem efeito positivo (ATE_IPW **+0,173**, Qini **+45,5**), enquanto StreamingTV (−0,048) e StreamingMovies (−0,054) não retêm.
>
> **🔧 Status de correção — 2026-06-14 (rodada 3):** resolvidos **Gap #6 / 5.6** (refutação robusta — Placebo + Random Common Cause + Data Subset + **varredura** de confundidor + discussão honesta da FCI) e **5.7** (tabela de baseline **autoritativa** com PR-AUC/ROC-AUC/Brier; texto/tabela/código consistentes). Reexecução **sem erros (104 células)**. A varredura mostra o ATE anulável por confundidor de força **~0,1** (fragilidade declarada). Os itens 🔴/🟡/❌ abaixo permanecem como histórico, com o respectivo *fix*.

| Etapa do notebook | Estado |
|---|---|
| §1–§8 Limpeza, EDA, baselines, viés, positividade, ATE/IPW | ✅ Sólido |
| §9–§11 Descoberta causal (PC/GES/FCI), scoring, DoWhy identify/estimate/refute | ✅ **Corrigido**: chisq/BDeu (4.1), scores comparáveis (4.2), refutação com varredura (#6) |
| §12 CATE (Meta-Learners) | ✅ **Corrigido** (#1+#2): covariáveis limpas, train/test, learners calibrados |
| §14 Unit Selection (estratos) | 🟡 Proxy de uplift (não os *bounds* de Li & Pearl), mas agora monotonicidade ≈ válida (Defiers 0,009) |
| §15 ROI Causal vs. Heurística | ✅ **Corrigido** (#3): valor de política via IPW em desfechos observados (teste) |
| Qini / AUUC | ✅ **Adicionado** (§15.3): Qini/AUUC corrigidos por IPW no teste |
| RQ1 multi-tratamento (TechSupport vs Streaming) | ✅ **Adicionado** (§16): 3 tratamentos comparados + política personalizada |
| Melhorias de código 4.1–4.5, 4.9 | ✅ Aplicadas (chisq/BDeu, scores comparáveis, tenure/Contract no backdoor, propensão escalada, `df.copy()`, título/warnings) |

---

## 1. Checklist de Validação (preenchido)

### Pilar 1 — Modelagem Causal e DAG
- ✅ **X, Y, C formalmente separados?** Sim. Célula `[10]`: `TREATMENTS=[TechSupport, StreamingTV, StreamingMovies]`, `TARGET=Y`, `COVARIATES=` resto. `Y` é a inversão correta de `Churn` (1=ficou).
- ✅ **DAG + identificação de backdoor via DoWhy?** Sim, e bem feito. `CausalModel` (`[70]`) recebe o DAG refinado em GML; `identify_effect()` retorna estimando `backdoor`. *(Após 4.3, o conjunto de ajuste passou a `{InternetService, Contract, tenure}` — antes era `{OnlineSecurity, SeniorCitizen, OnlineBackup, DeviceProtection}`, que omitia os confounders mais fortes.)* Frontdoor reportado como inexistente.

### Pilar 2 — Validação de Premissas (Overlap/Positividade)
- ✅ **Distribuição de Propensity Scores plotada?** Sim (`[47]`): modelo logístico de propensão para `TechSupport` + KDE Tratado vs. Controle. Há boa sobreposição.
- 🟡 **Tratamento de violação de positividade?** Parcial. Há *clipping* dos PS em `[0.05, 0.95]` no IPW (`[49]`) — é um tratamento, mas não há identificação/remoção explícita de unidades fora de suporte, e o overlap só foi validado para **TechSupport** (não para Streaming, nem para o conjunto de covariáveis efetivamente usado no CATE). *(Corrigido em 4.4: features escaladas — `ConvergenceWarning` eliminado.)*

### Pilar 3 — Estimação de CATE
- ✅ **Meta-learners via EconML?** Sim (`[82]`: `TLearner`, `XLearner`; `[84]`: DoWhy+EconML). **Corrigido (#1+#2)** — após limpar as covariáveis, ajustar no treino e calibrar:
  - T-Learner: ATE(teste) = **+0,111**, τ̂ ∈ [−0,30; +0,48] (sem mais extremos ±1).
  - X-Learner: ATE(teste) = **+0,077**, std = **0,080** → heterogeneidade real (não-colapsado). Consistentes com IPW manual (+0,107) e DoWhy PSW (+0,086, agora ajustando {InternetService, Contract, tenure}).
- ✅ **Extrai CATE individual com sucesso?** Sim. Sem vazamento, em conjunto de teste *out-of-sample*; 83% dos clientes com τ̂>0 e dispersão genuína → justifica a Unit Selection. *(Histórico: antes da correção o vetor estava contaminado por colunas internas do DoWhy derivadas de Y — ver Gap #1.)*

### Pilar 4 — Otimização de Política (Unit Selection)
- ✅ **Função de Benefício / ROI com custos e lucros?** Sim (`[92]`): `v=1000` (lucro retido), `c=50` (custo da ação), `benefit = P_complier·(v−c) − c·(P_always+P_never) − (v+c)·P_defier`.
- 🟡 **Lógica explícita de estratos contrafactuais focando Compliers?** Existe (`[91]`), mas é um **proxy ingênuo**, não o framework de Li & Pearl da proposta:
  - Decomposição por diferença de pontos (`P_complier=max(0,y1−y0)`, etc.) assume **monotonicidade**. *(Após as correções #1/#2, `P_defier médio = 0,009 ≈ 0` — a monotonicidade passou a valer empiricamente, legitimando a aproximação. No diagnóstico original era 0,17.)*
  - A proposta promete "extração dos **limites (bounds)** probabilísticos" (Li & Pearl 2019/2022). **Bounds ainda não são calculados** — a etapa permanece um *uplift* sob monotonicidade (Nível 2/3 aproximado), não os bounds formais. *(Único item da proposta ainda pendente.)*
  - A regra final (`benefit > 0`) seleciona **~52–53%** da base (single e multi-tratamento), priorizando quem tem massa de complier suficiente.

### Pilar 5 — Métricas de Avaliação Causal
- ✅ **Qini / AUUC?** **Adicionados (§15.3).** Implementação corrigida por IPW (apropriada a dados observacionais), avaliada no conjunto de teste sobre o *outcome transformado* $z=TY/e-(1-T)Y/(1-e)$. Resultados: **AUUC** Causal 188,5 vs. Aleatório 143,0 vs. Heurística 54,4; **Qini** Causal **+45,5** vs. Heurística **−88,7**. Leitura forte para a RQ1: o ranqueamento causal supera o aleatório, enquanto a **heurística de risco fica *abaixo* do aleatório** para selecionar respondedores. *(O Qini normalizado é vs. um oráculo IPW — limite superior conservador, inflado pela variância do IPW.)* O **§16** repete a métrica para os 3 tratamentos.
- ✅ **Comparativo política causal vs. ML preditivo?** Sim e **corrigido (#3)**: `[95]` agora avalia cada política por **valor de política via IPW sobre desfechos observados no teste** (independente do modelo que ranqueia). Causal **+R$ 206.955** vs. heurística de churn **−R$ 33.383**. *(Histórico: a versão anterior era circular — ranqueava e avaliava com as predições do próprio modelo, inflando o ganho para R$ 919k.)* O **Qini/AUUC formal** foi adicionado em §15.3 (item 5.1).

### Pilar 6 — Refutação e Robustez
- ✅ **Placebo Treatment?** Sim (§11.3 e §13). Resultado correto: o efeito colapsa (ATE 0,086 → −0,25, p=0; CATE 0,076 → ~2e-5, **p=0,96**). Excelente sinal.
- ✅ **Confundidor não-observado?** **Corrigido (#6).** A refutação agora tem **Placebo + Random Common Cause + Data Subset + varredura** de força do confundidor (curva de sensibilidade), além de um markdown de discussão. Resultado honesto: ATE estável a confundimento fraco, **anulável por confundidor moderado (~0,1)**. Com a FCI em qui-quadrado (4.1), o alerta `TechSupport <-> Y` **desapareceu** (restam arestas latentes só entre covariáveis, ex. `tenure <-> Contract`), reforçando a plausibilidade de *unconfoundedness*. *(Original: um único ponto fraquíssimo 0,01/0,02 + FCI Fisher-Z acusando `TechSupport<->Y`.)*

---

## 2. O que está implementado e alinhado (pontos fortes)

1. **Separação formal C/X/Y e limpeza de dados** (`[8]`–`[10]`): conversão de `TotalCharges`, imputação dos 11 clientes `tenure=0`, inversão de `Churn→Y`. Exatamente o que a proposta descreve.
2. **EDA + prova empírica do viés de seleção** (`[24]`–`[26]`): o boxplot `tenure` vs. `TechSupport` (44,9 vs 24,8 meses) é uma demonstração didática e correta do backdoor `X←C→Y`. Este é um dos melhores momentos do trabalho.
3. **Positividade/Overlap** (`[47]`): propensity score plotado, sobreposição evidente — premissa validada visualmente.
4. **ATE via IPW manual com contraste ingênuo vs. causal** (`[49]`): efeito ingênuo 26,5 p.p. → ATE 10,9 p.p. A narrativa "o ML tradicional superestimou em 15,6 p.p." é correta e poderosa.
5. **Descoberta causal completa** (`[55]`–`[68]`): PC + GES + FCI com *Background Knowledge* por *tiers* temporais, consenso de arestas (≥2 algoritmos), scoring BIC/BDeu em `pgmpy`, e seleção do DAG. Escopo muito acima do esperado para um TP.
6. **Pipeline DoWhy correto** (`[70]`–`[74]`): identify → estimate (PSW) → refute, com tratamento cuidadoso de versões de bibliotecas (try/except de imports). Boa engenharia.
7. **Robustez de código defensiva**: compatibilidade entre versões de `pgmpy`/`causal-learn`, checagem de aciclicidade (`assert nx.is_directed_acyclic_graph`), sanidade de `X→Y`.
8. **O placebo refuter passa de forma convincente** — é a evidência de robustez mais forte do notebook.

> Em "existência de código", o projeto cumpre **quase tudo** da proposta. O problema não é cobertura, é **correção/rigor** na terça parte final.

---

## 3. Gaps Críticos (severos — corrigir antes da entrega final)

### ✅ #1 (CORRIGIDO) — VAZAMENTO DE DADOS no CATE / Unit Selection / ROI (era o mais grave)
> **Status: corrigido em `[80]`.** `X_cate` passou a ser definido por uma lista explícita de 16 covariáveis reais (`C_COLS`), com *guard* `assert` anti-vazamento; o split treino/teste foi adicionado. Streaming foi removido das covariáveis (é tratamento e descendente de TechSupport). Notebook reexecutado sem erro.

*Diagnóstico original:* em `[80]`, `X_cate = df_dowhy.drop(columns=["TechSupport","Y"])`. O *print* revela **30 covariáveis**, entre elas:
`propensity_score, ips_weight, tips_weight, cips_weight, ips_normalized_weight, …, ips_stabilized_weight, …, d_y, dbar_y`.

Essas colunas **não existiam** em `df_causal` (que tinha 20 colunas, célula `[52]`). Elas foram **injetadas in-place** pelo `model_dw.estimate_effect(..., propensity_score_weighting)` da §11.2 (o DoWhy muta o DataFrame de entrada). As colunas `d_y` e `dbar_y` são quantidades **derivadas do desfecho Y**, e os pesos são funções do **tratamento**. → **Target leakage clássico.**

**Consequência:** TODOS os modelos de `[82]`, `[84]`, `[91]` (T/X-Learner, m1/m0 dos estratos) treinam com features que codificam o próprio Y/T. Isso explica os números patológicos (T-Learner com ±1, X-Learner colapsado). **Invalida CATE, estratos, benefício e ROI** — exatamente o núcleo do projeto.

**Correção:** definir explicitamente o conjunto de covariáveis *antes* de chamar o DoWhy, ou usar `.copy()`:
```python
C_COLS = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
          'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
          'DeviceProtection','Contract','PaperlessBilling','PaymentMethod',
          'MonthlyCharges','TotalCharges']   # NÃO incluir artefatos do DoWhy
X_cate = df_dowhy[C_COLS].copy()
```
(Decidir também se `StreamingTV`/`StreamingMovies` entram como C — são *tratamentos* na proposta; mantê-los como covariáveis é questionável.)

### ✅ #2 (CORRIGIDO) — Estimadores de CATE degenerados e inconsistência entre métodos
> **Status: corrigido em `[82]` e `[91]`.** Meta-learners ajustados **só no treino** e avaliados no **teste**; modelos contrafactuais (`m1`/`m0`) trocados para `RandomForestClassifier` envolto em `CalibratedClassifierCV` (isotônica). Resultado: T-Learner +0,111, X-Learner +0,077 (std 0,080), coerentes entre si e com IPW/DoWhy; Defiers ≈ 0,009.

*Diagnóstico original:* os ATEs estimados não fechavam entre si: IPW manual **+0,109**, DoWhy PSW **+0,178**, DoWhy+EconML T-Learner **+0,077**, T-Learner solto **−0,053**, X-Learner **−0,000**. Variação de −0,05 a +0,18 sem reconciliação.
- `RandomForestRegressor` em desfecho **binário** sem calibração produz os extremos ±1.
- O X-Learner com `std=0,0007` está efetivamente morto (provável efeito colateral do leakage + propensity degenerado).
Mesmo após corrigir o leakage, é preciso: (a) usar **classificadores calibrados** ou `RandomForestRegressor`/learners apropriados a probabilidade, (b) **train/test split** (hoje treina e avalia na mesma amostra), e (c) diagnosticar por que os métodos divergem.

### ✅ #3 (CORRIGIDO) — Avaliação de ROI é circular (ganho financeiro não era válido)
> **Status: corrigido em `[94]`/`[95]`.** A avaliação passou a usar um **estimador IPW de valor de política sobre os desfechos observados do teste** (`ipw_policy_value_curve`), com propensão ajustada no treino. O ranqueamento (modelo) e a métrica (Y observado) ficam independentes. Resultado honesto: causal +R$ 206.955 vs. heurística −R$ 33.383.

*Diagnóstico original:* em `[95]`, a política causal era **ordenada por** `benefit` (derivado de `y1,y0`) e **avaliada por** `incremental_profit = tau_unit·v − c` (também derivado de `y1,y0`, do mesmo modelo). A política causal é **garantida** vencedora porque é avaliada nas suas próprias predições. O comentário "usamos os contrafactuais estimados como ground truth simulado" reconhece, mas não resolve, o problema de validade. Some-se a isto que `benefit` (ranqueamento) e `incremental_profit` (avaliação) usam **fórmulas diferentes** (uma penaliza defiers, a outra não).
**Correção:** avaliar a política com um estimador **não-enviesado de valor de política** sobre desfechos **observados** — p.ex. *policy value* via IPW/DR, ou **curva Qini/AUUC clássica** (uplift por decis usando o Y real do grupo tratado vs. controle), preferencialmente em **conjunto de teste hold-out**. Para uma prova limpa do conceito, considerar também um **DGP sintético** com ground truth conhecido.

### ✅ #4 (CORRIGIDO) — Qini e AUUC (métrica obrigatória da proposta)
> **Status: adicionado em §15.3.** Implementação própria, corrigida por IPW (não assume RCT), sobre o *outcome transformado* no conjunto de teste. Calcula AUUC e o Qini coefficient para 4 ordenações (causal, heurística, aleatório, oráculo) + Qini normalizado, com gráfico da curva Qini/uplift. Resultados: Causal Qini **+40,06** (acima do aleatório) vs. Heurística **−90,07** (abaixo). Alternativa equivalente pronta: `causalml.metrics` (`qini_score`, `plot_qini`) — porém exigiria correção por IPW para validade observacional, que a implementação manual já faz.

### ✅ #5 (CORRIGIDO) — RQ1 (multi-tratamento) respondida em código
> **Status: adicionado em §16.** O pipeline causal foi generalizado para os **três tratamentos** (`TechSupport`, `StreamingTV`, `StreamingMovies`) no mesmo split, com avaliação **IPW no teste** (ATE/AUUC/Qini/ROI) + política personalizada (melhor incentivo por cliente). Resposta à RQ1: **apenas TechSupport** tem efeito positivo (ATE_IPW +0,173; Qini +45,5; melhor ROI), enquanto StreamingTV (−0,048) e StreamingMovies (−0,054) não retêm.

*Diagnóstico original:* todo o pipeline causal (CATE, estratos, ROI) rodava **apenas para TechSupport**; Streaming nunca era estimado como tratamento.

### ✅ #6 (CORRIGIDO) — Premissa central vs. FCI: agora discutida e estressada
> **Status: corrigido em §11.3 + nova discussão.** Três frentes: (a) a FCI passou a usar **qui-quadrado** (4.1) e **deixou de sinalizar `TechSupport ↔ Y`** — o alerta anterior era artefato do Fisher-Z sobre dados categóricos; (b) a refutação ganhou **Random Common Cause** e **Data Subset** (estimativa inalterada) + uma **varredura** de confundidor não-observado (não mais um único ponto fraco); (c) um **markdown de discussão honesta** sobre a *unconfoundedness*. Resultado honesto da varredura: o ATE (≈0,086) é robusto a confundimento fraco, mas **anulável por um confundidor moderado (força ~0,1)** — fragilidade declarada, não escondida. A conclusão (§12) foi reescrita para refletir isso.

*Diagnóstico original:* a FCI (Fisher-Z) sinalizava `TechSupport <-> Y` como latente, e a conclusão afirmava o oposto ("Placebo e Unobserved Common Cause confirmam a robustez"), sem discutir a tensão; o teste de confundidor era um único ponto fraco (0,01/0,02).

---

## 4. Melhorias de Código

1. ✅ *(aplicado 4.1)* **Teste de independência adequado à natureza dos dados.** PC/FCI usam `fisherz` (linear-gaussiano) sobre variáveis **nominais label-encoded** (`PaymentMethod`, `InternetService`…), o que impõe ordem artificial e invalida os testes. Use `chisq`/`gsq` (causal-learn) para dados discretos. Isso torna a descoberta — e o alerta da FCI — confiável.
2. ✅ *(aplicado 4.2)* **Scores BIC incomparáveis.** A tabela `[67]` tem BIC entre −4,2e5 e −5,6e8 (Baseline absurdo) por causa da discretização em quintis + cobertura de nós diferente. A "vitória do PC por BIC" repousa em números instáveis. Padronize a discretização, garanta o mesmo conjunto de nós em todos os grafos, e reporte por-variável.
3. ✅ *(aplicado 4.3)* **Conjunto de ajuste enxuto demais.** O backdoor do DoWhy ajusta só `{OnlineSecurity, SeniorCitizen, OnlineBackup, DeviceProtection}` — **não inclui `tenure` nem `Contract`**, justamente os confundidores que a §6 provou serem os mais fortes! Isso é consequência do DAG descoberto (PC) e merece reflexão: o DAG estatístico pode ter orientado mal as arestas de `tenure`/`Contract`. Considere fixar essas arestas como *background knowledge* obrigatório ou usar o DAG de domínio para a identificação.
4. ✅ *(aplicado 4.4)* **`StandardScaler` no modelo de propensão** (`[47]`) para eliminar o `ConvergenceWarning` e estabilizar os PS.
5. ✅ *(aplicado 4.5)* **Reprodutibilidade/limpeza de mutação**: nunca passe ao DoWhy o DataFrame que você reutilizará; passe `df_dowhy.copy()`. Isso teria evitado o Gap #1 inteiro.
6. **Separar `benefit` (ranqueamento) e o estimador de valor (avaliação)** com a mesma definição econômica, e mover toda a estimação de CATE/estratos para um **pipeline com train/test split** (e idealmente *cross-fitting*, que é o padrão da EconML para DML/X-Learner).
7. **Calibração das probabilidades** (`CalibratedClassifierCV`) antes de calcular `y1,y0` — a proposta enfatiza Brier/calibração justamente porque erros em `P(Y|X,C)` se propagam ao CATE.
8. **Bibliotecas que faltam para cumprir a proposta**: `causalml` (Qini/AUUC/uplift curves prontos) e, para os *bounds* de Li & Pearl, implementar as desigualdades de `[2,3]` (a lib `y0` citada no plano pode ajudar na manipulação do estimando, mas os bounds de unit selection precisam ser codificados à mão).
9. ✅ *(aplicado 4.9)* **Erros menores**: título do plot em `[76]` tem concatenação de string quebrada (faltou `\n`/espaço, fica "…BIC**Vermelho**"); avisos `FutureWarning` (`force_all_finite`, `palette sem hue`) — cosméticos.

---

## 5. Resumo Executivo para o Relatório Final

**O que enfatizar (está forte e é verdadeiro):**
- A **prova do viés de seleção** (tenure 44,9 vs 24,8) e a **queda do efeito ingênuo 26,5 p.p. → ATE 10,9 p.p.** — é a sua narrativa mais convincente e está correta.
- O **rigor metodológico da descoberta causal** (PC/GES/FCI + tiers + consenso + scoring + validação DoWhy) demonstra maturidade muito acima da média.
- O **placebo refuter** passando (p=0,96) é evidência legítima de robustez — destaque-a.

**O que já pode ser reportado com segurança (após as correções):**
- **Ganho financeiro honesto**: causal **+R$ 206.955** vs. heurística **−R$ 33.383** (policy value via IPW no teste). O número antigo de R$ 919k (avaliação circular) foi descartado.
- **Heterogeneidade real**: X-Learner com std 0,080 e 83% dos clientes com τ̂>0 — a frase "tratamento não-uniforme" agora é verdadeira.
- **Qini/AUUC**: o ranqueamento causal supera o aleatório e a heurística de risco fica abaixo do aleatório (RQ1).

**O que ainda exige cautela na redação:**
- "Classificamos clientes via **Li & Pearl 2019**": é uma decomposição de uplift sob monotonicidade. Agora a base **respeita** a monotonicidade (Defiers 0,009 ≈ 0), o que legitima a aproximação — mas os **bounds** formais de Li & Pearl ainda não são calculados. Seja explícito: "aproximação por uplift sob monotonicidade empiricamente válida", e mencione os bounds como trabalho futuro (Nível 3 pleno).

**Plano mínimo para uma entrega final defensável (em ordem de prioridade):**
1. ✅ **Corrigir o leakage** (Gap #1) — `X_cate` só com C reais; §12–§15 reexecutadas. *Feito.*
2. ✅ **Reestimar CATE** com learners calibrados + train/test split; ATEs reconciliados (+0,077 a +0,178). *Feito.*
3. ✅ **Calcular Qini e AUUC** sobre Y observado (IPW), em teste. *Feito (§15.3).*
4. ✅ **Avaliar ROI** por *policy value* não-enviesado (IPW) no teste. *Feito (§15.2).*
5. ✅ **Estender a TechSupport → Streaming** para responder RQ1 (multi-tratamento). *Feito (§16): só TechSupport tem efeito positivo.*
6. ✅ **Sensibilidade de confundimento como varredura** + **discussão da FCI**. *Feito (§11.3): Placebo + Random Common Cause + Data Subset + varredura de confundidor (cruza zero ~0,1) + markdown de discussão. A FCI em qui-quadrado já não acusa `TechSupport↔Y`.*
7. ✅ **Corrigir os números report×código**. *Feito (§5.3): tabela autoritativa com PR-AUC minoritária (LR 0,610 / RF 0,642), ROC-AUC (0,832 / 0,835) e Brier (0,1415 / 0,1388). O RF é o melhor nas três e é o base learner usado (calibrado) no estágio causal — texto, tabela e código agora consistentes.* **Ação restante (só redação):** alinhar o texto/Tabela 1 do **relatório final** a esses números (a Tabela 1 do intermediário, com 0,84/0,82 e "LogReg melhor", está incorreta).

> **Mensagem-chave:** você construiu o esqueleto causal completo e a fundação (Identify/Estimate-ATE/Refute) está sólida. O valor do projeto vive na terça final (CATE→Unit Selection→ROI), e hoje ela está minada por um único bug de leakage que se propaga, somado a uma avaliação circular. Corrigidos esses dois pontos e adicionados Qini/AUUC, o trabalho passa de "amplo, mas frágil no clímax" para "rigoroso de ponta a ponta".
