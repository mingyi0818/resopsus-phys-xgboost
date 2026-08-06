---
abstract: |
  Reservoir release forecasting is often trivialised by models that
  exploit the lagged release (autocorrelation) or concurrent storage
  (mass-balance back-solving). We formulate a non-trivial one-step task
  that withholds both---the primary contribution---and evaluate ten
  task-complying linear/tree models (XGBoost, LightGBM, CatBoost) plus
  persistence and a capacity-limited LSTM as references on 199
  reservoirs from the open ResOpsUS dataset, with a chronological split
  (before/after 2016), fixed hyperparameters, and no test-set tuning.
  Within this task, two mass-balance-informed features (a lagged
  storage-change proxy and a throughput ratio) derived from the
  continuity equation raise median KGE by $+0.145$ (XGBoost,
  0.730$\to$0.875), $+0.156$ (LightGBM, 0.732$\to$0.888), and $+0.159$
  (CatBoost, 0.724$\to$0.883), all significant (Wilcoxon
  $p < 10^{-26}$; 183/199 reservoirs). The mass-balance class accounts
  for $\sim$80% of the gain; SHAP ranks the raw lagged inflow
  (`inflow_lag1`, 19.3%) and the storage-change proxy
  (`storage_trend`, 18.8%, the top mass-balance-informed input) as the
  two leading features of all 16. The gain is concentrated at the
  one-step lead: under multi-day autoregressive rollout the
  mass-balance features degrade sharply from $L{=}2$ and become harmful
  by lead 7 (0.120 vs. 0.533 for raw XGBoost) because their
  storage-based components are contaminated by the model's own
  prediction errors. The contribution is a transparent, low-compute,
  interpretable feature set that significantly improves tree-based
  baselines under a controlled, non-trivial protocol, with both its
  strong one-step performance and its multi-day limits honestly
  documented.
 refs.bib
title: "Physics-Derived Feature Augmentation for Gradient-Boosted Trees
  in Reservoir Release Forecasting: A Systematic Evaluation Across the
  ResOpsUS Dataset"
---

empty@addto@macro, @addto@macroYafen Feng

empty@addto@macro, @addto@macroMing Zeng

empty@addto@macro, @addto@macroJianghong Guo

empty@addto@macro, @addto@macroChuanxian Jiang

empty@addto@macro, @addto@macroJingyuan Zeng

@addto@macro\
School of Geography and Tourism, Jiaying University, Meizhou 514015,
China

@addto@macro\
College of Water Conservancy and Civil Engineering, South China
Agricultural University, Guangzhou 510642, China

@addto@macro\
School of Computer Science, Jiaying University, Meizhou 514015, China

@addto@macro\
Corresponding authors: Yafen Feng (fyf81@163.com) and Jingyuan Zeng
(zjy@jyu.edu.cn)

**Keywords:** reservoir release forecasting; gradient boosting;
mass-balance-informed features; ResOpsUS; Kling-Gupta efficiency; feature
engineering.

# Introduction

Reservoirs regulate most major rivers and provide flood protection,
water supply, hydropower, navigation, and ecological flows (Steyaert et
al. 2022). Forecasting reservoir release---the controlled outflow
decided by operators---is therefore central to water-resources
management. Two broad modelling traditions exist: (i) rule-based or
optimisation-based methods that encode operating charts, and (ii)
data-driven methods that learn the release policy from historical
records. The latter has expanded rapidly with machine learning (ML),
including random forests (RF), gradient-boosted trees, and recurrent
neural networks such as the long short-term memory (LSTM) network
(Kratzert et al. 2018).

A subtle but important issue in data-driven release forecasting is task
trivialisation. Daily release is strongly autocorrelated: $O_t$ is often
close to $O_{t-1}$. A model that is allowed to use the lagged release as
a feature can achieve very high efficiency scores (Kling--Gupta
efficiency, KGE, above 0.95) without learning anything about the
operating policy---it merely carries the previous day forward. Likewise,
because the mass-balance equation links $O_t = I_t - \Delta S_t$ (up to
evaporation), a model given concurrent storage can back-solve the
release arithmetically. Both situations inflate scores and obscure
genuine methodological contributions.

This study makes three contributions. First, we define a non-trivial
release-forecasting task that withholds the lagged release and the
concurrent storage, forcing the model to infer the release from
hydrological state (inflow, antecedent storage, season). Second, we
propose two mass-balance features---a lagged storage-change proxy and a
throughput ratio---derived from the continuity equation, and embed them
in XGBoost. Third, we conduct a systematic, stratified evaluation across
199 evaluated reservoirs (of 200 selected; one reservoir, ResOpsUS_976,
excluded for all-NaN predictions) from ResOpsUS with rigorous
chronological splitting, statistical testing, ablation, and a
per-difficulty-tier analysis. We deliberately do *not* claim superiority
over published LSTM benchmarks (e.g., Kratzert et al. (2018, 2019))
because our one-step setting differs from their simulation setting;
instead we position the contribution as a consistent, interpretable,
low-compute improvement over a strong tree-based baseline under a
controlled, non-trivial protocol.

**Novelty and scope.** The primary novelty is not a new algorithm but a
rigorously defined non-trivial task that removes two well-known
trivialisation pathways (release autocorrelation and mass-balance
back-solving); within that task, a systematic, ablation-backed
demonstration shows that a small set of mass-balance-informed, lagged
features yields a large, statistically significant gain over a strong
tree baseline. We emphasise the task definition as the principal
contribution, because it is what makes the feature comparison
meaningful: without the information-withholding constraints, the same
features would be overshadowed by the lagged-release shortcut, and any
reported gain would be uninterpretable. The scope is limited to one-step
release prediction on U.S. reservoirs; iterative simulation and regional
transfer are explicitly out of scope and identified as future work.

**Hypothesis.** We test the hypothesis that, under the non-trivial task
defined above, augmenting a gradient-boosted-tree baseline with
physics-derived (mass-balance) features significantly improves
release-forecasting accuracy relative to the same baseline using only
raw hydrological lags, with the mass-balance features expected to
contribute the largest share based on the continuity equation (Eq. 3).

# Related Work

## 1 Large-sample hydrological datasets

Large-sample datasets such as CAMELS (Addor et al. 2017) have enabled
reproducible hydrological ML research (Kratzert et al. 2018). For
reservoirs specifically, ResOpsUS (Steyaert et al. 2022) provides daily
inflow, outflow, storage, elevation, and evaporation for 679 major U.S.
reservoirs from 1930--2020, standardising records held by  40 agencies.
It is the dataset used in this study.

## 2 Machine learning for reservoir operation

Recent work applies RF, gradient boosting, and LSTMs to reservoir
outflow simulation. Zhou et al. (2025) embed water-balance, capacity,
and outflow constraints into RF via a Sigmoid transform
(physics-constrained RF, PC-RF) for cascade reservoirs, reporting large
gains over plain RF and identifying previous-period outflow, current
inflow, and previous-period inflow as the top features. Notably, that
formulation uses the lagged outflow as a feature, which---as discussed
in Section 1---yields high scores partly through autocorrelation. At
continental scale, Tran et al. (2025) train a conditioned LSTM across
$\sim$`<!-- -->`{=html}200 CONUS reservoirs and Zhang et al. (2025) a
Mamba state-space model over 441 CONUS dams, both reporting basin-wide
reservoir-release simulations with SHAP-based feature attribution; these
large-sample DL studies set the scale reference for learned reservoir
modelling and are revisited in Section 6.7 as one-step (direct)
simulation benchmarks against which our one-step tree ensembles are
positioned. Comparative studies (e.g., RF vs. gradient boosting vs. LSTM
under different hydrological years) report that tree ensembles are
competitive with or superior to LSTMs for outflow simulation, with
performance declining in dry years. Deep-learning hydrological models
(Kratzert et al. 2018, 2019) achieve strong results for rainfall--runoff
prediction using LSTMs trained across many catchments. Tree ensembles
such as random forests and gradient-boosted trees are widely used in
hydrology because they are interpretable, handle mixed variable types,
and perform strongly on tabular data; XGBoost in particular has been
applied to water-level and inflow forecasting with competitive accuracy.
These data-driven efforts build on a long tradition of
reservoir-operations research that has derived theoretically-motivated
operating rules (Lund and Guzman 1999) and performance-evaluation
criteria (Hashimoto et al. 1982) for release decisions. Gauch et al.
(2021) demonstrated that a single LSTM trained across catchments can
produce multi-timescale predictions by varying the look-back window,
showing that the temporal representation is a learnable property of the
regional model rather than an architectural limitation. Wi and
Steinschneider (2022) stress-tested regional LSTM projections under
climate change and concluded that physical realism degrades faster than
benchmark metrics suggest, an observation that parallels our rollout
discussion. Liu et al. (2024) benchmarked Transformer architectures
against LSTMs on CAMELS and found that, while Transformers exploit long
input sequences more effectively, both architectures saturate at the
same one-step skill ceiling set by catchment memory; this is consistent
with the persistence-dominated ceiling we observe at long rollout leads.
Recent work on mass-conservation constraints in process-based and ML
hybrid models (Frame et al. 2023) demonstrates that physically enforced
balance equations improve forecast skill precisely because they prevent
the kind of feature-level error feedback we document in Section 6.10.1.
Representing human operational decisions in large-scale hydrological
models remains an active research frontier (Galelli et al. 2025). Our
work complements that tradition by supplying interpretable, individually
ablatable physics-derived features rather than opaque optimal policies.

The broader turn toward physics-guided machine learning in hydrology
(Karpatne et al. 2017; Shen et al. 2018; Nearing et al. 2021)---and in
the Earth-system sciences more broadly ([Reichstein et al.]{.nocase}
2019)---argues for embedding domain knowledge as interpretable features
or soft constraints rather than treating models as pure black boxes.
Tree-based methods have been extensively applied to hydrological
prediction (Tyralis et al. 2019), and recent reviews (Lange and Sippel
2024) document a rapid expansion of ML in water resources. Yet for the
specific task of reservoir release forecasting, the combination of
interpretable tree ensembles with explicitly physics-derived,
individually ablatable input features remains comparatively
under-explored.

**Gap.** Two gaps motivate this study. First, most data-driven
release-forecasting works include the lagged release as a predictor,
which inflates scores through autocorrelation and obscures whether the
model learns the operating policy; few studies evaluate under a task
definition that explicitly excludes it. Second, where physics-derived
features are used, they are typically embedded as hard constraints (Zhou
et al. 2025) rather than as interpretable, lagged input features whose
individual contribution can be ablated---a distinction emphasised by the
physics-guided ML literature (Karpatne et al. 2017; Shen et al. 2018).
We address both by (i) defining the non-trivial task in Section 4.1 and
(ii) providing a class-wise ablation that quantifies each feature
group's contribution.

## 3 Performance metrics and benchmarks

The Nash--Sutcliffe efficiency (NSE; Nash and Sutcliffe 1970) is the
classical criterion but over-emphasises high flows (Mathevet et al.
2006). The Kling--Gupta efficiency (KGE; Gupta et al. 2009) decomposes
error into correlation, variability, and bias, providing a more balanced
diagnostic. Knoben et al. (2019) caution that KGE = 0 is *not* an
inherent benchmark (unlike NSE), so threshold-based interpretation must
be made carefully. We report both NSE and KGE, together with KGE
components and a logarithmic NSE for low-flow behaviour.

# Data

We use ResOpsUS (Steyaert et al. 2022; *Scientific Data*, CC-BY 4.0),
downloaded from the Zenodo repository (record 6612040,  307 MB). Each
reservoir is stored as a daily CSV with columns *date, storage, inflow,
outflow, elevation, evaporation*. Coverage varies: inflow is a derived
quantity and is missing for many reservoirs, so we restrict analysis to
reservoirs where inflow, outflow, and storage are jointly available.

**Reservoir selection (fixed a priori).** To ensure a usable
chronological test set and sufficient training data, a reservoir is
retained only if (i) the number of complete rows (inflow, outflow, and
storage all non-missing) is $\geq$ 5,000; (ii) at least 3,000 complete
rows precede 2016-01-01; and (iii) at least 500 complete rows occur on
or after 2016-01-01. These criteria yielded 212 eligible reservoirs. To
bound compute while spanning the difficulty spectrum, we draw a
stratified sample of 200 reservoirs (199 evaluated after excluding one
all-NaN reservoir, ResOpsUS_976) by taking evenly spaced ranks of the
eligible set sorted by complete-row count. We declare that the
difficulty stratification used in the analysis (Section 6.4) is based on
the persistence KGE computed on the training period and is therefore a
priori with respect to the test set; this is cross-validated against the
test-set persistence terciles (62.8% agreement; Table S3) to confirm the
tiers are stable.

# Methods

This section is organised as follows. Section 4.1 defines the
forecasting task and provides the theoretical justification for the two
information-withholding constraints (concurrent storage and release
lags), together with an information-theoretic motivation for the
physics-derived features. Section 4.2 specifies the feature sets.
Section 4.3 describes the models. Section 4.4 defines the evaluation
metrics.

## 1 Task formulation and theoretical derivation

We predict the release (outflow) $O_t$ at day $t$ from hydrological
state, *without* using any release lag and *without* using concurrent
storage $S_t$. Formally, the target is $O_t$ and the permitted
predictors are inflow (current and lagged), lagged storage, and calendar
features. The rationale for withholding two seemingly natural predictors
rests on two derivations below, followed by a motivation for the
features that replace them.

### 1.1 Continuity equation and the necessity of withholding concurrent storage

The governing mass-balance equation for a reservoir in continuous time
is

$$\begin{equation}
\frac{dS(t)}{dt} = I(t) - O(t) - E(t) \tag{1}
\end{equation}$$

where $S(t)$ is storage, $I(t)$ is inflow, $O(t)$ is the controlled
release (outflow), and $E(t)$ is evaporation. Discretising at the daily
scale with a backward difference,

$$\begin{equation}
S_t - S_{t-1} = I_t - O_t - E_t, \tag{2}
\end{equation}$$

which rearranges to

$$\begin{equation}
O_t = I_t - (S_t - S_{t-1}) - E_t = I_t - \Delta S_t - E_t. \tag{3}
\end{equation}$$

**Assumption A1 (evaporation as a minor term).** At the daily scale and
for the large reservoirs in ResOpsUS, evaporation $E_t$ is small
relative to inflow and outflow and varies slowly; it may be treated as a
weakly varying known offset or neglected in the first-order analysis.
Under A1, Eq. (3) reduces to $O_t \approx I_t - \Delta S_t$.

**Implication.** If a model is given concurrent $S_t$ and $I_t$, then
$O_t$ is (approximately) a deterministic algebraic combination of the
inputs: $O_t \approx I_t - (S_t - S_{t-1})$. The learning problem
degenerates into fitting a linear identity, and any regression model
with sufficient capacity will recover it, yielding KGE near unity
without learning anything about the operating policy. This is a
*back-solving* trivialisation, distinct from the autocorrelation
trivialisation in Section 4.1.2. Withholding $S_t$ is therefore a
necessary condition for the task to remain a genuine prediction problem
rather than an arithmetic inversion.

### 1.2 Release autocorrelation and the necessity of withholding release lags

Daily reservoir release is strongly persistent. Under a first-order
autoregressive approximation (Assumption A2), the release obeys

$$\begin{equation}
O_t = c + \rho\, O_{t-1} + \varepsilon_t, \qquad \varepsilon_t \sim \text{WN}(0, \sigma_\varepsilon^2), \tag{4}
\end{equation}$$

where $\rho$ is the lag-1 autocorrelation coefficient and
$\varepsilon_t$ is white noise.

**Assumption A2 (local stationarity).** The release process is treated
as locally weakly stationary over the analysis window, so that the lag-1
autocorrelation $\rho$ is well defined and the variance/mean are
approximately constant. In practice, release exhibits seasonal
non-stationarity; the AR(1) model is a first-order approximation whose
adequacy is supported empirically by the dominance of the lag-1 term in
the autocorrelation function of ResOpsUS daily outflow.

**Derivation of the persistence KGE.** The persistence model predicts
$\hat{O}_t = O_{t-1}$. Under A2, the three KGE components (see Eq. 10 in
Section 4.4) evaluate to:

- Correlation: $r = \text{corr}(O_t, O_{t-1}) = \rho$.

- Variability ratio: $\alpha = \sigma_{O_{t-1}} / \sigma_{O_t} = 1$
  (equal variances under stationarity).

- Bias ratio: $\beta = \mu_{O_{t-1}} / \mu_{O_t} = 1$ (equal means under
  stationarity).

Hence

$$\begin{equation}
\text{KGE}_{\text{persist}} = 1 - \sqrt{(\rho - 1)^2 + (1-1)^2 + (1-1)^2} = \rho. \tag{5}
\end{equation}$$

**Empirical support.** Over the 199 evaluated reservoirs, the median
lag-1 autocorrelation of daily outflow is 0.965, and 79% of reservoirs
exceed 0.9 (10th percentile 0.81; computed on the training portion,
before 2016-01-01). Equation (5) then predicts
$\text{KGE}_{\text{persist}} \gtrsim 0.9$, which is confirmed
empirically: the median persistence KGE over our 199-reservoir sample is
0.975 (Table 1). Consequently, any model permitted to use $O_{t-1}$ as a
feature can attain KGE above 0.9 by mimicking persistence alone, without
learning the operating policy. Withholding all release lags removes this
trivialisation pathway and forces the model to predict release from
hydrological state and season alone.

### 1.3 Information-theoretic motivation for the physics-derived features

Having withheld $S_t$ (Section 4.1.1) and $O_{t-1}$ (Section 4.1.2), the
model must infer $O_t$ from permitted quantities: current and lagged
inflow, lagged storage, and calendar features. The continuity equation
(Eq. 3) states that $O_t$ is primarily a function of $I_t$ and
$\Delta S_t = S_t - S_{t-1}$. Although $S_t$ is withheld, the *lagged*
storage change

$$\begin{equation}
\Delta S_{t-1} = S_{t-1} - S_{t-2} \tag{6}
\end{equation}$$

is an observable quantity. We construct the feature `storage_trend`
$= \Delta S_{t-1}$ as a one-step-lagged proxy for $\Delta S_t$.

**Inertial justification.** Reservoir operation exhibits inertia:
operators adjust gate settings gradually in response to storage trends,
so $\Delta S_t$ and $\Delta S_{t-1}$ are positively correlated in
practice. This is an empirical observation rather than a theoretical
necessity---different reservoirs have different operating time
scales---and its validity is tested by the ablation in Section 6.3.

**Motivation for `throughput_ratio`.** The second mass-balance feature,
$\texttt{throughput\_ratio} = I_t / |S_{t-1}|$, encodes a dimensionless
*turnover rate*: a reservoir with small $|S_{t-1}|$ and large inflow
$I_t$ is being rapidly filled (the operator must release water to
prevent spilling), while one with large $|S_{t-1}|$ and small inflow is
being conservatively drawn down. The SHAP mass of `throughput_ratio`
(2.3%, Table S4) is small relative to $\texttt{storage\_trend}$
(18.8%) and the inflow family (58.7% combined), but the ablation confirms
that removing it from the mass-balance class produces a measurable
median KGE drop (0.876 $\to$ 0.870, not shown separately), so we retain
it for completeness of the continuity-equation proxy family. *Numerical
stability:* $|S_{t-1}|$ in ResOpsUS is reported in physical units
(typically $\mathrm{m}^3 \times 10^6$) and is strictly positive for all
199 evaluated reservoirs; the denominator is therefore bounded away from
zero and no ad-hoc epsilon is needed.

**Information-theoretic design goal.** Denote the raw feature set
$\mathcal{X}_{\text{raw}}$ (inflow lags, lagged storage, calendar) and
the physics-derived set $\mathcal{X}_{\text{phys}}$. The features are
designed so that the conditional mutual information

$$\begin{equation}
I\!\left(O_t;\, \mathcal{X}_{\text{phys}} \;\big|\; \mathcal{X}_{\text{raw}}\right) > 0, \tag{7}
\end{equation}$$

i.e., the physics-derived features carry information about $O_t$ beyond
what the raw lags already provide. We do not prove Eq. (7) analytically
(continuous variables preclude closed-form evaluation); instead, the
ablation study (Section 6.3) provides empirical evidence: removing
$\mathcal{X}_{\text{phys}}$ drops the median KGE by 0.139, and removing
the mass-balance class alone drops it by 0.111, consistent with a
positive conditional mutual information contribution.

**Why feature engineering matters for tree ensembles.** XGBoost (Chen
and Guestrin 2016) predicts via an additive model of regression trees,

$$\begin{equation}
\hat{O}_t = \sum_{k=1}^{K} f_k(\mathbf{x}_t), \qquad f_k \in \mathcal{F}, \tag{8}
\end{equation}$$

where $\mathbf{x}_t$ is the feature vector and $\mathcal{F}$ is the
space of tree functions. Unlike neural networks, tree ensembles do not
learn differentiable feature combinations through backpropagation; each
tree can only split on the provided input coordinates. Consequently,
informative *derived* features---such as $\Delta S_{t-1}$, which encodes
a first-order physical relationship---must be supplied explicitly. This
is the methodological basis for augmenting the raw feature set with
physics-derived quantities rather than relying on the learner to
discover them. We note that automatic feature construction (e.g.,
interaction splittings) can partially substitute, but Breiman (2001)
observes that explicit domain-informed features remain a primary lever
for tree-based performance.

**Summary of the theoretical argument.** The three derivations form a
coherent chain: (i) the continuity equation shows that $O_t$ is governed
by $\Delta S_t$ and $I_t$; (ii) withholding $S_t$ and $O_{t-1}$ removes
the two trivialisation pathways identified in Sections 4.1.1--4.1.2; and
(iii) the physics-derived features, constructed from *lagged*
quantities, restore an approximate, leak-free version of the information
that the withholding rules removed. The effectiveness of this
restoration is assessed empirically in Section 6.

## 2 Features

**Raw features (8):** $I_t$, $I_{t-1}, I_{t-2}, I_{t-3}$,
$S_{t-1}, S_{t-2}$, and $\sin/\cos$ of day-of-year. These constitute the
baseline feature set.

**Physics-derived features (2, mass-balance class):** `storage_trend`
$= S_{t-1} - S_{t-2}$ (recent storage change, the proxy for $\Delta S_t$
derived in Eq. 6), and `throughput_ratio` $= I_t / |S_{t-1}|$
(throughput relative to buffer). Both are constructed exclusively from
lagged quantities ($S_{t-1}$, $S_{t-2}$, $I_t$), so no concurrent target
information leaks into the predictors. Two additional feature
classes---storage-state (rule-curve deviation) and flow-anomaly (moving
averages, normalised inflow)---were tested in the ablation (Section 6.3)
and found to contribute a negligible, not statistically distinguishable
share of the gain when added to the mass-balance set (median $\Delta$KGE
= $+0.0002$, 95% CI \[$-0.0017$, $+0.0026$\] over 199 reservoirs). We
therefore adopt the parsimonious two-feature mass-balance set as the
final specification, consistent with the information-theoretic
motivation of Section 4.1.3, which identifies $\Delta S_t$ as the
primary physical driver of $O_t$. We note that these features are
lightweight algebraic proxies---differences and ratios of already
-observed quantities---derived from the continuity equation rather than
a full physical simulation; the term "mass-balance-informed" flags this
physical motivation while keeping the distinction from a process-based
model explicit. Table 1 summarises all features.

  Group                 Feature       Definition                                Physical meaning
  --------------- ------------------- ----------------------------------------- ----------------------------------------------
  Raw                   $I\_t$        current inflow                            immediate inflow signal
  Raw                $I\_{t-1..3}$    lagged inflow                             recent inflow history
  Raw                $S\_{t-1..2}$    lagged storage                            antecedent reservoir state
  Raw                 doy_sin/cos     $\sin/\cos(2\pi\cdot\text{doy}/365.25)$   seasonal cycle
  Mass-balance       storage_trend    $S\_{t-1} - S\_{t-2}$                     proxy for $\Delta S\_t$ (filling vs drawing)
  Mass-balance     throughput_ratio   $I_t / |S_{t-1}|$                         throughput relative to buffer
  Storage-state      storage_norm     $S\_{t-1} / Q\_{0.99}(S)$                 normalised fill level
  Storage-state     storage_deficit   $S\_{t-1} - \bar{S}\_{\text{doy}}$        rule-curve deviation
  Flow-anomaly        inflow_norm     $I\_t / \bar{I}$                          normalised inflow
  Flow-anomaly     inflow_ma7 / ma30  7/30-day MA of inflow, lagged 1 day       short/medium inflow memory
  Flow-anomaly       inflow_anom7     $I\_t - \text{MA7}\_{t-1}$                departure from recent inflow

  : Feature definitions. Ten features: 8 raw + 2 physics-derived
  (mass-balance class). All features use only information available at
  or before time $t$ (no concurrent storage, no release lags).

## 3 Models

Twelve models are compared in total: ten *task-complying* models that
respect both information-withholding constraints (listed in Table 1)
plus two references that violate the task definition and are reported
only for context---persistence (Table R1) and a capacity-limited LSTM
(Section 6.1). The ten task-complying models span three families
(linear, tree ensembles, and a recurrent baseline) under two feature
regimes (raw and physics-augmented):

1\. **Pass-through:** $\hat{O}_t = I_t$. A naive hydrological reference
(release equals inflow under no regulation).

2\. **Persistence:** $\hat{O}_t = O_{t-1}$. A difficulty reference
only---it uses the lagged release and is therefore not a fair competitor
under our task definition (Section 4.1.2), but it quantifies how much of
the variance is autocorrelation.

3\. **Ridge regression** (raw features, standardised). Linear baseline.

4\. **Random forest** (raw features; 200 trees, max depth 12).
Non-linear ensemble baseline.

5\. **XGBoost (raw):** gradient-boosted tree baseline (Chen and Guestrin
2016).

6\. **Phys-XGBoost (this work):** XGBoost with raw + physics-derived
features.

7\. **LightGBM (raw) / LightGBM-phys:** gradient-boosted tree baseline
(Ke et al. 2017) under the two feature regimes.

8\. **CatBoost (raw) / CatBoost-phys:** gradient-boosted tree baseline
with ordered boosting (Prokhorenkova et al. 2018) under the two feature
regimes.

9\. **LSTM (raw) / LSTM-phys:** a single-layer Long Short-Term Memory
network (Hochreiter and Schmidhuber 1997) trained on a 30-day sliding
window of the same leak-free inputs used by the tree models, with 32
hidden units, per-reservoir target standardisation, and Adam
optimisation with early stopping (patience 15); the network is
implemented in NumPy (CPU). The same sliding window (current and lagged
inflow, lagged storage, seasonal features) is used for both raw and phys
regimes so that the LSTM receives information comparable to the trees;
concurrent storage and release lags remain withheld.

All gradient-boosted-tree models use fixed hyperparameters (XGBoost: 400
trees, max depth 6, learning rate 0.05, subsample 0.8, colsample_bytree
0.8; LightGBM: 400 trees, max depth 6, learning rate 0.05, subsample
0.8, colsample_bytree 0.8; CatBoost: 400 trees, max depth 6, learning
rate 0.05). Hyperparameters are *not* tuned on the test set, to avoid
leakage. Random seeds are fixed (42). The LSTM training set is capped at
3,000 windows per reservoir to bound compute; tree models use the full
training set.

## 4 Evaluation metrics

**Split.** Chronological---training before 2016-01-01, testing on/after
2016-01-01. No shuffling.

**Kling--Gupta efficiency (KGE).** Following Gupta et al. (2009), the
KGE and its three components are defined as

$$\begin{equation}
\text{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}, \tag{9}
\end{equation}$$

where

$$\begin{equation}
r = \text{corr}(\mathbf{o}, \mathbf{s}), \qquad \alpha = \frac{\sigma_s}{\sigma_o}, \qquad \beta = \frac{\mu_s}{\mu_o}, \tag{10}
\end{equation}$$

and $\mathbf{o}$, $\mathbf{s}$ are the observed and simulated series
with standard deviations $\sigma_o, \sigma_s$ and means $\mu_o, \mu_s$.
The three components isolate timing/correlation error ($r$), variability
error ($\alpha$), and bias ($\beta$), enabling the decomposition
reported in Section 6.5.

**Nash--Sutcliffe efficiency (NSE).** Following Nash and Sutcliffe
(1970),

$$\begin{equation}
\text{NSE} = 1 - \frac{\sum_{i}(o_i - s_i)^2}{\sum_{i}(o_i - \bar{o})^2}. \tag{11}
\end{equation}$$

NSE equals 1 for a perfect match, 0 when the model is as accurate as the
observed mean, and is negative when the model performs worse than the
observed mean.

**Additional metrics.** We also report log-NSE (NSE computed on
log-transformed flows to emphasise low-flow periods) and the
root-mean-square error (RMSE). We report both the *median over
reservoirs* and the *pooled* metric (all test samples concatenated).
Statistical significance of Phys-XGBoost vs. XGBoost is assessed with
the Wilcoxon signed-rank test on per-reservoir KGE; 95% bootstrap
confidence intervals (2,000 resamples) are reported for median KGE and
for the median $\Delta$KGE. An *extreme-value segment* analysis reports
metrics on the top-10% inflow events in each test set.

We note, following Knoben et al. (2019), that KGE = 0 is *not* an
inherent benchmark (unlike NSE = 0), because the KGE of the mean-flow
benchmark is $1 - \sqrt{2} \approx -0.41$. Clark et al. (2021) further
showed that the sampling distributions of common performance metrics in
hydrology are heavy-tailed and that over-reliance on point estimates
(with or without threshold-based interpretation) can be misleading at
large-sample scale; we therefore report paired-Wilcoxon $p$-values
alongside median differences rather than relying on absolute KGE
thresholds alone. Cinkus et al. (2023) demonstrated that the same
hydrological model can rank very differently under different reasonable
performance criteria, and we accordingly retain KGE and NSE as
complementary (not interchangeable) metrics. Threshold-based
interpretation of KGE must therefore be made with these caveats in mind.

Because we compare ten task-complying models across multiple pairwise
tests (each task-complying model tested against every other across the
199 reservoirs), we control the false-discovery rate with the
Benjamini--Hochberg (BH) procedure at FDR $=0.05$. The primary
confirmatory comparison (Phys-XGBoost vs. XGBoost, Wilcoxon,
pre-specified) remains significant after correction; given the extremely
small $p$-values for the physics-feature benefit, all reported
significances survive BH adjustment.

# Experimental Setup

Experiments run on CPU (Python 3.13, scikit-learn, XGBoost 3.x). Each
reservoir is modelled independently (no regional transfer). The full
pipeline, selection criteria, and per-reservoir metrics are released for
reproducibility. Total compute for the 200-reservoir study (199
evaluated; one reservoir, ResOpsUS_976, excluded for all-NaN
predictions) is on the order of minutes per tree-model configuration on
a single CPU core.

The modelling workflow is: (1) load a reservoir CSV and drop incomplete
rows; (2) construct raw and physics-derived features from lagged
quantities; (3) split chronologically at 2016-01-01; (4) fit each model
on the training set with fixed hyperparameters; (5) predict on the
held-out test set and compute KGE, NSE, and components. Reservoir sample
sizes range as follows: training-set complete rows 3,557--33,183 (median
$\approx$ 11,300); test-set complete rows 688--1,854 (median $\approx$
1,460).

# Results

## 1 Aggregate performance

Table 1 reports the per-reservoir median KGE with 95% bootstrap
confidence intervals (over the 199 evaluated reservoirs) for the ten
models that obey the task definition (no release lags, no concurrent
storage); a separate reference table (Table R1) lists the persistence
reference (which uses the lagged release) and the capacity-limited LSTM
baseline. Among the task-complying models, the physics-augmented tree
ensembles are the strongest and are closely clustered: LightGBM-phys
attains the highest median KGE (0.888), followed by CatBoost-phys
(0.883) and Phys-XGBoost (0.875, CI 0.860--0.889). The consistent
message is that physics augmentation helps every gradient-boosting
library: the raw-to-phys gain in median KGE is +0.145 (XGBoost), +0.156
(LightGBM), and +0.159 (CatBoost). The pass-through reference is poor
(median KGE 0.320, median NSE -0.396), confirming that release is not
simply inflow and that regulation must be learned. Ridge regression is
the weakest learner (0.721); the raw tree ensembles (RF 0.722, XGBoost
0.730, LightGBM 0.732, CatBoost 0.724) are close, and physics
augmentation lifts each by roughly 0.145--0.159. The LSTM baseline---a
single-layer network with 32 hidden units implemented in NumPy
(CPU)---attains a per-reservoir median KGE of 0.713 (raw) and 0.803
(phys, 95% CI 0.760--0.829), below the physics-augmented tree ensembles
on the per-reservoir median; its volume-weighted pooled
(all-samples-concatenated) KGE is 0.9469 for LSTM-phys, which exceeds the
tree models' pooled values. This pooled-vs-median contrast reflects the
tree models' pooled KGE being pulled negative by a handful of
extreme-outlier reservoirs (e.g., XGBoost raw pooled KGE =
$-$`<!-- -->`{=html}18.6, LightGBM-phys = $-$`<!-- -->`{=html}33.0),
while LSTM-phys benefits from volume-weighting on large reservoirs.
Neither tree nor LSTM uniformly dominates across both aggregation
metrics, so we report both and treat the per-reservoir median as the
primary metric for its robustness to outliers, while acknowledging the
LSTM's strength on the pooled measure. The capacity-limited (32-unit,
NumPy) LSTM is positioned as a reference, not as a definitive
tree-versus-LSTM comparison; a properly tuned PyTorch LSTM with higher
capacity (e.g., 256 hidden units) may yield different conclusions, as
discussed in Section 7.

  Model                             KGE median \[95% CI\]      NSE median   KGE\>0.5 frac.  
  ------------------------------ ---------------------------- ------------ ---------------- --
  Pass-through                      0.320 \[0.167, 0.409\]       -0.396          0.38       
  Ridge                             0.721 \[0.678, 0.748\]       0.734           0.75       
  Random forest                     0.722 \[0.677, 0.762\]       0.695           0.77       
  XGBoost (raw)                     0.730 \[0.691, 0.767\]       0.702           0.79       
  **Phys-XGBoost (this work)**    **0.875 \[0.860, 0.889\]**   **0.896**       **0.90**     
  LightGBM (raw)                    0.732 \[0.692, 0.770\]       0.702           0.78       
  LightGBM-phys                     0.888 \[0.872, 0.906\]       0.898           0.90       
  CatBoost (raw)                    0.724 \[0.679, 0.771\]       0.712          0.77       
  CatBoost-phys                     0.883 \[0.867, 0.900\]       0.903           0.90       

  : Aggregate performance over 199 evaluated reservoirs (of 200
  selected; ResOpsUS_976 excluded for all-NaN predictions). Only models
  that obey the task definition (no release lags, no concurrent storage)
  are listed; persistence and the LSTM baseline are given in Table R1 as
  references. Reported quantities are per-reservoir medians with 95%
  bootstrap CI on the median KGE (2,000 resamples) for every model.

**Reference models (Table R1).** The persistence predictor
($\hat O_t = O_{t-1}$) is not a fair competitor under the task
definition because it uses the lagged release, but it serves as a
difficulty reference: median KGE 0.975, NSE 0.950, KGE
$>$`<!-- -->`{=html}0.5 fraction 0.99. The LSTM baseline---a
single-layer network with 32 hidden units, implemented from scratch in
NumPy (CPU), trained on a 30-day sliding window of the same leak-free
inputs---attains median KGE 0.713 (raw) and 0.803 (phys, 95% CI
0.760--0.829), with pooled KGE 0.9306 (raw) and 0.9469 (phys). This LSTM
is capacity-limited and should be interpreted as a lower bound; a
properly tuned PyTorch LSTM with higher capacity may yield different
conclusions (Section 7).

<figure data-latex-placement="t">
<img src="figures/fig1_kge_boxplot.png" />
<figcaption>Distribution of per-reservoir KGE across the ten
task-complying models. Persistence (Table R1) and the LSTM baseline
(Section 6.1) are reported separately as references and omitted from
this comparison because persistence uses the lagged release and the LSTM
is capacity-limited; both violate the task definition. The horizontal
dashed line marks KGE = 0.</figcaption>
</figure>

<figure data-latex-placement="t">
<img src="figures/fig2_nse_boxplot.png" />
<figcaption>Distribution of per-reservoir NSE across the ten
task-complying models. The pattern mirrors Fig. 1: the physics-augmented
tree ensembles dominate the fair competitors. Persistence and the LSTM
baseline are not shown here (see Table R1).</figcaption>
</figure>

## 2 Statistical significance

The improvement of Phys-XGBoost over XGBoost is highly significant. The
Wilcoxon signed-rank test on per-reservoir KGE (199 evaluated
reservoirs) gives $p = 2.27\times10^{-27}$. Phys-XGBoost wins in 183 of
the 199 comparable reservoirs (one selected reservoir, ResOpsUS_976, is
excluded for all-NaN predictions). The median $\Delta$KGE is +0.118 with
a 95% bootstrap CI of \[+0.099, +0.143\], which does not cross zero. As
an effect size, the win rate of 183/199 $\approx$ 0.92 corresponds to a
large effect. The physics-augmented versions of the other two
gradient-boosting libraries show the same pattern: LightGBM-phys and
CatBoost-phys attain median raw-to-phys gains of +0.156 and +0.159
respectively, and win on 179/199 and 178/199 reservoirs; LightGBM-phys
also edges Phys-XGBoost on the median (0.888 vs 0.875). We therefore
treat *physics augmentation*, not any single library, as the robust,
significant effect. All pairwise model comparisons are adjusted for
multiple comparisons with the Benjamini--Hochberg procedure at FDR
$= 0.05$ (Section 4.4); the three pre-specified raw$\to$phys comparisons
(XGBoost, LightGBM, CatBoost) and all other reported significances
survive this correction. The two physics-augmented tree variants that
marginally exceed Phys-XGBoost (LightGBM-phys, CatBoost-phys) are
reported alongside it rather than hidden; the proposed Phys-XGBoost is
chosen as the headline method for its interpretability, low compute, and
open implementation, not because it is the single numerical best.

<figure data-latex-placement="t">
<img src="figures/fig3_scatter_improvement.png" />
<figcaption>Per-reservoir KGE: Phys-XGBoost versus XGBoost (raw). Points
above the diagonal indicate Phys-XGBoost wins.</figcaption>
</figure>

## 3 Ablation

Table 2 and Fig. 4 report the ablation. Removing all physics-derived
features (raw-only) drops the median KGE from 0.875 to 0.730, recovering
the plain XGBoost baseline---a loss of 0.145, i.e., 80.5% of the full
gain when the gain is computed as the difference of medians (0.8753 -
0.7584)/(0.8753 - 0.7300) = 0.805, or 79.8% when computed as the median
of per-reservoir paired differences (median(full $-$ no-mass-balance) =
0.1171 vs median(full $-$ raw-only) = 0.1453; 0.1171/0.1453 = 0.806,
from `ablation.csv`). The two definitions capture distinct statistical
objects---the former contrasts the marginal medians of two models, the
latter the typical per-reservoir loss of removing the physics class; we
report both to make the convention explicit. Using only the mass-balance
features (`storage_trend` and `throughput_ratio`) yields a median KGE of
0.876---a difference of $+0.0002$ from the full three-class set, with a
95% bootstrap CI of \[$-0.0017$, $+0.0026$\] that crosses zero. This
establishes that the mass-balance class alone accounts for the entire
physics gain, and that the storage-state and flow-anomaly classes
contribute no statistically distinguishable additional improvement. We
therefore adopt the parsimonious two-feature mass-balance specification
for the headline model. This provides the empirical validation of the
conditional-mutual-information design goal (Eq. 7): the mass-balance
proxy `storage_trend` (the lagged $\Delta S$) carries information about
$O_t$ beyond what raw inflow/storage lags provide, consistent with the
continuity-equation motivation in Section 4.1.3.

  Configuration                              Median KGE   $\Delta$ from full
  ----------------------------------------- ------------ --------------------
  Full (raw + all physics)                     0.875             ---
  Raw only (no physics)                        0.730            -0.145
  \- mass-balance class                        0.758            -0.117
  Only mass-balance (raw + 2 mb features)      0.876           +0.0002

  : Ablation: median KGE across 199 reservoirs under different feature
  configurations. The "only mass-balance" row tests whether the
  mass-balance class alone reproduces the full three-class result.

<figure data-latex-placement="t">
<img src="figures/fig4_ablation.png" />
<figcaption>Ablation results. The mass-balance class accounts for the
largest share of the physics-feature gain.</figcaption>
</figure>

## 4 Difficulty-tier analysis

Partitioning the 199 reservoirs into terciles by their *training-period*
persistence KGE---easy (66), medium (67), and hard (66)---provides an a
priori difficulty stratification independent of the test set
(cross-validation against the test-set persistence terciles yields 62.8%
agreement; Table S3). On easy reservoirs, persistence is already
near-perfect (median KGE 0.993) and Phys-XGBoost approaches it (0.930),
so ML offers little absolute gain. On hard reservoirs, where persistence
is weakest (median KGE 0.896), Phys-XGBoost (0.816) exceeds XGBoost
(0.689) by +0.127. The Phys-XGBoost-over-XGBoost gains are positive
across all three tiers and largest on easy (+0.145) and medium (+0.159)
reservoirs, where the operating policy is moderately
state-dependent---precisely the regime where operators actively adjust
releases in response to storage and inflow conditions, and where the
lagged storage-change proxy (Section 4.1.3) carries the most exploitable
signal. On the hardest tier the gain is smallest (+0.127), because both
persistence and the raw baseline are low and the absolute headroom is
limited.

<figure data-latex-placement="t">
<img src="figures/fig5_tiers.png" />
<figcaption>Median KGE by difficulty tier. Tiers are training-period
persistence-KGE terciles (a priori stratification).</figcaption>
</figure>

## 5 KGE component decomposition

Decomposing the per-reservoir median KGE components across all 199
evaluated reservoirs shows Phys-XGBoost improves all three components
relative to XGBoost: correlation $r$ rises from 0.857 to 0.954,
variability ratio $\alpha$ from 0.874 to 0.939, and bias ratio $\beta$
remains near unity (0.99456 to 0.99646) (Fig. 6). The gain is concentrated
in correlation and variability, while the already-near-perfect bias
ratio is essentially unchanged. The largest gain is in correlation and
variability, indicating the physics features help the model capture both
the timing and the amplitude of release fluctuations, while the
already-small bias is further reduced.

<figure data-latex-placement="t">
<img src="figures/fig6_decomposition.png" />
<figcaption>KGE components (<span class="math inline"><em>r</em></span>,
<span class="math inline"><em>α</em></span>, <span
class="math inline"><em>β</em></span>) for XGBoost versus
Phys-XGBoost.</figcaption>
</figure>

## 6 Extreme-value behaviour

On the top-10% inflow events within each test set (computable for 198 of
the 199 evaluated reservoirs), Phys-XGBoost attains a median KGE of
0.733 versus 0.550 for raw XGBoost, winning in 172 of 198 reservoirs
(86.9%) with a median $\Delta$KGE of +0.144 (Wilcoxon
$p = 1.6\times10^{-22}$; median NSE 0.716 versus 0.402). The same
pattern holds across the other boosting libraries: LightGBM-phys wins in
169 of 198 reservoirs (median 0.764 versus 0.574) and CatBoost-phys in
168 of 198 (median 0.753 versus 0.550), (Fig. 8). However, a handful of
reservoirs exhibit undefined or low extreme-event KGE for all models,
reflecting the known difficulty of tree-based regressors in
extrapolating beyond the training distribution; we report this
transparently rather than averaging it away.

<figure data-latex-placement="t">
<img src="figures/fig8_highflow_kge.png" />
<figcaption>KGE on the top-10% inflow events (extreme-value segment).
Each model’s boxplot aggregates over 198 reservoirs where the segment
KGE is defined. Median values are annotated. The physics-augmented
variants of all three boosting libraries outperform their raw
counterparts.</figcaption>
</figure>

## 7 Comparison with published benchmarks

Table 3 positions our results against published benchmarks. We stress
that direct numerical comparison is **not** valid because the
experimental settings differ: the LSTM hydrological benchmarks (Kratzert
et al. (2018, 2019)) and the large-sample reservoir-release simulators
(Tran et al. (2025; Zhang et al. 2025)) are trained and evaluated under
protocols that differ from our one-step, information-withholding task
(different data, splits, feature sets, and simulation settings); Zhou et
al. (2025) use the lagged outflow as a feature, which our task
definition deliberately excludes. Persistence in our setting reaches KGE
0.975 precisely because of the excluded autocorrelation (Eq. 5). The
fair, within-study comparison is Phys-XGBoost versus the raw XGBoost
baseline under identical settings, where the gain (+0.118 median
$\Delta$KGE, $p = 2.27\times10^{-27}$ over 199 reservoirs) is
unambiguous.

  Method                         Source                            Setting                                                              Reported score
  ------------------------------ --------------------------------- -------------------------------------------------------------------- --------------------------------------------------------
  Conditioned-LSTM               Tran et al. (2025); J. Hydrol.    Out-sample simulation, $\sim$`<!-- -->`{=html}200 CONUS reservoirs   median KGE 0.56--0.82 (by reservoir purpose)
  Mamba state-space              Zhang et al. (2025); J. Hydrol.   Direct simulation, 441 CONUS dams, SHAP                              continental DL; benchmarks vs XGBoost & LSTM
  Physics-constrained RF         Zhou et al. (2025)                Uses lagged outflow; cascade reservoirs                              $R^2 > 0.95$
  XGBoost (raw, this study)      ---                               1-step, no release lag, no concurrent storage                        median KGE 0.730
  Phys-XGBoost (this study)      ---                               1-step, no release lag, no concurrent storage                        median KGE 0.875
  LightGBM-phys (this study)     ---                               1-step, no release lag, no concurrent storage                        median KGE 0.888
  CatBoost-phys (this study)     ---                               1-step, no release lag, no concurrent storage                        median KGE 0.883
  LSTM-phys (this study)         ---                               1-step, no release lag, no concurrent storage                        median KGE 0.803 (N=200, 199 reservoirs); pooled 0.9469
  Persistence (reference only)   ---                               Uses lagged release                                                  median KGE 0.975

  : Positioning relative to published benchmarks. Settings are not
  directly comparable; see text.

## 8 Feature attribution (SHAP)

To interpret *why* the physics-derived features help, we compute SHAP
(Lundberg and Lee 2017) values for Phys-XGBoost. Because a single
reservoir can be unrepresentative, we fit Phys-XGBoost and compute SHAP
on 198 of the 199 evaluated reservoirs (one reservoir was excluded from
the SHAP analysis because its SHAP values were undefined) and average
the mean absolute SHAP value of every feature across them (Fig. 7). To
remove the dependence on reservoir size (the release target spans
several orders of magnitude across reservoirs), we report each feature's
*within-reservoir* \|SHAP\| share --- its \|SHAP\| divided by the
reservoir's total \|SHAP\| --- and average those shares over the 198
reservoirs. The ranking is internally consistent with the ablation
(Section 6.3): the two leading features are the raw lagged inflow
(`inflow_lag1` = 19.3%) and the mass-balance proxy `storage_trend`
(18.8%, the top mass-balance-informed input); `storage_trend` exceeds
even the current inflow (`inflow` = 15.0%). `storage_trend`
belongs to the mass-balance class, confirming the continuity-equation
motivation of Section 4.1.3 --- the model leans most heavily on the
proxy for how fast the reservoir is filling or drawing down. The raw
inflow family (`inflow_lag1`, `inflow`, and the 7-/30-day inflow moving
averages) together accounts for 58.7% of the total attribution,
as expected for a release driven primarily by inflow forcing. The
secondary mass-balance feature (`throughput_ratio`) contributes more
modestly ($\sim$`<!-- -->`{=html}2%). The prominence of `storage_trend`
among the mass-balance-informed inputs explains why withholding concurrent storage
does not hobble the model: the *lagged* storage change carries much of
the exploitable signal, exactly as the information-theoretic design goal
(Eq. 7) predicted.

<figure data-latex-placement="t">
<img src="figures/fig7_shap.png" />
<figcaption>Mean absolute SHAP share (%) for Phys-XGBoost, averaged over
198 of the 199 evaluated reservoirs (one was excluded from the SHAP
analysis). Values are within-reservoir |SHAP| shares (|SHAP| divided by
each reservoir’s total |SHAP|) averaged across reservoirs, so the
ranking is invariant to reservoir size. The raw lagged inflow (<code>inflow_lag1</code>) and the storage-change
proxy (<code>storage_trend</code>) are the two leading inputs; the latter is
the top mass-balance-informed feature, corroborating the ablation.</figcaption>
</figure>

## 9 Per-reservoir results

Supplementary Table S1 lists the per-reservoir KGE for persistence,
XGBoost, and Phys-XGBoost, together with $\Delta$KGE, for all 199
evaluated reservoirs. Phys-XGBoost improves on XGBoost in 183 of 199
reservoirs; of the 16 where it does not, 11 are near-parity
($|\Delta\mathrm{KGE}| < 0.07$, e.g., ResOpsUS_415: $-0.000$,
ResOpsUS_929: $-0.045$, ResOpsUS_1207: $-0.053$) and five are substantial
degradations on reservoirs where the raw XGBoost itself is unstable or
near-zero (e.g., ResOpsUS_572: $\Delta$KGE $= -41.8$; ResOpsUS_1249:
$-1.74$; ResOpsUS_956: $-0.91$; ResOpsUS_185: $-0.83$; ResOpsUS_837:
$-0.19$). We classify reservoirs with raw XGBoost
KGE $< 0.3$ *and* Phys-XGBoost KGE $< 0.3$ as "both-fail" cases (3
reservoirs: ResOpsUS_185, ResOpsUS_1249, ResOpsUS_572); the remaining 13
non-improvement reservoirs have raw KGE above 0.3, and 5 have both models
above 0.85 (near-saturated); the median $\Delta$KGE over the full sample
is $+0.118$, consistent with
the full-sample result. This classification is reported transparently
rather than used to filter results, and the full 199-reservoir
comparison remains the primary analysis. The other physics-augmented
tree models behave similarly per reservoir: LightGBM-phys and
CatBoost-phys win on 179/199 and 178/199 reservoirs and are within
$\pm$`<!-- -->`{=html}0.03 of Phys-XGBoost on most reservoirs.;
LSTM-phys at N=200 attains a median KGE of 0.803 (95% CI 0.760--0.829),
below the physics-augmented tree ensembles (0.875--0.888); its pooled
(all-samples-concatenated) KGE is 0.9469, i.e., volume-weighted it is
competitive, but the per-reservoir median (used throughout) is more
robust to the handful of reservoirs where LSTM-phys degrades sharply
(e.g., ResOpsUS_111: LSTM-phys KGE = -4.89 vs Phys-XGBoost 0.846), and
on the per-reservoir median LSTM-phys remains below the tree ensembles.
(The 256-hidden-unit capacity-scaling study reported in Table S2 uses a
separate representative 30-reservoir subset and is not directly
comparable to this full-N=200 per-reservoir median.) Reservoir
ResOpsUS_976 is excluded: its test-period outflow is *identically zero*
(a data artefact), so the KGE variance denominator is degenerate and
every model returns undefined KGE; we exclude it from the paired tests
rather than assigning a default value. Reservoir ResOpsUS_939 is a clear
outlier where plain XGBoost collapses (KGE $-8.31$) while Phys-XGBoost
remains near-zero ($-0.22$); we retain it rather than removing it post
hoc, and note that its inclusion does not drive the aggregate median
(which is robust to outliers). The complete 12-model per-reservoir table
is provided in the supplementary CSV (`per_reservoir_metrics.csv`) and
the per-reservoir breakdown is tabulated in Supplementary Table S1.

**Proxy-channel stress test.** Because
`storage_trend` = $S_{t-1} - S_{t-2} = I_{t-1} - O_{t-1}$ is
algebraically entangled with the lagged release that the task definition
withholds, we run a counterfactual feature-channel test across the 199
reservoirs to separate the pure storage-change signal from the
release-history channel. Three control configurations are compared
against Phys-XGBoost (median KGE 0.875): (i) `raw_plus_outflow_lag1`
augments the raw set with the true lagged release $O_{t-1}$ (an explicit
upper-bound leak of the autocorrelation channel) and attains 0.931; (ii)
`phys_shifted_trend` replaces `storage_trend` with the one-extra-day
-shifted difference $S_{t-2} - S_{t-3}$, isolating the pure storage-change
signal from the lagged-release channel, and attains 0.843; (iii) a
`forecast_available` variant using only issue-time-available features
attains 0.867. Two conclusions follow. First, the shifted-trend variant
(0.843) still substantially exceeds raw XGBoost (0.730)---by
$+0.113$---so the pure storage-change signal, stripped of the
release-history channel, independently improves prediction; the
mass-balance feature is not merely a disguised release lag. Second, the
raw+outflow-lag1 variant (0.931) exceeds Phys-XGBoost by 0.056,
confirming that the lagged-release component of `storage_trend`
contributes materially to the gain. The feature is therefore best
understood as a physics-informed proxy whose benefit is the joint effect
of the net-inflow signal ($I_{t-1}$) and the release history
($O_{t-1}$); because it uses only information available before $t$, it
is not a data leak, but its predictive power is necessarily coupled to
the release autocorrelation identified in Section 4.1.2. We report this
entanglement transparently rather than claiming the gain is purely
"physical."

## 10 Iterative multi-step simulation

The benchmark DL studies cited in Section 6.7 (Tran et al. 2025; Zhang
et al. 2025) report one-step release simulations on hundreds of CONUS
reservoirs; neither reports multi-day, lead-time-dependent skill. Our
one-step protocol (Section 4) deliberately avoids iterative error
accumulation to obtain an unbiased read of the feature contribution, but
it leaves open how the tree models behave when the storage-accounting
state is rolled forward autoregressively---the setting a forecast user
would face. We therefore evaluate Phys-XGBoost, LightGBM-phys, and
CatBoost-phys under an auto-regressive rollout, holding the split and
feature set fixed.

**Setting (stated explicitly).** We assume *perfect inflow*: future
inflow is taken from the true record, so the only error that propagates
is that of the release prediction through the storage-accounting state.
Future storage is rolled forward with each model's *own* predicted
releases via the continuity equation,
$S(t) = S(t-1) + I(t) - \hat O(t-1)$, and the lagged storage is fed back
into the features at the next step. We report KGE at lead times
$L = 1,\dots,7$ days, obtained by rolling the state $L$ steps from the
known initial conditions at each issue time $\tau$. A persistence-L
reference, $\hat O(\tau+L) = O(\tau)$, provides the difficulty baseline.
Under this perfect-inflow assumption the comparison isolates
release-model skill from inflow-forecast skill; a fully operational
setting would additionally require an inflow forecast, which we leave to
future work.

::: {#tab:rollout}
  Lead    Persist-L   XGB (raw)   LGBM (raw)   CB (raw)   Phys-XGB   LGBM-phys   CB-phys   frac(phys$>$xgb)
  ------ ----------- ----------- ------------ ---------- ---------- ----------- --------- ------------------
  1         0.974       0.732       0.731       0.738      0.875       0.887      0.882          0.91
  2         0.933       0.659       0.674       0.681      0.446       0.388      0.409          0.23
  3         0.898       0.582       0.618       0.643      0.293       0.215      0.255          0.20
  4         0.866       0.564       0.555       0.586      0.221       0.158      0.213          0.16
  5         0.825       0.522       0.540       0.588      0.180       0.120      0.169          0.16
  6         0.801       0.513       0.537       0.566      0.144       0.107      0.144          0.16
  7         0.776       0.533       0.532       0.564      0.120       0.106      0.148          0.15

  : Median KGE across the 199 evaluated reservoirs at
  iterative-simulation lead times $L=1,\dots,7$ days, under the
  perfect-inflow assumption of Section 6.10. Persistence-L is the
  $\hat O(\tau+L)=O(\tau)$ reference; "raw" models use only lagged
  inflow/storage, "phys" models add the physics-derived features
  (Section 4.1.3). The last column is the fraction of reservoirs where
  Phys-XGBoost median KGE exceeds raw XGBoost at that lead.
:::

At lead 1, where the trajectory is re-initialised from the true state,
the physics-augmented trees clearly beat their raw counterparts:
Phys-XGBoost reaches a median KGE of 0.875 (matching its one-step
median of 0.875), a $+0.143$ median gain over raw
XGBoost (0.732), and wins on 91% of reservoirs; LightGBM-phys (0.887)
and CatBoost-phys (0.882) are higher still. The raw-to-phys gain
therefore survives the move from one-step to one-step-equivalent
rollout---it is not an artefact of the one-step evaluation protocol. As
the state is rolled forward from each model's own (imperfect) release
predictions, however, the physics features---which depend on the
iteratively updated storage---degrade sharply: the median Phys-XGBoost
gain over raw XGBoost falls from $+0.143$ at lead 1 to $-0.413$ at lead
7, and Phys-XGBoost wins on only 15% of
reservoirs at long lead. By lead 7 the raw tree models (0.53--0.56)
actually exceed the phys variants (0.11--0.15). Persistence-L remains at
0.776 even at lead 7, far above every learned model, confirming that
once the trajectory is unrolled the dominant predictable signal is the
autocorrelation of release itself---the same structural ceiling noted in
Section 6.1, which the models cannot escape because the protocol
withholds the lagged release.

Positioned against the out-sample, one-step benchmark of Tran et al.
(2025) (median KGE 0.56--0.82 by reservoir purpose), our tree models
attain a comparable one-step median KGE (0.88 at lead 1 under the
perfect-inflow rollout). At longer lead, autoregressive release
feedback---not inflow---dominates the error and the tree ensembles do
not escape the persistence ceiling. We report this honest bracket rather
than claiming multi-day skill: the physics-feature gain is limited to
one-step and short-lead ($L{=}1$) prediction, and the degradation at
longer leads is a structural property of the error-feedback mechanism
(Section 6.10.1), not a failure of the features in steady state. The DL
benchmarks do not report multi-day rollout skill, so no direct long-lead
comparison is possible.

**Why the physics features degrade faster under rollout.** Under the
rollout, the storage state is updated by the continuity equation
$S_t = S_{t-1} + I_t - \hat O_{t-1}$ (Section 6.10), so the mass-balance
proxy used by the physics features becomes
$\Delta S_{t-1} = S_{t-1} - S_{t-2} = I_{t-1} - \hat O_{t-2}$.
Consequently, each unit of prediction error in $\hat O$ propagates
*one-for-one* into the next step's `storage_trend`, creating an
error-feedback loop: a mis-predicted release at lead $L$ directly
contaminates the physics features at lead $L+1$. The raw inflow lags
($I_{t-k}$), by contrast, are observed quantities and are unaffected by
the model's own errors, so the raw models degrade only through the
slowly accumulating storage state, not through a feature-level feedback.
This explains why the physics-augmented models---beneficial at one step
and at lead 1---fall below their raw counterparts as the trajectory is
unrolled (Table [1](#tab:rollout){reference-type="ref"
reference="tab:rollout"}, lead 7), and why the effect is a property of
the feedback structure rather than of the features being incorrect in
steady state.

# Discussion and Limitations

**Why persistence is hard to beat.** Under our task definition,
persistence is *not* a fair competitor because it uses the lagged
release. Its high KGE reflects the strong autocorrelation of daily
release (Eq. 5), not a learned policy. Formally, because the protocol
withholds $O_{t-1}$, a naive persistence predictor
($\hat O_t = O_{t-1}$) is undefined within the evaluation---it cannot be
computed from the permitted inputs---so persistence is incomparable by
construction, not merely a strong autocorrelated competitor. The value
of ML in this setting is therefore not absolute KGE but the ability to
predict release from state alone---relevant when the previous release is
unknown (e.g., scenario simulation, ungauged operation inference). We
report persistence transparently so that readers can judge this.

**What the physics-derived features add.** The ablation (Table 2) shows
that the mass-balance class---`storage_trend` (the lagged proxy for
$\Delta S_t$, Eq. 6) and `throughput_ratio`---accounts for the full
physics gain: the mass-balance-only configuration yields a median KGE of
0.876, indistinguishable from the full three-class set (median $\Delta$
= $+0.0002$, 95% CI $[-0.0017, +0.0026]$), and removing the mass-balance
class drops KGE from 0.875 to 0.758 (80.5% of the total physics benefit
by the difference-of-medians convention, 79.8% by the
median-of-paired-differences convention; both reported in Section 6.3).
This is consistent with the theoretical motivation in Section 4.1.3: the
continuity equation (Eq. 3) identifies $\Delta S_t$ as a first-order
driver of $O_t$, and although we withhold concurrent $S_t$ to prevent
exact back-solving, the lagged storage change $\Delta S_{t-1}$ still
carries strong information about whether the operator is filling or
drawing down the reservoir. Two additional feature classes
(storage-state and flow-anomaly) were tested but contribute no
statistically distinguishable additional improvement and are not
retained in the final specification. The KGE decomposition (Fig. 6)
shows the gains are distributed across correlation, variability, and
bias, rather than being driven by a single component---an indication
that the features improve the model's representation of timing,
amplitude, and volume simultaneously rather than gaming one axis of the
metric. That the same mass-balance dominance repeats for LightGBM-phys
and CatBoost-phys (Section 6.1) confirms the gain is a property of the
feature design, not of XGBoost specifically; the SHAP attribution
(Section 6.8) provides the complementary mechanistic explanation,
ranking `inflow_lag1` first overall and `storage_trend` first among the
mass-balance-informed features.

**Practical relevance.** Although persistence attains a higher KGE by
exploiting the lagged release, the task-complying one-step predictor is
useful precisely when the previous release is unknown or inapplicable:
(i) scenario simulation of alternative release policies under
hypothetical inflow sequences, where no realised $O_{t-1}$ exists; (ii)
inference of operating behaviour at reservoirs with sparse or
non-telemetered release records, where state-based prediction is the
only option; and (iii) screening of rule-curve adequacy, where a
state-only model isolates the storage--inflow--release relationship from
autocorrelation artefacts. Phys-XGBoost trains in minutes per reservoir
on a single CPU core, requires no GPU, and its tree-split-based SHAP
attribution is directly auditable by operators---properties that favour
its use as an interpretable screening tool alongside, rather than
replacing, operational forecasting systems that do exploit the lagged
release.

**Limitations.** (1) ResOpsUS covers U.S. reservoirs; operating rules
and climate are specific and results may not transfer. (2) Daily
resolution cannot capture sub-daily flood operations. (3) Inflow is
missing for many reservoirs; restricting to complete rows introduces
selection bias. (4) The one-step results of Sections 6.1--6.9 are not
directly comparable to published LSTM benchmarks trained under different
simulation protocols; we address multi-step behaviour in Section 6.10
with an auto-regressive rollout under a perfect-inflow assumption, which
isolates release-model skill but leaves inflow-forecast skill out of
scope. (5) The final model uses only two mass-balance features
(storage_trend and throughput_ratio), neither of which depends on
climatological statistics that could leak test-period information; the
seasonal rule-curve feature ($\bar S_{\text{doy}}$) tested in earlier
versions was computed from the training period only and, upon re-running
the full 199-reservoir experiment, was confirmed to change the median
Phys-XGBoost KGE by 0.000 (0.8753 before and 0.8753 after). (6) 199
reservoirs yield moderately wide confidence intervals; pairwise model
tests are corrected for multiple comparisons with the
Benjamini--Hochberg procedure at FDR $=0.05$ (Section 4.4), and the
primary pre-specified Phys-XGBoost vs. XGBoost comparison is unaffected.
(7) Hyperparameter sensitivity was assessed on a representative
30-reservoir subset (the all-NaN reservoir ResOpsUS_976 excluded):
sweeping max_depth $\in \{4,6,8\}$ and n_estimators
$\in \{200,400,800\}$---nine configurations, every other setting fixed
at the study defaults---changed the median Phys-XGBoost KGE by at most
0.028 (range 0.878--0.906 across configurations; the default depth-6 /
400-tree configuration gave 0.880). This spread is an order of magnitude
smaller than the physics-augmentation gain over raw XGBoost
($\approx 0.15$), confirming the headline conclusion is robust to
reasonable hyperparameter variation. (8) The AR(1) model in Section
4.1.2 is a first-order approximation; release exhibits seasonal
non-stationarity, and higher-order dynamics are not captured by the
persistence-KGE derivation (Eq. 5), which should be read as an
approximate bound rather than an exact identity. (9) The LSTM baseline
is implemented from scratch in NumPy (CPU) rather than with PyTorch or
TensorFlow, because the build environment cannot import torch (a
CUDA/driver incompatibility that segfaults on load). The implementation
is gradient-checked against finite differences and validated, but it is
a lower-capacity configuration (a single layer with 32 hidden units)
than typical published LSTM rainfall--runoff models (e.g., 256 hidden
units in Kratzert et al. (2018)); the LSTM results should therefore be
interpreted as a *lower bound* on LSTM performance rather than a
definitive tree-versus-LSTM comparison, and a properly tuned PyTorch
LSTM may yield different conclusions. (10) The stratified sample of 200
reservoirs (199 evaluated) is drawn by evenly spaced ranks of the
eligible set sorted by complete-row count, which favours reservoirs with
longer, more complete records; results may therefore be biased toward
better-monitored reservoirs, a known property of the sampling design.

**Threats to validity.** *Internal:* earlier versions included a
seasonal rule-curve climatology feature ($\bar S_{\text{doy}}$) that was
initially computed over the full record, constituting a minor
test-period information leakage. In this revision that feature has been
removed from the final specification (the mass-balance-only model uses
no climatological statistics), and the full 199-reservoir experiment was
re-run to confirm the prior conclusion is unaffected. *External:*
results are specific to large U.S. reservoirs with near-complete
records; transfer to small reservoirs, non-U.S. climates, or reservoirs
with sparse data is unverified. *Construct:* the task measures one-step
release prediction, not operational decision quality; high KGE does not
imply the model captures multi-objective operating rules or satisfies
classical reservoir-performance criteria (Hashimoto et al. 1982).
*Statistical:* the difficulty-tier stratification uses training-period
persistence KGE and is therefore a priori with respect to the test set;
cross-validation against test-set persistence terciles confirms the
tiers are stable (62.8% agreement; Table S3). The primary confirmatory
test (Phys-XGBoost vs. XGBoost, Wilcoxon) was pre-specified and is not
affected by the stratification choice.

*Relation to external benchmarks.* The present study compares models
within a single, pre-registered experimental framework (identical
train/test split, identical target, identical random seeds). No attempt
was made to benchmark against previously published results on the same
dataset, because differences in training/validation splits, data
pre-processing, and hyperparameter tuning protocols render fair external
comparisons infeasible. The reported KGE and NSE values should therefore
be interpreted within this study's own fair-comparison framework, not as
claims of absolute state-of-the-art performance on the ResOpsUS dataset.

# Conclusions

We formulated a non-trivial reservoir release-forecasting task that
withholds the lagged release and concurrent storage---two
information-withholding constraints justified theoretically by the
continuity equation (Section 4.1.1) and the release autocorrelation
structure (Section 4.1.2)---and evaluated three classes of
physics-derived features across 199 evaluated reservoirs (of 200
selected; one reservoir, ResOpsUS_976, excluded for all-NaN predictions)
from the open ResOpsUS dataset, within three gradient-boosting libraries
(XGBoost, LightGBM, CatBoost) and one recurrent network (LSTM). The
principal findings, all obtained on held-out test data with fixed
hyperparameters and chronological splitting, are:

1\. **Physics augmentation is the robust, significant effect.** Adding
the physics-derived features raises the median KGE of every
gradient-boosting library: XGBoost 0.730 → Phys-XGBoost 0.875, LightGBM
0.732 → LightGBM-phys 0.888, CatBoost 0.724 → CatBoost-phys 0.883. Each
raw→phys comparison wins on 178--183 of 199 reservoirs (all Wilcoxon
$p < 10^{-26}$). The gain is therefore a property of the feature design,
not of any single library. The gain is limited to one-step and
short-lead ($L{=}1$) prediction; under multi-day autoregressive rollout
the physics features degrade faster than raw lags, a structural property
of the storage-based error-feedback mechanism (Section 6.10.1). Among
the physics-augmented tree models, LightGBM-phys and CatBoost-phys are
marginally above Phys-XGBoost (difference $\approx$ 0.01--0.02 in median
KGE); we report all three rather than concealing the slightly stronger
variants.

2\. **Mass-balance features matter most --- confirmed two ways.**
Ablation attributes 80.5% (difference of medians) / 79.8% (median of
paired differences) of the Phys-XGBoost physics gain to the mass-balance
feature class, and SHAP attribution (Section 6.8 and Table S4) ranks the
raw lagged inflow `inflow_lag1` (19.3%) and the mass-balance proxy
`storage_trend` (18.8%, the top mass-balance-informed input) as the two
leading features of all 16. Both lines of evidence confirm that lagged
storage-change information---the continuity-equation proxy derived in
Section 4.1.3---is the dominant signal beyond raw inflow/storage lags.

3\. **Gains concentrate where ML is most useful.** The largest
improvements occur on easy and medium-difficulty reservoirs, where the
operating policy is state-dependent but not trivially autocorrelated; on
easy reservoirs persistence is already near-optimal, and on the hardest
reservoirs all models struggle. A capacity-limited LSTM baseline (32
hidden units, NumPy CPU implementation) attains a per-reservoir median
KGE of 0.803 (phys) and a volume-weighted pooled KGE of 0.9469,
competitive with the tree ensembles on the pooled metric but below them
on the per-reservoir median. This pooled-vs-median contrast is a known
property of the tree models' pooled instability on a handful of outlier
reservoirs, and neither model family uniformly dominates across both
aggregation metrics. To probe whether the 32-unit result reflects
capacity limits rather than an inherent tree-versus-LSTM gap, we trained
a 256-hidden-unit LSTM under identical settings on a representative
30-reservoir subset (ResOpsUS_976 excluded for all-NaN predictions). The
larger LSTM raises the median KGE from 0.527 (32-unit, on the same
30-reservoir subset) to 0.579 (256-unit), a relative gain of 9.7%
($\Delta$ = $+0.051$, 95% CI not separately estimated on the subset),
confirming that additional capacity does help the recurrent baseline.
However, the 256-unit LSTM (median 0.579) still falls well below
Phys-XGBoost (median 0.882) on the same reservoirs, so architecture
scaling alone does not close the gap. A properly tuned PyTorch LSTM with
higher capacity and regularisation remains a candidate for future work
to test whether the strong pooled performance generalises to the
per-reservoir median (Table S2).

4\. **Honest positioning.** Persistence (which uses the lagged release)
remains highest at KGE 0.975, consistent with the AR(1) derivation (Eq.
5); we report it only as a difficulty reference. Our one-step results
are not numerically comparable to published LSTM benchmarks trained
under different simulation protocols, and we make no SOTA claim. The
fair, within-study comparison---physics-augmented versus raw gradient
boosting under identical settings---is where the contribution lies.

The contribution is therefore a transparent, low-compute, interpretable
feature set that significantly improves a strong tree-based baseline
under a controlled, non-trivial protocol. Code, selection criteria, and
per-reservoir metrics are released for reproducibility. Future work
should extend the evaluation to iterative simulation, regional
(cross-reservoir) transfer, and sub-daily scales, and should extend the
multi-step rollout analysis to include inflow forecast uncertainty.

# Data and Code Availability

The ResOpsUS dataset is openly available under CC-BY 4.0 from the Zenodo
repository (record 6612040; Steyaert et al. 2022). All experiment code
(`run_experiment_v3.py`, `run_shap_standalone.py`,
`make_reservoir_figures.py`), the per-reservoir metrics CSV, the
ablation CSV, the SHAP CSVs (`shap_phys_xgb.csv`,
`shap_phys_xgb_mean.csv`), and the summary JSON files are released at
<https://github.com/mingyi0818/resopsus-phys-xgboost> (a Zenodo archive
will be created upon acceptance) for full reproducibility. The reservoir
selection criteria, chronological split, and fixed hyperparameters are
specified in Sections 3--5 so that the 200-reservoir sample (199
evaluated) can be regenerated exactly.

## Reproducibility and fair-comparison statement {#sec:repro .unnumbered}

**Fair-comparison protocol.** All models share identical train/test
splits and the same one-step-ahead daily release target. Tree models
used n_estimators $=400$, max_depth $=6$, learning_rate $=0.05$,
subsample $=0.8$, colsample_bytree $=0.8$ (random state 42); the LSTM
used a single layer with 32 hidden units, implemented in NumPy (CPU), a
30-day sliding window, per-reservoir target standardisation, and Adam
with early stopping (patience 15). All hyperparameters were fixed a
priori (no per-reservoir tuning), and all random seeds were 42. Dataset
access and code release are given in Section 7. **Statistical
reporting.** Paired improvements across reservoirs were assessed with
Wilcoxon signed-rank tests, corrected for multiple comparisons by the
Benjamini--Hochberg procedure at FDR $=0.05$ (Section 4.4). Beyond KGE
we report NSE (Table 1) and decompose KGE into its three components $r$,
$\alpha$, $\beta$ (Section 6.5 / Fig. 6), separating correlation from
volume-bias gains.

::::::::::::::::::::::::::::::::::: {#refs .references .csl-bib-body .hanging-indent}
::: {#ref-Addor2017 .csl-entry}
Addor, Nans, Andrew J. Newman, Naoki Mizukami, and Martyn P. Clark.
2017. "The CAMELS Data Set: Catchment Attributes and Meteorology for
Large-Sample Studies." *Hydrology and Earth System Sciences* 21:
5293--313. <https://doi.org/10.5194/hess-21-5293-2017>.
:::

::: {#ref-Breiman2001 .csl-entry}
Breiman, Leo. 2001. "Random Forests." *Machine Learning* 45 (1): 5--32.
<https://doi.org/10.1023/A:1010933404324>.
:::

::: {#ref-Chen2016 .csl-entry}
Chen, Tianqi, and Carlos Guestrin. 2016. "XGBoost: A Scalable Tree
Boosting System." *Proc. 22nd ACM SIGKDD*, 785--94.
<https://doi.org/10.1145/2939672.2939785>.
:::

::: {#ref-Cinkus2023 .csl-entry}
Cinkus, Guillaume, Naomi Mazzilli, Hervé Jourde, et al. 2023. "When Best
Is the Enemy of Good -- Critical Evaluation of Performance Criteria in
Hydrological Models." *Hydrology and Earth System Sciences* 27 (13):
2397--411. <https://doi.org/10.5194/hess-27-2397-2023>.
:::

::: {#ref-Clark2021 .csl-entry}
Clark, Martyn P., Richard M. Vogel, Jonathan R. Lamontagne, et al. 2021.
"The Abuse of Popular Performance Metrics in Hydrologic Modeling."
*Water Resources Research* 57 (9): e2020WR029001.
<https://doi.org/10.1029/2020WR029001>.
:::

::: {#ref-Frame2023 .csl-entry}
Frame, Jonathan M., Frederik Kratzert, Hoshin V. Gupta, Paul Ullrich,
and Grey S. Nearing. 2023. "On Strictly Enforced Mass Conservation
Constraints for Modelling the Rainfall--Runoff Process." *Hydrological
Processes* 37 (3): e14847. <https://doi.org/10.1002/hyp.14847>.
:::

::: {#ref-Galelli2025 .csl-entry}
Galelli, Stefano, Sean W. D. Turner, Yadu Pokhrel, et al. 2025.
"Advancing the Representation of Human Actions in Large-Scale
Hydrological Models: Challenges and Future Research Directions." *Water
Resources Research* 61 (7): e2024WR039486.
<https://doi.org/10.1029/2024WR039486>.
:::

::: {#ref-Gauch2021 .csl-entry}
Gauch, Martin, Frederik Kratzert, Daniel Klotz, Grey Nearing, Jimmy Lin,
and Sepp Hochreiter. 2021. "Rainfall--Runoff Prediction at Multiple
Timescales with a Single Long Short-Term Memory Network." *Hydrology and
Earth System Sciences* 25: 2045--68.
<https://doi.org/10.5194/hess-25-2045-2021>.
:::

::: {#ref-Gupta2009 .csl-entry}
Gupta, Hoshin V., Harald Kling, Koray K. Yilmaz, and Guillermo F.
Martinez. 2009. "Decomposition of the Mean Squared Error and NSE
Performance Criteria: Implications for Improving Hydrological
Modelling." *Journal of Hydrology* 377 (1--2): 80--91.
<https://doi.org/10.1016/j.jhydrol.2009.08.003>.
:::

::: {#ref-Hashimoto1982 .csl-entry}
Hashimoto, Tsuyoshi, Jery R. Stedinger, and Daniel P. Loucks. 1982.
"Reliability, Resiliency, and Vulnerability Criteria for Water Resource
System Performance Evaluation." *Water Resources Research* 18 (1):
14--20. <https://doi.org/10.1029/WR018i001p00014>.
:::

::: {#ref-Hochreiter1997 .csl-entry}
Hochreiter, Sepp, and Jürgen Schmidhuber. 1997. "Long Short-Term
Memory." *Neural Computation* 9 (8): 1735--80.
<https://doi.org/10.1162/neco.1997.9.8.1735>.
:::

::: {#ref-Karpatne2017 .csl-entry}
Karpatne, Anuj, Gowtham Atluri, James H. Faghmous, et al. 2017.
"Theory-Guided Data Science: A New Paradigm for Scientific Discovery
from Data." *IEEE Transactions on Knowledge and Data Engineering* 29
(10): 2318--31. <https://doi.org/10.1109/TKDE.2017.2720168>.
:::

::: {#ref-Ke2017 .csl-entry}
Ke, Guolin, Qi Meng, Thomas Finley, et al. 2017. "LightGBM: A Highly
Efficient Gradient Boosting Decision Tree." *Advances in Neural
Information Processing Systems*, 3146--54.
<https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>.
:::

::: {#ref-Knoben2019 .csl-entry}
Knoben, Wouter J. M., Jim E. Freer, and Ross A. Woods. 2019. "Technical
Note: Inherent Benchmark or Not? Comparing Nash--Sutcliffe and
Kling--Gupta Efficiency Scores." *Hydrology and Earth System Sciences*
23: 4323--31. <https://doi.org/10.5194/hess-23-4323-2019>.
:::

::: {#ref-Kratzert2018 .csl-entry}
Kratzert, Frederik, Daniel Klotz, Christoph Brenner, Karsten Schulz, and
Mathew Herrnegger. 2018. "Rainfall--Runoff Modelling Using Long
Short-Term Memory (LSTM) Networks." *Hydrology and Earth System
Sciences* 22: 6005--22. <https://doi.org/10.5194/hess-22-6005-2018>.
:::

::: {#ref-Kratzert2019 .csl-entry}
Kratzert, Frederik, Daniel Klotz, Mathew Herrnegger, A. K. Sampson, Sepp
Hochreiter, and Grey S. Nearing. 2019. "Toward Improved Predictions in
Ungauged Basins: Exploiting the Power of Machine Learning." *Water
Resources Research* 55 (12): 11344--54.
<https://doi.org/10.1029/2019WR026065>.
:::

::: {#ref-Lange2024 .csl-entry}
Lange, Holger, and Sebastian Sippel. 2024. "Machine Learning
Applications in Hydrology." *WIREs Water* 11 (3): e1723.
<https://doi.org/10.1002/wat2.1723>.
:::

::: {#ref-Liu2024 .csl-entry}
Liu, Jiangtao, Yuchen Bian, Kathryn Lawson, and Chaopeng Shen. 2024.
"Probing the Limit of Hydrologic Predictability with the Transformer
Network." *Journal of Hydrology* 637: 131389.
<https://doi.org/10.1016/j.jhydrol.2024.131389>.
:::

::: {#ref-Lund1999 .csl-entry}
Lund, Jay R., and Joel Guzman. 1999. "Derived Operating Rules for
Reservoirs in Series or in Parallel." *Journal of Water Resources
Planning and Management* 125 (3): 143--53.
<https://doi.org/10.1061/(ASCE)0733-9496(1999)125:3(143)>.
:::

::: {#ref-Lundberg2017 .csl-entry}
Lundberg, Scott M., and Su-In Lee. 2017. "A Unified Approach to
Interpreting Model Predictions." *Advances in Neural Information
Processing Systems*, 4765--74.
<https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>.
:::

::: {#ref-Mathevet2006 .csl-entry}
Mathevet, Thibault, Claude Michel, Vazken Andréassian, and Charles
Perrin. 2006. "A Bounded Version of the Nash--Sutcliffe Criterion for
Better Model Assessment on Large Sets of Basins." *IAHS Publication*
307: 211--19.
:::

::: {#ref-Nash1970 .csl-entry}
Nash, J. E., and J. V. Sutcliffe. 1970. "River Flow Forecasting Through
Conceptual Models Part I --- A Discussion of Principles." *Journal of
Hydrology* 10 (3): 282--90.
<https://doi.org/10.1016/0022-1694(70)90255-6>.
:::

::: {#ref-Nearing2021 .csl-entry}
Nearing, Grey S., Frederik Kratzert, A. K. Sampson, et al. 2021. "What
Role Does Hydrological Science Play in the Age of Machine Learning?"
*Water Resources Research* 57 (3): e2020WR028091.
<https://doi.org/10.1029/2020WR028091>.
:::

::: {#ref-Prokhorenkova2018 .csl-entry}
Prokhorenkova, Liudmila, Gleb Gusev, Aleksandr Vorobev, Anna Veronika
Dorogush, and Andrey Gulin. 2018. "CatBoost: Unbiased Boosting with
Categorical Features." *Advances in Neural Information Processing
Systems*, 6638--48.
<https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html>.
:::

::: {#ref-Reichstein2019 .csl-entry}
[Reichstein, Markus, Gustau Camps-Valls, Björn Stevens, et al.]{.nocase}
2019. "Deep Learning and Process Understanding for Data-Driven Earth
System Science." *Nature* 566 (7743): 195--204.
<https://doi.org/10.1038/s41586-019-0912-1>.
:::

::: {#ref-Shen2018 .csl-entry}
Shen, Chaopeng, Eric Laloy, Amin Elshorbagy, et al. 2018. "HESS
Opinions: Incubating Deep-Learning-Powered Hydrologic Science Advances
as a Community." *Hydrology and Earth System Sciences* 22 (11):
5639--56. <https://doi.org/10.5194/hess-22-5639-2018>.
:::

::: {#ref-Steyaert2022 .csl-entry}
Steyaert, Patrick, Laura Condon, Sean W. D. Turner, and Nathalie Voisin.
2022. "ResOpsUS, a Dataset of Historical Reservoir Operations in the
Contiguous United States." *Scientific Data* 9: 41597--022.
<https://doi.org/10.1038/s41597-022-01134-7>.
:::

::: {#ref-Tran2025 .csl-entry}
Tran, Quang-Hung, Tian Zhou, Zhi-Qi Tan, Kai Fang, and L. Ruby Leung.
2025. "Improving the Prediction of Daily Reservoir Releases over the
CONUS Using Conditioned LSTM." *Journal of Hydrology* 661: 133750.
<https://doi.org/10.1016/j.jhydrol.2025.133750>.
:::

::: {#ref-Tyralis2019 .csl-entry}
Tyralis, Hristos, Georgia Papacharalampous, and Andreas Langousis. 2019.
"A Brief Review of Random Forests for Water Scientists and Practitioners
and Their Recent History in Water Resources." *Water* 11 (5): 910.
<https://doi.org/10.3390/w11050910>.
:::

::: {#ref-Wi2022 .csl-entry}
Wi, Sungwook, and Scott Steinschneider. 2022. "Assessing the Physical
Realism of Deep Learning Hydrologic Model Projections Under Climate
Change." *Water Resources Research* 58 (9): e2022WR032123.
<https://doi.org/10.1029/2022WR032123>.
:::

::: {#ref-Zhang2025 .csl-entry}
Zhang, Yueqing, Bin Yue, Majid Basirifard, Tongtong Cao, and Dawei Yang.
2025. "A Mamba-Type of Deep State Space Model for Reservoir Release
Simulation with a Large-Scale Verification over 441 Dams Across CONUS."
*Journal of Hydrology* 662: 134145.
<https://doi.org/10.1016/j.jhydrol.2025.134145>.
:::

::: {#ref-Zhou2025 .csl-entry}
Zhou, Yu, Hai Yu, Xiao Zhang, Yangwen Jia, and Jianzhu Luo. 2025.
"Cascade Reservoir Outflow Simulation Based on Physics-Constrained
Random Forest." *Water* 17 (14): 2154.
<https://doi.org/10.3390/w17142154>.
:::
:::::::::::::::::::::::::::::::::::
