# 22AIE301 — Probabilistic Reasoning Project Presentation
## Slide-by-Slide Content (Updated for Graph-Bayesian Neural Architecture)

> Copy-paste ready content for each slide in the `22AIE301_Probabilistic_Reasoning_C7.pptx` template.

---

## Slide 1 — Title Slide

**Title:**
> Graph-Bayesian Neural Architecture for Explainable Financial Risk Prediction with Uncertainty Quantification

**Subtitle:**
> Group C7
> *(Add your member names & roll numbers here)*
>
> Course: 22AIE301 — Probabilistic Reasoning
> Semester 5

---

## Slide 2 — Outline of the Presentation

**Content (keep as-is from template):**

1. Introduction
2. Literature Review
3. Takeaways / Research Gaps
4. Objectives
5. Methodology
6. Progress in terms of Project Objectives
7. Weekly Timeline for Project Completion
8. References

---

## Slide 3 — Introduction

**Content:**

- **Problem:** Financial credit risk prediction requires high accuracy, but traditional black-box machine learning models fail to explain *why* a customer is risky and **cannot estimate prediction confidence** (epistemic uncertainty).
- **Why Uncertainty & Structure Matter:** A model predicting "70% default probability" may be highly certain or guessing due to lack of similar historical data. Financial risk officers need calibrated uncertainty ($\sigma^2$) to flag edge-cases for human review.
- **Dataset:** UCI *Default of Credit Card Clients* — 30,000 credit card clients in Taiwan, 23 features (demographics, credit limit, 6-month repayment sequence, bill levels, payment amounts) and binary target `Y` (default next month: Yes/No).
- **Our Solution — Graph-Bayesian Neural Architecture:**
  1. **Graph Neural Network (GNN):** Constructs adaptive customer similarity graphs to capture relational interactions ($H$).
  2. **Bayesian Network (BN):** Discovers interpretable probabilistic dependencies and extracts factor posteriors ($Z$).
  3. **Bayesian Neural Network (BNN):** Fuses embeddings $[H \parallel Z]$ and performs uncertainty-aware discrimination via Monte Carlo sampling ($T=50$).
- **Key Output:** Dual prediction output — **Risk Score ($\hat{y}$)** + **Epistemic Model Uncertainty ($\sigma^2$)**.

---

## Slide 4 — Literature Review

**Content:**

| # | Source | Key Contribution | Relevance to Our Project |
|:--|:-------|:-----------------|:------------------------|
| 1 | Yeh & Lien (2009), *Expert Systems with Applications* | Compared 6 traditional data mining methods on UCI Credit dataset; identified neural networks as best probability estimators | **Original dataset paper** — benchmark baseline and financial problem framing |
| 2 | Kipf & Welling (2017) / Veličković et al. (2018) | Introduced Graph Convolutional Networks (GCN) & Graph Attention Networks (GAT) for relational node embeddings | **Graph encoding layer** — constructs customer similarity graphs and neighborhood features ($H$) |
| 3 | Koller & Friedman (2009), *Probabilistic Graphical Models* | Comprehensive formulation of Bayesian Networks, DAG structure learning, BDeu priors, and exact inference | **PGM theoretical basis** — Stage 1 BN factor group modeling and Variable Elimination |
| 4 | Blundell et al. (2015), *"Weight Uncertainty in Neural Networks"* (ICML) | Developed Bayes by Backprop for learning Gaussian weight posterior distributions $q(w)$ in deep networks | **BNN theoretical foundation** — variational weight learning with KL divergence loss |
| 5 | Gal & Ghahramani (2016), *"Dropout as a Bayesian Approximation"* (ICML) | Proved Monte Carlo sampling over stochastic weights approximates posterior predictive distributions | **Inference engine** — $T=50$ forward passes to compute mean risk $\hat{y}$ and variance $\sigma^2$ |

---

## Slide 5 — Takeaways / Research Gaps

**Content:**

**Motivation:**
- Standard ML (Logistic Regression, XGBoost, standard MLPs) treats financial customers as isolated i.i.d. samples, ignoring **relational customer similarity**.
- Black-box GNNs learn strong representations but lack **transparent probabilistic reasoning** and **uncertainty calibration**.
- Direct GNN-to-BN discretization pipelines suffer from severe **information loss** by forcing continuous embeddings into discrete buckets before classification.

**Research Gaps Addressed:**
1. **Relational Topology in Tabular Data:** We construct an adaptive $k$-NN feature-similarity graph to let GNNs learn neighborhood financial risk interactions ($H$).
2. **Intermediate Probabilistic Structuring (No Discretization Loss):** Instead of forcing the BN to be a low-capacity final classifier, we use the BN as an intermediate reasoning module to derive structured latent factors ($Z$).
3. **Dual Representation Fusion ($[H \parallel Z]$):** We fuse continuous relational GNN embeddings ($H$) with structured BN latent factors ($Z$), passing maximum information to the BNN discriminator.
4. **Epistemic Uncertainty Quantification:** Monte Carlo weight sampling ($T=50$) in the BNN yields calibrated prediction variance ($\sigma^2$) for financial auditing.

---

## Slide 6 — Objectives

**Content:**

**Objective 1: Construct Financial Similarity Graph & GNN Relational Encoding ($H$)**
- Build adaptive $k$-NN customer similarity graph from continuous financial profiles.
- Train Graph Attention Network (GAT/GCN) to extract neighborhood-aware latent embeddings $H$.

**Objective 2: Learn Bayesian Network Structure & Extract Latent Factors ($Z$)**
- Enforce temporal payment constraints (`PAY_6` $\to \dots \to$ `PAY_0`) via Hill-Climb + BIC search.
- Estimate CPTs with BDeu priors; extract factor posteriors $Z$ via exact Variable Elimination.

**Objective 3: Dual-Representation Fusion & Variational BNN Training**
- Concatenate relational embeddings with BN posteriors into fused vector $[H \parallel Z]$.
- Train 3-layer Variational BNN using Bayes by Backprop (combined BCE + $\lambda \cdot \text{KL}$ loss).

**Objective 4: Monte Carlo Uncertainty Estimation & Explainability Analysis**
- Execute $T=50$ Monte Carlo passes to output risk score $\hat{y}$ and epistemic variance $\sigma^2$.
- Perform t-SNE visualization of $[H \parallel Z]$ space, $\lambda$-sensitivity sweep, and high-uncertainty auditing.

---

## Slide 7 — Methodology — Overall

**Content:**

**Overall Unified Architecture Flowchart:**

```
               Raw Tabular Data (30,000 × 23)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Continuous Scaled Features         Quantile-Binned Features
            │                                 │
            ▼                                 │
  Adaptive k-NN Similarity Graph              │
            │                                 │
            ▼                                 │
   Graph Neural Network (GNN)                 │
   Neighborhood Embeddings (H)                │
            │                                 │
            │          ┌──────────────────────┘
            ▼          ▼
     ┌────────────────────────────┐
     │   Bayesian Network (BN)    │  ← Structure Learning (Hill-Climb + BIC)
     │   Factor Extraction        │     Parameter Learning (BDeu Prior)
     └─────────────┬──────────────┘
                   │
                   ▼
         Latent Factor Posteriors (Z)
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
Embedding H                Latent Posteriors Z
    │                             │
    └──────────────┬──────────────┘
                   ▼
        Dual-Representation Fusion [H ║ Z]
                   │
                   ▼
        Bayesian Neural Network (BNN)  ← Bayes-by-Backprop (BCE + λ·KL)
                   │
                   ▼
        Monte Carlo Sampling (T=50)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
Risk Probability (ŷ)    Epistemic Uncertainty (σ²)
```

---

## Slide 8 — Methodology — PR Inference

**Content:**

### Probabilistic Reasoning Inference Methods

**1. Exact Inference — Variable Elimination (Stage 2: Bayesian Network)**
- For each sample, queries conditional state probabilities: $P(Z_{\text{factor}} \mid \text{Markov Blanket evidence})$.
- Uses exact Variable Elimination from `pgmpy` to sum out non-query variables across joint factor tables.
- Computes 27 latent posterior probabilities across 5 factor groups (`z_demo`, `z_credit`, `z_pay_status`, `z_bill`, `z_pay_amt`).
- **Classification:** Exact PGM Inference.

**2. Approximate Variational Inference — Monte Carlo Sampling (Stage 4: BNN)**
- Performs $T = 50$ stochastic forward passes per test sample through variational weight distributions $w \sim q(w) = \mathcal{N}(\mu, \sigma^2)$.
- **Estimates Output Expectations:**
  $$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t \quad \text{(Expected Risk Score / Prediction Mean)}$$
  $$\sigma^2 = \frac{1}{T} \sum_{t=1}^{T} (\hat{y}_t - \hat{y})^2 \quad \text{(Epistemic Model Uncertainty / Prediction Variance)}$$

**Inference Comparison:**

| Aspect | Variable Elimination (BN) | Monte Carlo Weight Sampling (BNN) |
|:-------|:--------------------------|:-----------------------------------|
| **Inference Type** | Exact PGM Inference | Approximate Variational Inference |
| **Output** | Latent Factor Posteriors $Z \in \mathbb{R}^{27}$ | Expected Risk $\hat{y} \in [0,1]$ + Uncertainty $\sigma^2$ |
| **Role** | Structured Domain Knowledge | Calibrated Discriminative Classification |
| **Complexity** | Exponential in Treewidth | Linear in Pass Count $T \times N$ |

---

## Slide 9 — Methodology — PR Learning

**Content:**

### 1. Relational Graph Learning (GNN Layer)
- **Graph Construction:** $k$-nearest neighbors ($k=5$) feature-similarity graph on standardized financial profiles.
- **Message Passing:** Graph Convolutional Network (GCN) / Graph Attention Network (GAT) learns neighborhood interaction embeddings $H \in \mathbb{R}^{32}$.

### 2. Bayesian Network Learning (BN Layer)
- **Structure Learning:** Hill-Climb Search with BIC scoring, subject to domain constraints:
  - *Required edges:* Temporal chain $\text{PAY\_6} \to \text{PAY\_5} \to \text{PAY\_4} \to \text{PAY\_3} \to \text{PAY\_2} \to \text{PAY\_0}$.
  - *Forbidden edges:* No outgoing edges from target $Y$ to feature nodes.
- **Parameter Learning:** Bayesian Estimator with BDeu prior (equivalent sample size $= 10$) to estimate CPTs.

### 3. Feature Fusion & Variational BNN Learning (BNN Layer)
- **Fusion:** Concatenates GNN continuous embeddings $H$ and BN latent factor posteriors $Z$ into $[H \parallel Z] \in \mathbb{R}^{59}$.
- **Weight Parameterization:** Reparameterization trick: $w = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$ and $\sigma = \text{softplus}(\rho)$.
- **Loss Function (Bayes by Backprop):**
  $$\mathcal{L} = \text{BCE}(\hat{y}, y) + \lambda \cdot \text{KL}(q(w) \parallel p(w))$$
  - $\text{BCE}$: Binary Cross-Entropy task loss.
  - $\text{KL}$: Closed-form Gaussian KL divergence against zero-mean prior $p(w) = \mathcal{N}(0, 0.1^2 I)$.
  - $\lambda = 10^{-4}$ (Prior regularization coefficient).

---

## Slide 10 — Methodology — Decision, Intervention & What-If Scenarios

**Content:**

### 1. Uncertainty-Aware 2D Financial Decision Framework

```
                 Low Risk Score (ŷ < 0.5)      High Risk Score (ŷ ≥ 0.5)
              ┌─────────────────────────────┬─────────────────────────────┐
Low           │  AUTOMATED CREDIT APPROVAL  │  AUTOMATED REJECTION        │
Uncertainty   │  (High confidence, low risk)│  (High confidence, high risk)│
(σ² ≤ 75th %) │                             │                             │
              ├─────────────────────────────┼─────────────────────────────┤
High          │  FLAG FOR HUMAN AUDIT       │  FLAG FOR HUMAN AUDIT       │
Uncertainty   │  (Ambiguous client profile) │  (Complex / OOD case)       │
(σ² > 75th %) │                             │                             │
              └─────────────────────────────┴─────────────────────────────┘
```

### 2. What-If / Prior Regularization ($\lambda$) Sensitivity Analysis
- We sweep $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 3.0, 10.0\}$ over validation AUROC.
- **Finding:** Moderate $\lambda$ balances data-likelihood with weight prior. Extremely high $\lambda$ causes over-regularization (posterior collapses to prior), while very low $\lambda$ leads to overconfident misclassifications.

### 3. Structural Interventional Insights
- The temporal DAG constraints ($\text{PAY\_6} \to \dots \to \text{PAY\_0} \to Y$) confirm that recent repayment status ($\text{PAY\_0}$) acts as the primary causal mediator for credit default.

---

## Slide 11 — Methodology — Combining with Modern AI

**Content:**

### Integration of Graph Machine Learning, PGMs & Variational Deep Learning

| Pipeline Stage | Technology Domain | Paradigm & Role in Architecture |
|:---|:---|:---|
| **Graph Construction & GNN** | Graph Machine Learning | Non-Euclidean relational learning; captures structural client interactions ($H$) |
| **Bayesian Network** | Probabilistic Graphical Models (PGM) | Structural CPT modeling, domain-constrained DAG search, exact posterior inference ($Z$) |
| **Feature Fusion Layer** | Modern Feature Engineering | Fuses non-linear continuous representations ($H$) with interpretable probabilistic priors ($Z$) |
| **Variational BNN** | Bayesian Deep Learning | Variational inference via Bayes-by-Backprop; learns weight distributions $q(w)$ |
| **Monte Carlo Inference** | Uncertainty Quantification | Epistemic variance sampling ($T=50$) to quantify model confidence ($\sigma^2$) |

**Why this combination excels:**
- **GNN** captures relational topology across tabular clients.
- **BN** provides transparent, causal-constrained factor dependencies.
- **BNN** delivers high non-linear discriminative power with calibrated uncertainty.
- **Fusion $[H \parallel Z]$** prevents the discretization information loss seen in conventional GNN-BN models.

---

## Slide 12 — Progress in terms of Project Objectives

**Content:**

| Objective | Status | Implementation Details |
|:----------|:------:|:-----------------------|
| **Obj 1:** Data prep, $k$-NN Graph & GNN Embeddings ($H$) | ✅ Completed | Preprocessed 30,000 samples; built 5-NN similarity graph; extracted neighborhood embeddings $H$. |
| **Obj 2:** Bayesian Network structure & factor posteriors ($Z$) | ✅ Completed | Learned DAG via Hill-Climb+BIC with temporal constraints; extracted 27-dim latent vectors $Z$ via Variable Elimination. |
| **Obj 3:** Dual-Representation Fusion & Variational BNN | ✅ Completed | Fused $[H \parallel Z]$; trained 3-layer BNN (57/59-dim $\to$ 64 $\to$ 32 $\to$ 1) for 100 epochs with Bayes-by-Backprop. |
| **Obj 4:** MC Uncertainty ($T=50$), Evaluation & Visuals | ✅ Completed | Achieved 81.27% Acc, 0.769 AUROC; generated loss curves, t-SNE uncertainty overlay, and $\lambda$-sensitivity sweep. |

**Performance Metrics Summary:**

| Metric | Value | Interpretation |
|:-------|:-----:|:---------------|
| **Accuracy** | **81.27%** | Strong overall discrimination on imbalanced credit data |
| **Precision** | **66.10%** | Reliable default risk identification |
| **F1-Score** | **42.56%** | High minority-class default detection without heavy resampling |
| **AUROC** | **0.7694** | Well-calibrated probabilistic ranking |

---

## Slide 13 — Weekly Timeline for Project Completion

**Content:**

| Week | Phase | Key Tasks & Milestones | Deliverables |
|:-----|:------|:-----------------------|:-------------|
| **Week 1** | Literature Review | Study credit scoring models, PGM theory, GNNs, and BNN weight uncertainty | Problem formulation & replication plan |
| **Week 2** | Data Engineering | Preprocess UCI Credit dataset; produce continuous scaled & binned discrete splits | `data_prep.py` & 6 dataset split files |
| **Week 3** | Graph Encoding | Construct $k$-NN customer similarity graph; implement GNN/GAT feature extractor | `graph_builder.py` & GNN embedding matrix ($H$) |
| **Week 4** | PGM Modeling | Perform BN structure search (Hill-Climb+BIC) & BDeu parameter estimation | `bayes_network.py` & DAG structure |
| **Week 5** | Latent Inference | Execute exact Variable Elimination across factor groups; generate posteriors ($Z$) | Latent feature datasets (`latent_z_*.csv`) |
| **Week 6** | BNN & Fusion | Implement `BayesLinear` layers; fuse $[H \parallel Z]$; train BNN with Bayes-by-Backprop | `bnn_model.py` & trained state dict `bnn_model.pth` |
| **Week 7** | Uncertainty Evaluation | Run $T=50$ Monte Carlo passes; generate t-SNE plot, loss curves, and $\lambda$ sweep | `evaluate.py` & all 3 figure artifacts |
| **Week 8** | Documentation | Prepare final repository, `Abstract.md`, `README.md`, and presentation slides | Complete project repository & slides |

---

## Slide 14 — References

**Content:**

1. Yeh, I. C., & Lien, C. H. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." *Expert Systems with Applications*, 36(2), 2473–2480.
2. Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR 2017*.
3. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). "Graph Attention Networks." *ICLR 2018*.
4. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). "Weight Uncertainty in Neural Networks." *Proceedings of ICML 2015*.
5. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *Proceedings of ICML 2016*.
6. Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques.* MIT Press.
7. UCI Machine Learning Repository — Default of Credit Card Clients Dataset. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
8. pgmpy & PyTorch — Open-Source Probabilistic Graphical Models & Deep Learning Frameworks.
