# Replication Plan: Uncertainty-Aware Robust Financial Risk Discrimination via Bayesian Network and Bayesian Neural Modeling

Dataset: UCI Default of Credit Card Clients (30,000 samples, 23 features, binary label `Y`)

---

## 1. Overview

Pipeline: **Raw features → Bayesian Network (structured latent factors) → Fusion → Bayesian Neural Network (uncertainty-aware discriminator) → risk probability + uncertainty**

---

## 2. Data Preparation

**Feature groups**
- Demographics: SEX, EDUCATION, MARRIAGE, AGE
- Credit: LIMIT_BAL
- Repayment status (temporal chain): PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
- Bill amounts: BILL_AMT1–6
- Payment amounts: PAY_AMT1–6

**Cleaning**
- Drop duplicates, check missing/outliers
- Non-negativity checks on monetary fields; cap extreme outliers
- Standardize numeric types

**Encoding**
- One-hot: SEX, EDUCATION, MARRIAGE
- Keep PAY_0…PAY_6 as ordered discrete values
- Standardize/robust-scale continuous features
- Log-transform skewed monetary fields (LIMIT_BAL, BILL_AMT*, PAY_AMT*)

**Two parallel representations needed:**
1. **Discrete/binned version** for the Bayesian Network (quantile-bin continuous vars)
2. **Continuous/normalized version** for the BNN input

**Splits**
- Stratified train/val/test (e.g., 70/15/15), preserve class ratio (~22% default) in val/test
- Class weighting or resampling on training set only

---

## 3. Stage 1 — Bayesian Network (structured latent factors)

**Library:** `pgmpy`

- **Structure learning:** Hill-Climb Search with BIC/BDeu score, restricted via whitelist/blacklist:
  - Enforce temporal ordering across PAY_0 → PAY_2 → ... → PAY_6
  - Disallow edges pointing backward into evidence nodes
  - Allow edges from factor groups into label node Y
- **Parameter learning:** `BayesianEstimator` with BDeu prior → CPDs
- **Inference:** `VariableElimination` (or `BeliefPropagation`) to compute posterior `p(z_i | evidence)` per factor group per sample
- **Output:** latent vector `z = [z_demo, z_credit, z_pay_status, z_bill, z_pay_amt]` (posterior probabilities per latent state)

Corresponds to paper Eq. (1)–(2):
```
p(z|G) = Π p(z_i | Pa(z_i))
p(y|x) = Σ_z p(y|z,x) p(z|G)
```

---

## 4. Stage 2 — Fusion

- Concatenate BN latent posteriors `z` with normalized/continuous raw features `x`
- Result: fused representation `h(x, z)` fed to the discriminator

---

## 5. Stage 3 — Bayesian Neural Network Discriminator

**Library:** `blitz-bayesian-pytorch` (or `torchbnn`)

**Architecture**
```
fused_input → BayesLinear(64) → ReLU
            → BayesLinear(32) → ReLU
            → BayesLinear(1)  → Sigmoid
```
- Gaussian weight priors N(0, σ²I); variational posterior q(w) learned via Bayes-by-Backprop

**Loss (Eq. 5–6)**
```
L_task = -log p(y|x)          # BCE / NLL
L = L_task + λ · KL(q(w) || p(w))
```
- λ = prior regularization coefficient, tunable

**Training**
- Adam optimizer, mini-batch training
- Track train/val loss curves → replicate Fig. 2

---

## 6. Stage 4 — Inference (Eq. 7)

- Monte Carlo sampling: T=50 stochastic forward passes per test sample
- Risk score = mean of T predictions
- Uncertainty = variance of T predictions
```
ŷ = (1/T) Σ ŷ_t
Var = (1/T) Σ (ŷ_t - ŷ)²
```

---

## 7. Evaluation

- **Metrics:** Accuracy, Precision, F1, AUROC — compare to paper's Table 1 baselines (target ~0.90/0.89/0.88/0.94)
- **Loss curves:** train vs val loss over epochs (Fig. 2 style)
- **t-SNE:** project fused representations to 2D, color by class, mark high-variance points as outliers (Fig. 3 style)
- **λ sensitivity sweep:** scan KL weight λ over {1e-4 … 10} log-scale, plot AUROC vs λ (Fig. 4 style) — expect rise-then-fall pattern

---

## 8. Suggested Module Layout

```
data_prep.py       # cleaning, binning, encoding, splits
bayes_network.py   # BN structure/parameter learning, latent inference
bnn_model.py        # BNN architecture, training loop, KL loss
evaluate.py         # metrics, t-SNE, loss curves, λ-sensitivity sweep
main.py             # orchestrates end-to-end pipeline
```

## 9. Key Dependencies

```
pgmpy
blitz-bayesian-pytorch  # or torchbnn
torch
scikit-learn
pandas, numpy
matplotlib, seaborn
```

## 10. Caveats / Risks

- Full discrete BN structure learning on continuous financial data is sensitive to binning choices — validate with domain-informed edge constraints
- Bayesian NN sampling adds inference latency (paper notes this as a deployment limitation)
- Class imbalance handling must be applied to training set only to keep val/test realistic
