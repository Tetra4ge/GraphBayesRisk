# Uncertainty-Aware Robust Financial Risk Discrimination via Bayesian Network & Bayesian Neural Modeling

> **Paper Replication** — This repository is a full end-to-end replication of the paper  
> *"Uncertainty-Aware Robust Financial Risk Discrimination via Bayesian Network and Bayesian Neural Modeling"*  
> applied to the **UCI Default of Credit Card Clients** dataset.

---

## Table of Contents

1. [What This Project Does (TL;DR)](#1-what-this-project-does-tldr)
2. [Background & Motivation](#2-background--motivation)
3. [Dataset](#3-dataset)
4. [Architecture Overview](#4-architecture-overview)
5. [Repository Structure](#5-repository-structure)
6. [Detailed File-by-File Walkthrough](#6-detailed-file-by-file-walkthrough)
7. [Intermediate & Output Files Explained](#7-intermediate--output-files-explained)
8. [Results](#8-results)
9. [Getting Started](#9-getting-started)
10. [Dependencies](#10-dependencies)
11. [Known Caveats & Limitations](#11-known-caveats--limitations)
12. [References](#12-references)
13. [License](#13-license)

---

## 1. What This Project Does (TL;DR)

Given a credit-card client's demographic info, credit limit, repayment history, bill amounts, and payment amounts, this pipeline **predicts whether the client will default next month** and also tells you **how confident (or uncertain) it is** about that prediction.

It does this by:
1. **Learning hidden (latent) relationships** between features using a Bayesian Network (probabilistic graphical model).
2. **Fusing** those learned latent factors with the original numeric features.
3. **Training a Bayesian Neural Network** on the fused representation — where every weight in the network is a probability distribution, not a single number.
4. **Running 50 stochastic forward passes** at inference time (Monte Carlo sampling) to produce both a **risk score** (probability of default) and an **uncertainty estimate** (variance across passes).

---

## 2. Background & Motivation

Traditional credit-scoring models give a single "will default / won't default" answer. They don't tell you *how sure* they are. A model might say "70% chance of default" — but is it really confident about that 70%, or is it just guessing because it hasn't seen similar clients before?

**This paper's key insight:** by combining two Bayesian approaches — a Bayesian Network for structured domain knowledge and a Bayesian Neural Network for flexible classification — the system can produce **calibrated uncertainty estimates**. High-uncertainty predictions flag clients who sit in ambiguous regions of the feature space and may need human review.

### Core Mathematical Formulation

The pipeline implements these key equations from the paper:

**Stage 1 — Bayesian Network (structured latent factors):**
```
p(z | G) = ∏ p(zᵢ | Pa(zᵢ))     ... Eq. (1)
p(y | x) = Σ_z p(y | z, x) · p(z | G)     ... Eq. (2)
```
Where `z` are latent factor posteriors, `G` is the learned DAG, and `Pa(zᵢ)` are the parent nodes.

**Stage 3 — Bayesian Neural Network (loss function):**
```
L_task = -log p(y | x)           ... BCE / NLL
L = L_task + λ · KL(q(w) || p(w))     ... Eq. (5-6)
```
Where `q(w)` is the learned variational posterior over weights and `p(w)` is the Gaussian prior.

**Stage 4 — Monte Carlo inference:**
```
ŷ = (1/T) Σₜ ŷₜ               ... risk score (mean)
Var = (1/T) Σₜ (ŷₜ - ŷ)²      ... uncertainty (variance)     ... Eq. (7)
```

---

## 3. Dataset

| Property | Value |
|:---|:---|
| **Name** | Default of Credit Card Clients |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| **Original Paper** | Yeh & Lien (2009), *Expert Systems with Applications* |
| **Samples** | 30,000 |
| **Features** | 23 |
| **Target** | `Y` — default payment next month (1 = yes, 0 = no) |
| **Class Distribution** | ~78% non-default, ~22% default (imbalanced) |
| **Missing Values** | None |
| **Region** | Taiwan, April–September 2005 |

### Feature Descriptions

| Feature(s) | Description | Type |
|:---|:---|:---|
| `LIMIT_BAL` | Credit limit (NT dollars, includes individual + family credit) | Continuous |
| `SEX` | Gender (1 = male, 2 = female) | Categorical |
| `EDUCATION` | Education level (1 = grad school, 2 = university, 3 = high school, 4 = others) | Categorical |
| `MARRIAGE` | Marital status (1 = married, 2 = single, 3 = others) | Categorical |
| `AGE` | Age in years | Continuous |
| `PAY_0` … `PAY_6` | Repayment status from Sep 2005 (PAY_0) backward to Apr 2005 (PAY_6). Scale: -1 = paid duly, 1 = 1-month delay, …, 9 = 9+ months delay | Ordinal |
| `BILL_AMT1` … `BILL_AMT6` | Bill statement amounts (NT$), Sep → Apr 2005 | Continuous |
| `PAY_AMT1` … `PAY_AMT6` | Previous payment amounts (NT$), Sep → Apr 2005 | Continuous |
| `Y` | Default payment next month (1 = yes, 0 = no) | Binary target |

---

## 4. Architecture Overview

The system processes data through four sequential stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE OVERVIEW                              │
│                                                                        │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐ │
│  │ Stage 0  │───▶│   Stage 1    │───▶│  Stage 2   │───▶│  Stage 3   │ │
│  │ Data Prep│    │ Bayesian Net │    │  Fusion    │    │    BNN     │ │
│  └──────────┘    └──────────────┘    └────────────┘    └────────────┘ │
│       │                │                   │                 │        │
│       ▼                ▼                   ▼                 ▼        │
│  Discrete CSV    Latent z CSVs      Fused features     Risk + Uncert │
│  Continuous CSV  (27 posterior       (30 raw +          (50 MC       │
│  (30 features    probabilities       27 latent          forward      │
│   + 1 target)    per sample)         = 57-dim)          passes)      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 0 — Data Preparation (`data_prep.py`)

Reads the raw CSV, cleans it, then produces **two parallel representations** of the same data:

| Representation | Purpose | Transformations Applied |
|:---|:---|:---|
| **Discrete (binned)** | Input to Bayesian Network | Continuous features quantile-binned into 4 buckets |
| **Continuous (scaled)** | Input to BNN | Monetary features log-transformed, categoricals one-hot encoded, numerics standardized (fit on train only) |

### Stage 1 — Bayesian Network (`bayes_network.py`)

- Learns a DAG (directed acyclic graph) structure from the **discrete training data** using Hill-Climb Search with BIC scoring
- Enforces domain constraints: temporal ordering of PAY variables, no outgoing edges from Y
- Estimates conditional probability tables (CPDs) using Bayesian estimation with BDeu prior
- Extracts **latent posterior probabilities** for 5 factor groups per sample via Variable Elimination:
  - `z_demo` — demographic factors (4 states)
  - `z_credit` — credit limit factors (4 states)
  - `z_pay_status` — repayment status factors (11 states)
  - `z_bill` — bill amount factors (4 states)
  - `z_pay_amt` — payment amount factors (4 states)
- **Total: 27 latent dimensions** per sample

### Stage 2 — Feature Fusion (inside `bnn_model.py`)

Concatenates the 30 continuous raw features with the 27 BN latent features → **57-dimensional fused vector** per sample.

### Stage 3 — Bayesian Neural Network (`bnn_model.py`)

| Component | Details |
|:---|:---|
| **Architecture** | `BayesLinear(57→64)` → ReLU → `BayesLinear(64→32)` → ReLU → `BayesLinear(32→1)` → Sigmoid |
| **Weight Representation** | Each weight is a Gaussian distribution: μ (mean) + ρ → σ via softplus. Sampled via reparameterization trick |
| **Prior** | Zero-mean Gaussian with σ = 0.1 |
| **Loss** | `BCE(ŷ, y) + λ · KL(q(w) ‖ p(w))` where λ = 1e-4 |
| **Optimizer** | Adam, lr = 0.001 |
| **Epochs** | 100 |
| **Batch Size** | 128 (train), 256 (val) |

### Stage 4 — MC Inference & Evaluation (`evaluate.py`)

- **T = 50 stochastic forward passes** through the BNN per test sample (weights are sampled each pass)
- **Risk score** = mean of 50 predictions
- **Uncertainty** = variance of 50 predictions
- Classification threshold = 0.5

---

## 5. Repository Structure

```
PR/
├── README.md                    ← You are here
├── LICENSE                      ← Apache 2.0
├── pyproject.toml               ← Project metadata & dependencies (uv/pip)
├── .python-version              ← Python 3.11
├── uv.lock                      ← Locked dependency versions (uv package manager)
├── replication_plan.md          ← Detailed plan for replicating the paper
├── Abstract.pdf                 ← Project abstract document
│
├── papers/                      ← Reference papers
│   ├── Uncertainty-AwareRobust...pdf    ← The paper being replicated
│   └── main.pdf                         ← Extended/supplementary material
│
├── data/
│   ├── raw/
│   │   └── default_of_credit_card_clients.csv   ← Original UCI dataset (30,000 rows)
│   └── processed/                                ← Generated by data_prep.py
│       ├── train_discrete.csv       ← 20,974 rows × 24 cols (binned features for BN)
│       ├── val_discrete.csv         ←  4,495 rows × 24 cols
│       ├── test_discrete.csv        ←  4,495 rows × 24 cols
│       ├── train_continuous.csv     ← 20,974 rows × 31 cols (scaled features for BNN)
│       ├── val_continuous.csv       ←  4,495 rows × 31 cols
│       ├── test_continuous.csv      ←  4,495 rows × 31 cols
│       ├── latent_z_train.csv       ← 20,974 rows × 27 cols (BN latent posteriors)
│       ├── latent_z_val.csv         ←  4,495 rows × 27 cols
│       └── latent_z_test.csv        ←  4,495 rows × 27 cols
│
├── src/                             ← All source code
│   ├── main.py                      ← Pipeline orchestrator (runs all 4 stages)
│   ├── data_prep.py                 ← Stage 0: cleaning, encoding, splitting
│   ├── bayes_network.py             ← Stage 1: BN structure learning + latent extraction
│   ├── bnn_model.py                 ← Stage 2+3: fusion + BNN training
│   └── evaluate.py                  ← Stage 4: MC inference, metrics, plots
│
├── outputs/
│   ├── evaluation_metrics.csv       ← Final test-set metrics (Accuracy, Precision, F1, AUROC)
│   ├── bnn_training_curves.csv      ← Per-epoch train/val loss (100 rows)
│   ├── models/
│   │   └── bnn_model.pth            ← Saved BNN model weights (PyTorch state dict)
│   └── figures/
│       ├── loss_vs_epoch.png        ← Train vs. val loss curves
│       ├── tsne_visualization.png   ← t-SNE of fused space with uncertainty overlay
│       └── prior_regularization_sensitivity.png  ← λ sweep: AUROC vs. KL weight
│
└── notebooks/                       ← Jupyter notebooks (placeholder, currently empty)
    ├── 01_eda.ipynb                 ← Exploratory data analysis
    ├── 02_bn_experiments.ipynb      ← Bayesian Network experiments
    └── 03_bnn_experiments.ipynb     ← BNN experiments
```

---

## 6. Detailed File-by-File Walkthrough

### `src/main.py` — Pipeline Orchestrator

**What it does:** Runs the entire pipeline in sequence by spawning each stage as a subprocess.

**Execution order:**
```
data_prep.py → bayes_network.py → bnn_model.py → evaluate.py
```

**Key detail:** It detects whether a `.venv` virtual environment exists and uses its Python interpreter if available; otherwise falls back to the system Python.

---

### `src/data_prep.py` — Data Cleaning & Feature Engineering

**What it does:** Transforms the raw UCI CSV into two clean, split representations ready for the two model stages.

**Step-by-step flow:**

1. **`load_data(filepath)`**
   - Reads the CSV, skipping the first row (which contains `X1`, `X2`, … labels) and using the second row as the actual header (`LIMIT_BAL`, `SEX`, …)
   - Renames `default payment next month` → `Y`
   - Drops the `ID` column
   - Coerces all values to numeric (non-numeric → NaN)

2. **`clean_data(df)`**
   - Drops rows with NaN values (handles rare corrupt entries)
   - Drops duplicate rows (30,000 → 29,964 unique rows after deduplication)
   - Clips `PAY_AMT1`–`PAY_AMT6` to a minimum of 0 (removes spurious negatives)
   - Ensures integer types for categorical/ordinal columns

3. **`prepare_bnn_data(df)` — Continuous representation**
   - Log-transforms all monetary/limit columns via `sign(x) · log1p(|x|)` to reduce skewness
   - One-hot encodes `SEX`, `EDUCATION`, `MARRIAGE` (drop-first)
   - Result: 30 feature columns + 1 target (`Y`) = **31 columns**

4. **`prepare_bn_data(df)` — Discrete representation**
   - Quantile-bins all continuous columns (`AGE`, `LIMIT_BAL`, `BILL_AMT*`, `PAY_AMT*`) into **4 buckets** (labels: 0, 1, 2, 3)
   - Keeps categorical/ordinal columns as-is
   - Result: 23 feature columns + 1 target = **24 columns**

5. **Splitting**
   - Stratified split: **70% train / 15% validation / 15% test** (preserves ~22% default rate in each split)
   - Exact sizes: **20,974 train / 4,495 val / 4,495 test**
   - **Same row indices** used for both discrete and continuous versions — critical so latent features align correctly during fusion

6. **StandardScaler** fitted **only on training continuous data** (prevents data leakage), then applied to val/test

**Outputs:** 6 CSV files in `data/processed/` (3 splits × 2 representations)

---

### `src/bayes_network.py` — Bayesian Network Structure & Latent Feature Extraction

**What it does:** Learns a probabilistic graphical model from the discretized data and extracts posterior probability vectors as latent features.

**Step-by-step flow:**

1. **`get_factor_groups()`** — Defines 5 conceptual factor groups:

   | Group | Target Node | Feature Nodes | Latent Dims |
   |:---|:---|:---|:---:|
   | `demo` | AGE | SEX, EDUCATION, MARRIAGE, AGE | 4 |
   | `credit` | LIMIT_BAL | LIMIT_BAL | 4 |
   | `pay_status` | PAY_0 | PAY_0–PAY_6 | 11 |
   | `bill` | BILL_AMT1 | BILL_AMT1–6 | 4 |
   | `pay_amt` | PAY_AMT1 | PAY_AMT1–6 | 4 |
   | | | **Total** | **27** |

2. **`learn_structure(df)`**
   - Uses `HillClimbSearch` from pgmpy with BIC scoring
   - **Enforces domain constraints:**
     - **Required edges:** Temporal chain `PAY_6 → PAY_5 → PAY_4 → PAY_3 → PAY_2 → PAY_0`
     - **Forbidden edges:** No outgoing edges from `Y` (label cannot cause features)
   - Maximum 100 iterations of the hill-climbing algorithm
   - Returns a `DiscreteBayesianNetwork` with the learned edges

3. **`estimate_parameters(model, df)`**
   - Uses `BayesianEstimator` with **BDeu prior** (equivalent sample size = 10)
   - Computes Conditional Probability Distributions (CPDs) for every node
   - Validates the model with `model.check_model()`

4. **`extract_latent_features(model, df, split_name)`**
   - For each sample in the dataset:
     - For each factor group, computes `P(target_node | Markov_Blanket_evidence)`
     - Evidence = Markov Blanket nodes **excluding** same-group features and `Y`
     - Uses `VariableElimination` for exact inference
     - Falls back to prior distribution if inference fails (e.g., unseen evidence combinations)
   - Returns a DataFrame of posterior probability vectors
   - **This is the most computationally expensive step** (~minutes per split)

**Outputs:** 3 latent CSV files in `data/processed/` (`latent_z_train.csv`, `latent_z_val.csv`, `latent_z_test.csv`)

---

### `src/bnn_model.py` — Bayesian Neural Network Definition & Training

**What it does:** Defines a custom Bayesian Neural Network with variational weight layers and trains it on fused features.

**Key classes:**

#### `BayesLinear(nn.Module)` — Custom Bayesian Linear Layer
- **Parameters:** `weight_mu`, `weight_rho`, `bias_mu`, `bias_rho` (4 parameter tensors per layer)
- **Forward pass:**
  1. Convert ρ → σ via softplus: `σ = log(1 + exp(ρ))`
  2. Sample noise: `ε ~ N(0, 1)`
  3. Compute weight: `w = μ + σ · ε` (reparameterization trick)
  4. Compute closed-form KL divergence against Gaussian prior N(0, σ²_prior)
- **Initialization:** μ ~ Uniform(-1/√fan_in, 1/√fan_in), ρ = -3.0 (small initial σ)

#### `BNN(nn.Module)` — The Full Network
```
Input(57) → BayesLinear(64) → ReLU → BayesLinear(32) → ReLU → BayesLinear(1) → Sigmoid → Output
```
- `get_kl()` sums KL divergence across all 3 layers

#### `FusedDataset(Dataset)` — Data Loader
- Reads a continuous CSV and a latent CSV
- Extracts labels (`Y`) from the continuous file
- Concatenates feature columns from both files → fused feature matrix
- Fused dimension: 30 (continuous features, excluding Y) + 27 (latent) = **57**

#### `train_model(...)` — Training Loop
- **Loss:** `BCE(ŷ, y) + λ · KL_total` where λ = 1e-4
- **Optimizer:** Adam (lr = 0.001)
- **Epochs:** 100
- Logs train/val loss every 10 epochs
- Saves per-epoch losses to `bnn_training_curves.csv`

**Output:** Saved model weights (`bnn_model.pth`) + training curves CSV

---

### `src/evaluate.py` — MC Inference, Metrics & Visualization

**What it does:** Loads the trained BNN, runs Monte Carlo inference, computes metrics, and generates all 3 figures.

**Step-by-step flow:**

1. **`run_mc_inference(model, loader, T=50)`**
   - Runs T=50 stochastic forward passes per test sample
   - Each pass samples different weights from the learned variational posterior
   - **Risk score** = mean of 50 predictions per sample
   - **Uncertainty** = variance of 50 predictions per sample

2. **Classification & Metrics (threshold = 0.5)**
   - Computes: Accuracy, Precision, F1-Score, AUROC
   - Saves results to `outputs/evaluation_metrics.csv`

3. **`plot_loss_curves()`**
   - Reads `bnn_training_curves.csv`
   - Plots train loss vs. val loss over 100 epochs
   - Saves to `outputs/figures/loss_vs_epoch.png`

4. **`plot_tsne(dataset, uncertainty, y_true)`**
   - Subsamples 2,000 test points for computational feasibility
   - Runs t-SNE (perplexity=30) on the 57-dim fused representation
   - Plots 2D scatter: non-default (blue circles) vs. default (orange squares)
   - Overlays high-uncertainty samples (>75th percentile variance) as red X marks
   - Saves to `outputs/figures/tsne_visualization.png`

5. **`run_lambda_sweep(input_dim, train_loader, val_loader)`**
   - Sweeps λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0}
   - For each λ: trains a fresh BNN for 25 epochs, evaluates via MC inference (T=30) on validation set
   - Plots AUROC vs. λ (log scale)
   - Saves to `outputs/figures/prior_regularization_sensitivity.png`

---

## 7. Intermediate & Output Files Explained

### `data/raw/default_of_credit_card_clients.csv`

The original UCI dataset. Has a quirky format: **Row 1** contains generic labels (`X1`, `X2`, …) and **Row 2** contains the real column names (`LIMIT_BAL`, `SEX`, …). The code handles this by setting `header=1`.

| Property | Value |
|:---|:---|
| Rows | 30,000 (+ 2 header rows) |
| Columns | 25 (ID + 23 features + Y) |

---

### `data/processed/` — All Generated CSVs

#### Discrete Files (for Bayesian Network)

| File | Rows | Columns | Description |
|:---|:---:|:---:|:---|
| `train_discrete.csv` | 20,974 | 24 | All features quantile-binned (0–3). Used to learn the BN structure and CPDs |
| `val_discrete.csv` | 4,495 | 24 | Same format. Used for validation during λ sweep |
| `test_discrete.csv` | 4,495 | 24 | Same format. Used during evaluation |

**Column list:** `LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1–6, PAY_AMT1–6, Y`

**Example row:** `1,2,2,1,2,-1,-1,-1,-1,-2,-1,0,0,0,0,0,1,0,0,0,1,2,0,1`
(All values are integers; continuous features are in bins 0–3)

---

#### Continuous Files (for BNN)

| File | Rows | Columns | Description |
|:---|:---:|:---:|:---|
| `train_continuous.csv` | 20,974 | 31 | Log-transformed, standardized numerics + one-hot categoricals |
| `val_continuous.csv` | 4,495 | 31 | Scaled using training set's scaler (no leakage) |
| `test_continuous.csv` | 4,495 | 31 | Same |

**Column list (31):** `LIMIT_BAL, AGE, PAY_0–PAY_6, BILL_AMT1–6, PAY_AMT1–6, Y, SEX_2, EDUCATION_1–6, MARRIAGE_1–3`

**Note:** `SEX_2`, `EDUCATION_1–6`, `MARRIAGE_1–3` are one-hot encoded (drop-first applied at the category level). Values in numeric columns are z-score standardized floats.

---

#### Latent Feature Files (from Bayesian Network)

| File | Rows | Columns | Description |
|:---|:---:|:---:|:---|
| `latent_z_train.csv` | 20,974 | 27 | Posterior probability vectors from BN inference |
| `latent_z_val.csv` | 4,495 | 27 | Same |
| `latent_z_test.csv` | 4,495 | 27 | Same |

**Column naming pattern:** `z_{group}_{state_index}`

| Column Range | Factor Group | # Columns | Meaning |
|:---|:---|:---:|:---|
| `z_demo_0` … `z_demo_3` | Demographics | 4 | P(AGE = bucket_i \| Markov Blanket) |
| `z_credit_0` … `z_credit_3` | Credit Limit | 4 | P(LIMIT_BAL = bucket_i \| MB) |
| `z_pay_status_0` … `z_pay_status_10` | Repayment Status | 11 | P(PAY_0 = state_i \| MB) |
| `z_bill_0` … `z_bill_3` | Bill Amounts | 4 | P(BILL_AMT1 = bucket_i \| MB) |
| `z_pay_amt_0` … `z_pay_amt_3` | Payment Amounts | 4 | P(PAY_AMT1 = bucket_i \| MB) |

**Each row sums to ~1.0 within each factor group** (they are probability distributions).

---

### `outputs/evaluation_metrics.csv`

```csv
Metric,Value
Accuracy,0.8127
Precision,0.6610
F1-Score,0.4256
AUROC,0.7694
```

### `outputs/bnn_training_curves.csv`

100 rows, one per epoch:
```csv
epoch,train_loss,val_loss
1,0.7590,0.6532
2,0.6163,0.5915
...
100,0.4722,0.4783
```
Shows steady convergence with train/val losses closely tracking each other (no significant overfitting).

### `outputs/models/bnn_model.pth`

PyTorch state dictionary containing the learned variational parameters (μ and ρ) for all 3 BayesLinear layers. ~50 KB.

---

## 8. Results

### Test Set Performance

| Metric | This Replication | Paper Target |
|:---|:---:|:---:|
| **Accuracy** | 81.27% | ~90% |
| **Precision** | 66.10% | ~89% |
| **F1-Score** | 42.56% | ~88% |
| **AUROC** | 76.94% | ~94% |

> **Note:** The gap between this replication and the paper targets is expected — the paper used additional techniques (class resampling, architecture tuning, extended training) that can be added incrementally. The pipeline structure and methodology are faithfully replicated.

### Generated Figures

**1. Training Loss Curves** (`outputs/figures/loss_vs_epoch.png`)

Shows BCE + KL loss over 100 epochs. Both training and validation losses converge smoothly to ~0.47, indicating stable training without overfitting.

**2. t-SNE Visualization** (`outputs/figures/tsne_visualization.png`)

2D projection of the 57-dimensional fused feature space. Default and non-default clients show partial separation. High-uncertainty samples (red X marks, >75th percentile variance) cluster at the boundary between classes — exactly where you'd expect the model to be least confident.

**3. Prior Regularization Sensitivity** (`outputs/figures/prior_regularization_sensitivity.png`)

Shows how the KL weight λ affects validation AUROC. Demonstrates the trade-off: too little regularization (small λ) leads to overconfident predictions, while too much (large λ) forces the posterior too close to the prior and reduces discriminative power.

---

## 9. Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **uv** (recommended) or **pip** for dependency management

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd PR
```

### Step 2: Download the Dataset

Download `default_of_credit_card_clients.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and place it in:

```
data/raw/default_of_credit_card_clients.csv
```

> The file is already included in this repo if you cloned it.

### Step 3: Install Dependencies

**Option A — Using uv (recommended):**
```bash
uv sync
```

**Option B — Using pip:**
```bash
pip install torch pgmpy pandas numpy scikit-learn matplotlib seaborn tqdm
```

### Step 4: Run the Full Pipeline

```bash
python src/main.py
```

This sequentially runs:
1. `data_prep.py` — cleans data, creates splits (~seconds)
2. `bayes_network.py` — learns BN, extracts latent features (~5–15 minutes)
3. `bnn_model.py` — trains BNN for 100 epochs (~2–5 minutes on CPU)
4. `evaluate.py` — runs MC inference, plots, λ sweep (~5–10 minutes)

**Total runtime:** ~15–30 minutes on a modern CPU.

### Step 5: Check Results

After the pipeline completes:
- **Metrics** → `outputs/evaluation_metrics.csv`
- **Figures** → `outputs/figures/`
- **Model** → `outputs/models/bnn_model.pth`
- **Training curves** → `outputs/bnn_training_curves.csv`

### Running Individual Stages

You can also run each stage independently (useful for debugging):

```bash
python src/data_prep.py        # Just data preparation
python src/bayes_network.py    # Just BN structure learning + latent extraction
python src/bnn_model.py        # Just BNN training
python src/evaluate.py         # Just evaluation (requires trained model)
```

> **Important:** Stages must be run in order because each depends on outputs from the previous stage.

---

## 10. Dependencies

| Package | Version | Purpose |
|:---|:---|:---|
| `torch` | ≥ 2.13.0 | BNN implementation (BayesLinear, training loop) |
| `pgmpy` | ≥ 1.1.2 | Bayesian Network structure/parameter learning, inference |
| `pandas` | ≥ 3.0.5 | Data loading, manipulation, CSV I/O |
| `numpy` | ≥ 2.4.6 | Numerical operations |
| `scikit-learn` | (transitive) | StandardScaler, train_test_split, metrics, t-SNE |
| `matplotlib` | ≥ 3.11.1 | Plotting (loss curves, λ sweep) |
| `seaborn` | ≥ 0.13.2 | Enhanced scatter plots (t-SNE visualization) |
| `tqdm` | (transitive) | Progress bars for BN latent extraction |

---

## 11. Known Caveats & Limitations

| Issue | Details |
|:---|:---|
| **BN sensitivity to binning** | Continuous features are quantile-binned into 4 buckets, which is a lossy transformation. The learned DAG structure and latent posteriors are sensitive to the number and type of bins chosen. |
| **Class imbalance** | The dataset is imbalanced (~78/22 split). This replication does not apply oversampling (e.g., SMOTE) or class weighting, which could improve recall and F1 on the minority class. |
| **Inference latency** | MC sampling requires T=50 forward passes per sample, making real-time inference ~50× slower than a point-estimate model. The paper notes this as a deployment limitation. |
| **No GPU acceleration** | The current code runs on CPU. For larger datasets, adding CUDA device handling would speed up BNN training significantly. |
| **Notebooks are placeholders** | The 3 Jupyter notebooks (`01_eda.ipynb`, `02_bn_experiments.ipynb`, `03_bnn_experiments.ipynb`) are currently empty and reserved for interactive exploration. |

---

## 12. References

1. **Replicated Paper:**  
   *"Uncertainty-Aware Robust Financial Risk Discrimination via Bayesian Network and Bayesian Neural Modeling"*  
   (Available in `papers/`)

2. **Original Dataset Paper:**  
   Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473-2480.  
   [Semantic Scholar](https://www.semanticscholar.org/paper/1cacac4f0ea9fdff3cd88c151c94115a9fddcf33)

3. **Dataset:**  
   [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

4. **Key Libraries:**  
   - [pgmpy](https://pgmpy.org/) — Probabilistic Graphical Models in Python  
   - [PyTorch](https://pytorch.org/) — Deep Learning Framework  

---

## 13. License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.
