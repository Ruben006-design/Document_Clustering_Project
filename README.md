# Optimal Document Clustering via Hybrid Feature Selection & Soft Computing Classifier

An independent, reproducible, and leakage-free Python implementation of the document clustering framework described in:
> **“An optimal document clustering method using hybrid optimal feature selection and efficient soft computing classifier”**  
> *P. Pitchandi and R. Kingsy Grace, Expert Systems With Applications, Vol. 294, Part A (2025), Article 128762.*  
> **DOI:** [10.1016/j.eswa.2025.128762](https://doi.org/10.1016/j.eswa.2025.128762)

---

## 1. Project Purpose & Pipeline Architecture

This project provides an end-to-end, scientifically validated pipeline for high-dimensional document clustering and classification. The framework integrates two-stage feature selection with an efficient kernelized soft-computing classifier:

```
[Raw Documents]
       │
       ▼
[Stage 1: Stratified Train / Test Partitioning]  <-- Strict leakage barrier (fit on train ONLY)
       │
       ▼
[Stage 2: Text Preprocessing & Cleaning]
       │
       ▼
[Stage 3: TF-IDF Feature Extraction]
       │
       ▼
[Stage 4: Fuzzy Density Fruit Fly Optimization (FD-FFO)]  <-- Density-based feature screening
       │
       ▼
[Stage 5: Teaching-Learning Harris Hawks Optimization (TL-HHO)]  <-- Metaheuristic wrapper selection
       │
       ▼
[Stage 6: Support Vector Regression - Probabilistic Neural Network (SVR-PNN)]  <-- Exact KKT / Scalable Nyström
       │
       ▼
[Stage 7: Comprehensive Evaluation & Export]  <-- Supervised & Clustering metrics on unseen test set
```

### Core Design Guarantees
- **Zero Data Leakage:** All stateful components (TF-IDF vectorizer, FD-FFO density thresholds, TL-HHO internal validation, and SVR-PNN kernel representations) are fitted strictly on training data. The held-out test partition is evaluated only after the entire pipeline is frozen.
- **Dual Solver Architecture:**
  - `solver_mode='exact'` (default): Solves the standard $(N+1) \times (N+1)$ LS-SVR dual Karush-Kuhn-Tucker (KKT) linear system directly via analytical matrix inversion.
  - `solver_mode='nystrom'` (opt-in): Low-rank Nyström approximation ($m=200$ landmarks) mapping samples into primal space, enabling full-dataset execution ($N=18,846$ docs) in sub-20 seconds on standard PCs without memory thrashing.
- **Sparse Matrix Preservation:** Pairwise distance calculations in FD-FFO operate directly over sparse matrices (`scipy.sparse.csr_matrix`), avoiding full-matrix densification.

---

## 2. Environment Setup & Requirements

### System Requirements
- Python 3.10+ (tested on Python 3.14.7 64-bit on Windows)
- Dependencies: `numpy`, `scipy`, `scikit-learn`

### Setup Instructions
```powershell
# Clone or navigate to the repository
cd D:\DocumentClusteringProject

# Create and activate a virtual environment (if not already existing)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install numpy scipy scikit-learn
```

### Dataset Instructions
1. **20 Newsgroups (18,846 documents, 20 categories):**
   Loaded automatically via the local scikit-learn offline cache (`C:\Users\<User>\scikit_learn_data\20news-bydate_py3.pkz`). No internet connection is required.
2. **Web-Snippets (12,340 snippets, 8 domains):**
   Parsed automatically from local zip archive (`ENwebsnippets.zip` containing `train.txt` and `test.txt`).

---

## 3. Algorithm Implementation Status & Equation Traceability

| Paper Section & Equations | Component | File & Function | Implementation Status | Implementation Details & Ambiguity Resolutions |
| :--- | :--- | :--- | :---: | :--- |
| **Section 4.1, Eqs. (1)–(3)** | Local density $\rho$ and expanse $\delta$ | `src/fd_ffo.py`<br>`_calculate_density`, `_calculate_delta` | **Faithful** (Typo Resolved) | Resolves paper typo $D_b < 0$ to positive cutoff $d_c$ such that $D_{ji} < d_c$ (Rodriguez & Laio, 2014). Paper typo $\rho_j > \rho_i$ in Eq. (3) corrected to $\rho_i > \rho_j$. |
| **Section 4.1, Eqs. (4)–(10)** | Fruit Fly Optimization (FFO) | `src/fd_ffo.py`<br>`fit` | **Faithful / Interpretation** | Implements Pan (2012) 2D fly coordinate updates. Eq. (7) $\text{Smell}_j = \text{fitness}(T_j)$ is undefined in paper; implemented as $\text{Smell}_j = T_j$. |
| **Section 4.1, Algorithm 1** | **Fuzzy Density Score** | `src/fd_ffo.py`<br>`fit` | **Documented Variant** (Undefined in Paper) | **Paper Ambiguity:** The paper text claims *"fuzzy density score below 0.4 eliminated... 70% removed"*, but gives **no formula**. Supported via `scoring_mode`: `passthrough` (default, faithful), `density_score` ($\bar{\rho} \cdot \bar{\delta} \ge 0.4$), and `fixed_ratio` (top 30%). |
| **Section 4.2, Eqs. (12)–(16)** | TLBO Teaching & Learner Phases | `src/tl_hho.py`<br>`teaching_phase`, `learner_phase` | **Faithful** (Bug Fixed) | Learner phase corrected so that learner $i$ interacts with randomly drawn peer $p \neq i$, properly updating learner $i$. |
| **Section 4.2, Eqs. (17)–(28)** | HHO Exploration & Exploitation | `src/tl_hho.py`<br>`hho_exploration`, `hho_exploitation` | **Faithful** | Implements prey escape energy $H$, soft besiege, Mantegna Lévy flight ($\beta=1.5$), and rapid dive selection $\min(G(X), G(K))$. |
| **Section 4.2, Eq. (31)** | Wrapper Fitness Objective | `src/tl_hho.py`<br>`calculate_fitness` | **Faithful** | Fitness evaluates internal validation k-NN error ($\alpha=0.9$) plus feature ratio penalty ($\beta=0.1$). Zero test leakage. |
| **Section 4.3, Eqs. (32)–(40)** | LS-SVR Dual KKT System | `src/svr_pnn.py`<br>`fit` | **Faithful** | Analytical solution of the dual KKT block system for regression. |
| **Section 4.3, Eqs. (32)–(40) (Ext)** | Nyström Low-Rank Approximation | `src/svr_pnn.py`<br>`fit` (`solver_mode='nystrom'`) | **Scalable Extension** | Opt-in low-rank landmark projection ($m=200$) solving primal ridge regression in an $(m+1) \times (m+1)$ system in milliseconds. |
| **Section 4.3, Eq. (41)** | Gaussian RBF Kernel | `src/svr_pnn.py`<br>`rbf_kernel` | **Faithful** | Squared Euclidean distance RBF kernel: $\exp(-\|x - x'\|^2 / \sigma^2)$. |
| **Section 4.3, Eqs. (42)–(43)** | PNN Activations & Multiclass Mapping | `src/svr_pnn.py`<br>`predict_proba` | **Defensible Extension** | Resolves paper ambiguity connecting 1D scalar SVR to multiclass PNN via continuous one-hot indicator embedding and batched evaluation. |
| **Section 5.1, Eqs. (44)–(49)** | Metrics (Acc, Prec, Rec, F1, RI, NMI) | `main.py`<br>`run_experiment` | **Faithful** | Scikit-learn multi-class implementations evaluated strictly on held-out test sets. |

---

## 4. Final Measured Experimental Results

All experiments were executed with random seed **42** using the current environment (`.\.venv\Scripts\python.exe`). Output metrics, confusion matrices, and execution logs were written directly to `results/final/`:

| Experiment | Dataset | Mode | Solver Mode | Train / Test Docs | Features (TF-IDF $\to$ FD-FFO $\to$ TL-HHO) | Accuracy | Weighted F1 | Rand Index (RI) | NMI | Runtime |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A (Smoke Test)** | 20 Newsgroups | `smoke_test` | `exact` | 160 / 40 | 100 $\to$ 100 $\to$ 31 | **15.00%** | **13.10%** | **86.03%** | **0.6251** | 1.31s |
| **Experiment B (Exact Benchmark)** | 20 Newsgroups | `benchmark` | `exact` | 800 / 200 | 300 $\to$ 300 $\to$ 221 | **45.50%** | **46.71%** | **92.21%** | **0.5261** | 2.63s |
| **Experiment C (Nyström Benchmark)** | 20 Newsgroups | `benchmark` | `nystrom` ($m=200$) | 800 / 200 | 300 $\to$ 300 $\to$ 221 | **40.50%** | **42.60%** | **90.18%** | **0.4919** | 2.54s |
| **Experiment D (Full Official Split)** | 20 Newsgroups | `full` | `nystrom` ($m=200$) | 11,314 / 7,532 | 100 $\to$ 100 $\to$ 97 | **24.23%** | **22.30%** | **81.11%** | **0.2103** | 18.89s |
| **Experiment E (Web-Snippets)** | Web-Snippets | `benchmark` | `exact` | 384 / 96 | 300 $\to$ 300 $\to$ 134 | **52.08%** | **52.36%** | **81.47%** | **0.3899** | 1.56s |

*Note: For stratified splits with equal class representations (10 samples/class in benchmark), weighted and macro averages are mathematically identical. Raw machine-readable logs and metrics are archived in `results/final/final_experiment_metrics.csv`.*

---

## 5. Execution & Reproduction Commands

### Run Full Test Suite
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```
*(Runs all 29 unit and mathematical validation tests.)*

### Run Final Experiments

**Experiment A — 20 Newsgroups Smoke Test:**
```powershell
.\.venv\Scripts\python.exe main.py --mode smoke_test --seed 42 --results_dir results/final/smoke_test
```

**Experiment B — 20 Newsgroups Exact Benchmark:**
```powershell
.\.venv\Scripts\python.exe main.py --mode benchmark --dataset 20newsgroups --fdffo_mode passthrough --mapping_mode one_hot --max_features 300 --pop_size 10 --max_iter 5 --seed 42 --results_dir results/final/20ng_exact
```

**Experiment C — 20 Newsgroups Nyström Benchmark ($m=200$):**
```powershell
.\.venv\Scripts\python.exe main.py --mode benchmark --dataset 20newsgroups --fdffo_mode passthrough --mapping_mode one_hot --solver_mode nystrom --n_components 200 --max_features 300 --pop_size 10 --max_iter 5 --seed 42 --results_dir results/final/20ng_nystrom
```

**Experiment D — Full 20 Newsgroups Evaluation (11,314 Train / 7,532 Test Docs):**
```powershell
.\.venv\Scripts\python.exe main.py --mode full --dataset 20newsgroups --solver_mode nystrom --n_components 200 --pop_size 5 --max_iter 3 --seed 42 --results_dir results/final/20ng_full_nystrom
```

**Experiment E — Web-Snippets Benchmark:**
```powershell
.\.venv\Scripts\python.exe main.py --mode benchmark --dataset web_snippets --fdffo_mode passthrough --mapping_mode one_hot --max_features 300 --pop_size 10 --max_iter 5 --seed 42 --results_dir results/final/web_snippets
```

---

## 6. Known Paper Discrepancies & Scientific Limitations

1. **Undefined "Fuzzy Density Score":**
   Section 4.1 claims features with *"fuzzy density score below 0.4 are eliminated"*, but **never provides a mathematical formula** defining this score in the text or algorithms. To maintain scientific integrity without fabricating equations, `src/fd_ffo.py` defaults to `scoring_mode='passthrough'` (keeps all features), while providing `density_score` ($\bar{\rho} \cdot \bar{\delta} \ge 0.4$) and `fixed_ratio` (drops lowest 70%) as explicit, documented heuristic variants.
2. **Scalar SVR to Multiclass PNN Interface:**
   The paper formulates standard 1D scalar LS-SVR (Eqs. 32–40) alongside standard vector PNN formulas (Eqs. 42–43), omitting the mathematical interface between them for multiclass datasets (e.g. 20 classes). The implementation resolves this via `mapping_mode='one_hot'`, fitting LS-SVR with a continuous one-hot indicator matrix to avoid imposing artificial ordinal relationships on nominal categories.
3. **Table 7 Scalability Contradiction:**
   The paper claims an execution time of 19.8 seconds on 20,000 documents on an Intel Core i3 PC. As proven mathematically, an exact dual KKT system on 20,000 documents requires allocating a $3.05 \text{ GB}$ dense float64 matrix and $\approx 2.67 \times 10^{12}$ FLOPs, which requires several minutes on a CPU. The authors could not have used an exact dual solver for the full dataset; our opt-in Nyström low-rank approximation ($m=200$) reproduces sub-20 second execution times safely without memory exhaustion.
4. **Finite-Sample Bias in NMI:**
   The smoke test exhibits higher NMI ($0.6251$) than the benchmark ($0.5261$) due to the well-known positive finite-sample bias of Shannon mutual information when evaluating very few samples per class ($N/K = 2.0$ vs. $10.0$).

---

## 7. Project Structure

```
D:\DocumentClusteringProject/
├── main.py                        # Pipeline entry point & CLI orchestrator
├── README.md                      # Project documentation & reproduction guide
├── src/
│   ├── data_loader.py             # Offline 20 Newsgroups & Web-Snippets loaders
│   ├── preprocessing.py          # Lowercasing, punctuation & whitespace cleaner
│   ├── feature_extraction.py      # TF-IDF vectorizer wrapper (fit on train ONLY)
│   ├── fd_ffo.py                  # Fuzzy Density Fruit Fly Optimization
│   ├── tl_hho.py                  # Teaching-Learning Harris Hawks Optimization
│   └── svr_pnn.py                 # SVR-PNN (Dual exact KKT & Primal Nyström)
├── tests/
│   ├── test_data_loader.py        # Dataset loading & stratification tests
│   ├── test_fd_ffo.py             # FD-FFO distance, density & mode tests
│   ├── test_tl_hho.py             # TL-HHO position mapping & fitness tests
│   ├── test_svr_pnn.py            # SVR-PNN KKT, kernel & numerical stability tests
│   ├── test_pipeline.py           # End-to-end data leakage prevention test
│   └── test_mathematical_validation.py # Hand-calculated proofs & invariant tests
├── results/
│   ├── final/                     # Frozen final experimental deliverables
│   │   ├── final_experiment_metrics.csv # Master metrics summary table
│   │   ├── smoke_test/            # Experiment A outputs
│   │   ├── 20ng_exact/            # Experiment B outputs
│   │   ├── 20ng_nystrom/          # Experiment C outputs
│   │   ├── 20ng_full_nystrom/     # Experiment D outputs
│   │   └── web_snippets/          # Experiment E outputs
│   └── verification_20261020_000800/ # Audit artifacts & equation traceability
└── backups/                       # Timestamped pre-modification snapshots
```
