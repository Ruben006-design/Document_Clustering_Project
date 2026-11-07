# An Adaptive Density-Aware and Scalable Framework for Document Clustering with Hybrid Feature Selection and Soft Computing Classification

An independent, reproducible, and leakage-free Python implementation and research extension of the document clustering framework described in:
> **“An optimal document clustering method using hybrid optimal feature selection and efficient soft computing classifier”**  
> *P. Pitchandi and R. Kingsy Grace, Expert Systems With Applications, Vol. 294, Part A (2025), Article 128762.*  
> **DOI:** [10.1016/j.eswa.2025.128762](https://doi.org/10.1016/j.eswa.2025.128762)

Extended with **Adaptive Density-Aware Feature Selection (ADAFS)** to provide an explicit, mathematically rigorous, and data-driven alternative to the source paper's undefined fuzzy density screening step.

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
[Stage 4: Feature Selection (FD-FFO Modes OR Proposed ADAFS)]  <-- Density & relevance screening
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
- **Zero Data Leakage:** All stateful components (TF-IDF vectorizer, FD-FFO / ADAFS density thresholds, TL-HHO internal validation, and SVR-PNN kernel representations) are fitted strictly on training data. The held-out test partition is evaluated only after the entire pipeline is frozen.
- **Dual Solver Architecture:**
  - `solver_mode='exact'` (default): Solves the standard $(N+1) \times (N+1)$ LS-SVR dual Karush-Kuhn-Tucker (KKT) linear system directly via analytical matrix inversion.
  - `solver_mode='nystrom'` (opt-in): Low-rank Nyström approximation ($m=200$ landmarks) mapping samples into primal space, enabling full-dataset execution ($N=18,846$ docs) in sub-20 seconds on standard PCs without memory thrashing.
- **Sparse Matrix Preservation:** Pairwise distance calculations and relevance computations in FD-FFO/ADAFS operate directly over sparse matrices (`scipy.sparse.csr_matrix`), avoiding full-matrix densification.

---

## 2. Environment Setup & Requirements

### System Requirements
- Python 3.10+ (tested on Python 3.14.7 64-bit on Windows)
- Dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`

### Setup Instructions
```powershell
# Clone or navigate to the repository
cd C:\Users\ruben\Document_clustering\DocumentClusteringProject

# Create and activate a virtual environment (if not already existing)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install numpy scipy scikit-learn matplotlib
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
| **Research Extension** | **Adaptive Density-Aware Feature Selection (ADAFS)** | `src/fd_ffo.py`<br>`fit` (`scoring_mode='adaptive_density'`) | **Novel Extension** (Explicit & Reproducible) | Explicit ANOVA correlation ratio relevance minus density-based redundancy ($S_j = Rel_j - \lambda Red_j \in [0, 1]$) with data-driven adaptive thresholding (median, Otsu, mean, std) fitted on training data only. |
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

All original reproduced experiments were executed with random seed **42** using the current environment (`.\.venv\Scripts\python.exe`). Output metrics, confusion matrices, and execution logs are archived in `results/final/`:

| Experiment | Dataset | Mode | Solver Mode | Train / Test Docs | Features (TF-IDF $\to$ FD-FFO $\to$ TL-HHO) | Accuracy | Weighted F1 | Rand Index (RI) | NMI | Runtime |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment A (Smoke Test)** | 20 Newsgroups | `smoke_test` | `exact` | 160 / 40 | 100 $\to$ 100 $\to$ 31 | **15.00%** | **13.10%** | **86.03%** | **0.6251** | 1.31s |
| **Experiment B (Exact Benchmark)** | 20 Newsgroups | `benchmark` | `exact` | 800 / 200 | 300 $\to$ 300 $\to$ 221 | **45.50%** | **46.71%** | **92.21%** | **0.5261** | 2.63s |
| **Experiment C (Nyström Benchmark)** | 20 Newsgroups | `benchmark` | `nystrom` ($m=200$) | 800 / 200 | 300 $\to$ 300 $\to$ 221 | **40.50%** | **42.60%** | **90.18%** | **0.4919** | 2.54s |
| **Experiment D (Full Official Split)** | 20 Newsgroups | `full` | `nystrom` ($m=200$) | 11,314 / 7,532 | 100 $\to$ 100 $\to$ 97 | **24.23%** | **22.30%** | **81.11%** | **0.2103** | 18.89s |
| **Experiment E (Web-Snippets)** | Web-Snippets | `benchmark` | `exact` | 384 / 96 | 300 $\to$ 300 $\to$ 134 | **52.08%** | **52.36%** | **81.47%** | **0.3899** | 1.56s |

*Note: For stratified splits with equal class representations (10 samples/class in benchmark), weighted and macro averages are mathematically identical. Master metrics summary table: `results/final/final_experiment_metrics.csv`.*

---

## 5. Proposed Research Extension: Adaptive Density-Aware Feature Selection (ADAFS)

### Motivation
In Section 4.1, Pitchandi & Grace state that features with a *"fuzzy density score below 0.4 are eliminated"*, which supposedly removes approximately $70\%$ of features. However, the paper **does not provide any mathematical definition or equation** for this score. Naive heuristics (such as setting the product $\rho_j \cdot \delta_j \ge 0.4$ or dropping a fixed $70\%$ of features) either catastrophically collapse performance or discard critical discriminative terms.

Rather than claiming to reconstruct an unstated formula, we propose **ADAFS** as an explicit, transparent, and reproducible method. Comprehensive mathematical derivations are documented in [`docs/adaptive_density_feature_selection.md`](docs/adaptive_density_feature_selection.md).

### Mathematical Formulation
ADAFS computes a composite score strictly on the training partition ($X_{train}, y_{train}$):

1. **Supervised Feature Relevance ($Rel_j$):**
   Using multiclass one-way ANOVA between-class separation ($\eta_j^2$), completely non-parametric for nominal classes:
   $$Rel_j = \eta_j^2 = \frac{SSB_j}{SST_j} = \frac{\sum_{c=1}^C N_c (\bar{x}_{c,j} - \bar{x}_j)^2}{\sum_{i=1}^N (x_{ij} - \bar{x}_j)^2} \in [0, 1]$$
   Zero-variance features safely yield $Rel_j = 0$.

2. **Unsupervised Density-Based Redundancy ($Red_j$):**
   Using the normalized Fruit Fly local density ($\rho_j$) derived from pairwise cosine/Euclidean distance $D$:
   $$Red_j = \frac{\rho_j}{\max_{k} \rho_k} \in [0, 1], \quad \rho_j = \sum_{i \neq j} \chi(D_{ji} < d_c)$$

3. **Combined ADAFS Score ($S_j$):**
   Balancing discriminative power against feature redundancy with trade-off parameter $\lambda \in [0, 1]$ (default $\lambda = 0.5$):
   $$\tilde{S}_j = Rel_j - \lambda \cdot Red_j, \qquad S_j = \frac{\tilde{S}_j - \min_k \tilde{S}_k}{\max_k \tilde{S}_k - \min_k \tilde{S}_k + \epsilon} \in [0, 1]$$

4. **Data-Driven Adaptive Thresholding:**
   Instead of an arbitrary constant cutoff ($0.4$), ADAFS calculates an adaptive threshold $\tau$ dynamically from the empirical distribution of $S$:
   - `percentile` (default, 50th percentile / median)
   - `mean` ($\tau = \bar{S}$)
   - `otsu` (maximal between-class variance threshold on $S$)
   - `std` ($\tau = \bar{S} + k \cdot \sigma_S$)
   - Deterministic fallback guarantees at least 1 feature is always selected ($\ge 1$).

### Controlled Benchmark Comparison (20 Newsgroups Benchmark)

Under identical conditions ($N=1,000$ documents, 800 train / 200 test, stratified across 20 classes, random seed 42, exact SVR solver):

| Selection Mode | Display Name | Initial TF-IDF | After Stage 1 | Stage 1 Removal | Final TL-HHO Features | Accuracy | Weighted F1 | Rand Index (RI) | NMI | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `passthrough` | Faithful Baseline (Passthrough) | 300 | 300 | 0.0% | 221 | **45.50%** | **46.71%** | **92.21%** | **0.5261** | 6.49s |
| `density_score` | Density-Delta Heuristic ($\bar{\rho} \cdot \bar{\delta} \ge 0.4$) | 300 | 1 | 99.7% | 1 | 5.00% | 0.48% | 4.52% | 0.0000 | 3.67s |
| `fixed_ratio` | Fixed Ratio Heuristic (Drop 70%) | 300 | 90 | 70.0% | 39 | 19.50% | 17.85% | 77.44% | 0.3367 | 5.70s |
| **`adaptive_density`** | **Proposed ADAFS (Novelty)** | **300** | **150** | **50.0%** | **65** | **37.50%** | **40.48%** | **86.74%** | **0.4711** | **5.83s** |

*Artifacts: [`results/adaptive_density/adaptive_density_comparison.csv`](results/adaptive_density/adaptive_density_comparison.csv) and [`results/adaptive_density/adaptive_density_comparison.png`](results/adaptive_density/adaptive_density_comparison.png).*

### Key Findings & Empirical Analysis
1. **Catastrophic Failure of Uncalibrated Density Cutoffs:** Naively filtering on the product of local density and expanse ($\bar{\rho} \cdot \bar{\delta} \ge 0.4$) removes $99.7\%$ of terms, leaving only 1 feature and collapsing accuracy to $5.00\%$ (random guess) and NMI to $0.0$.
2. **Fixed-Ratio Information Loss:** Arbitrarily discarding the lowest $70\%$ of features without relevance weighting (`fixed_ratio`) drops accuracy from $45.50\%$ down to $19.50\%$ and NMI to $0.3367$.
3. **ADAFS Effectiveness:** ADAFS achieves **37.50% Accuracy**, **86.74% Rand Index**, and **0.4711 NMI** while pruning the feature space by **78.3% overall** ($300 \to 65$ final terms). This represents a nearly twofold improvement in accuracy over the fixed-ratio heuristic ($37.50\%$ vs $19.50\%$) while requiring only $65$ features compared to $221$ in the unpruned passthrough pipeline.

---

## 6. Execution & Reproduction Commands

### Run Full Test Suite (39 Tests)
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```
*(Runs all 39 unit, mathematical proof, data-leakage, and ADAFS validation tests.)*

### Run Original Baseline Benchmark
```powershell
.\.venv\Scripts\python.exe main.py --mode benchmark --dataset 20newsgroups --fdffo_mode passthrough --mapping_mode one_hot --max_features 300 --pop_size 10 --max_iter 5 --seed 42 --results_dir results/final/20ng_exact
```

### Run Proposed ADAFS Pipeline
```powershell
.\.venv\Scripts\python.exe main.py --mode benchmark --dataset 20newsgroups --fdffo_mode adaptive_density --adafs_lambda 0.5 --adafs_threshold_method percentile --adafs_threshold_value 50.0 --seed 42 --results_dir results/adaptive_density/benchmark_adafs
```

### Run 4-Way Controlled Benchmark & Figure Generation
```powershell
.\.venv\Scripts\python.exe scripts\run_adaptive_density_experiment.py
```

---

## 7. Known Paper Discrepancies & Scientific Limitations

1. **Undefined "Fuzzy Density Score":**
   Section 4.1 claims features with *"fuzzy density score below 0.4 are eliminated"*, but **never provides a mathematical formula** defining this score in the text or algorithms. To maintain scientific integrity without fabricating equations, `src/fd_ffo.py` defaults to `scoring_mode='passthrough'` (keeps all features), provides documented heuristics, and introduces ADAFS as an explicit novel solution.
2. **Scalar SVR to Multiclass PNN Interface:**
   The paper formulates standard 1D scalar LS-SVR (Eqs. 32–40) alongside standard vector PNN formulas (Eqs. 42–43), omitting the mathematical interface between them for multiclass datasets (e.g. 20 classes). The implementation resolves this via `mapping_mode='one_hot'`, fitting LS-SVR with a continuous one-hot indicator matrix to avoid imposing artificial ordinal relationships on nominal categories.
3. **Table 7 Scalability Contradiction:**
   The paper claims an execution time of 19.8 seconds on 20,000 documents on an Intel Core i3 PC. As proven mathematically, an exact dual KKT system on 20,000 documents requires allocating a $3.05 \text{ GB}$ dense float64 matrix and $\approx 2.67 \times 10^{12}$ FLOPs, which requires several minutes on a CPU. The authors could not have used an exact dual solver for the full dataset; our opt-in Nyström low-rank approximation ($m=200$) reproduces sub-20 second execution times safely without memory exhaustion.
4. **Finite-Sample Bias in NMI:**
   The smoke test exhibits higher NMI ($0.6251$) than the benchmark ($0.5261$) due to the well-known positive finite-sample bias of Shannon mutual information when evaluating very few samples per class ($N/K = 2.0$ vs. $10.0$).

---

## 8. Figures and Reproducibility Analysis

Publication-quality research presentation figures, component-level reproducibility status audits, and automated generation scripts are archived under [`results/figures/`](results/figures/) and [`results/adaptive_density/`](results/adaptive_density/):

- **Master Experimental Results Table:** [`results/final/final_experiment_metrics.csv`](results/final/final_experiment_metrics.csv)
- **Reproducibility Status Matrix:** [`results/figures/reproducibility_status.csv`](results/figures/reproducibility_status.csv)
- **Detailed Figure Documentation & Catalog:** [`results/figures/README.md`](results/figures/README.md)
- **Figure Generation Script:** [`scripts/generate_figures.py`](scripts/generate_figures.py)
- **ADAFS Novelty Benchmark Chart:** [`results/adaptive_density/adaptive_density_comparison.png`](results/adaptive_density/adaptive_density_comparison.png)

### Generated Research Figures:
1. **[Figure 1: Classification Accuracy Comparison](results/figures/01_accuracy_comparison.png)** — Benchmarking supervised test accuracy across all 5 final pipelines against random-guess baselines.
2. **[Figure 2: Clustering Metrics Comparison (Rand Index & NMI)](results/figures/02_ri_nmi_comparison.png)** — Evaluating pairwise Rand Index (percentage scale) and Normalized Mutual Information (decimal scale).
3. **[Figure 3: Feature Selection Reduction Progression](results/figures/03_feature_selection_reduction.png)** — Visualizing term counts across TF-IDF $\to$ FD-FFO $\to$ TL-HHO stages.
4. **[Figure 4: Exact Dual Solver vs. Nyström Low-Rank Approximation](results/figures/04_exact_vs_nystrom.png)** — Comparative evaluation of classification accuracy, Rand Index, NMI, and execution time under an identical 20 Newsgroups benchmark split.
5. **[Figure 5: Scientific Audit — Paper Claims vs. Reproduced Results](results/figures/05_paper_vs_reproduced.png)** — Demonstrates close reproduction of paper Table 6 Rand Index (92.21% vs 92.34%), compatible NMI ranges, and highlights the non-equivalence of the paper's $>95\%$ accuracy claim under undefined "Chunk Size" subsets.
6. **[ADAFS Novelty Comparison](results/adaptive_density/adaptive_density_comparison.png)** — Head-to-head empirical comparison of ADAFS against Passthrough, Density-Delta heuristic, and Fixed-Ratio heuristic across Accuracy, Rand Index, NMI, and feature retention rates.

---

## 9. Project Structure

```
DocumentClusteringProject/
├── main.py                        # Pipeline entry point & CLI orchestrator
├── README.md                      # Project documentation & reproduction guide
├── docs/
│   └── adaptive_density_feature_selection.md # ADAFS mathematical specification
├── scripts/
│   ├── generate_figures.py        # Automated publication figure generator
│   └── run_adaptive_density_experiment.py # ADAFS 4-way controlled benchmark
├── src/
│   ├── data_loader.py             # Offline 20 Newsgroups & Web-Snippets loaders
│   ├── preprocessing.py          # Lowercasing, punctuation & whitespace cleaner
│   ├── feature_extraction.py      # TF-IDF vectorizer wrapper (fit on train ONLY)
│   ├── fd_ffo.py                  # FD-FFO & ADAFS density-aware feature selection
│   ├── tl_hho.py                  # Teaching-Learning Harris Hawks Optimization
│   └── svr_pnn.py                 # SVR-PNN (Dual exact KKT & Primal Nyström)
├── tests/
│   ├── test_data_loader.py        # Dataset loading & stratification tests
│   ├── test_fd_ffo.py             # FD-FFO distance, density & mode tests
│   ├── test_adafs.py              # ADAFS mathematical proofs & unit tests
│   ├── test_tl_hho.py             # TL-HHO position mapping & fitness tests
│   ├── test_svr_pnn.py            # SVR-PNN KKT, kernel & numerical stability tests
│   ├── test_pipeline.py           # End-to-end data leakage prevention test
│   └── test_mathematical_validation.py # Hand-calculated proofs & invariant tests
├── results/
│   ├── final/                     # FROZEN master experimental deliverables
│   │   ├── final_experiment_metrics.csv # Master metrics summary table
│   │   ├── smoke_test/            # Experiment A outputs
│   │   ├── 20ng_exact/            # Experiment B outputs
│   │   ├── 20ng_nystrom/          # Experiment C outputs
│   │   ├── 20ng_full_nystrom/     # Experiment D outputs
│   │   └── web_snippets/          # Experiment E outputs
│   ├── figures/                   # Presentation figures & reproducibility status
│   │   ├── 01_accuracy_comparison.png
│   │   ├── 02_ri_nmi_comparison.png
│   │   ├── 03_feature_selection_reduction.png
│   │   ├── 04_exact_vs_nystrom.png
│   │   ├── 05_paper_vs_reproduced.png
│   │   ├── reproducibility_status.csv
│   │   └── README.md
│   └── adaptive_density/          # Proposed ADAFS novelty experimental outputs
│       ├── adaptive_density_comparison.csv
│       ├── adaptive_density_comparison.png
│       ├── passthrough/
│       ├── density_score/
│       ├── fixed_ratio/
│       └── adaptive_density/
└── backups/                       # Timestamped pre-modification snapshots
```
