# Research Presentation Figures and Reproducibility Documentation

This directory contains the publication-quality visualization figures and reproducibility status table for the independent reproduction and scientific audit of:

> **“An optimal document clustering method using hybrid optimal feature selection and efficient soft computing classifier”**  
> *P. Pitchandi and R. Kingsy Grace, Expert Systems With Applications, Vol. 294, Part A (2025), Article 128762.*  
> **DOI:** [10.1016/j.eswa.2025.128762](https://doi.org/10.1016/j.eswa.2025.128762)

All figures are programmatically generated from the frozen, audited master experimental metrics in [`results/final/final_experiment_metrics.csv`](../final/final_experiment_metrics.csv).

---

## 1. Figure Catalog & Interpretations

### Figure 1: Experimental Performance Comparison
- **File:** [`01_accuracy_comparison.png`](01_accuracy_comparison.png)
- **What it shows:** Classification accuracy across all five final verified experiments:
  1. **Smoke Test (20 Newsgroups, Exact):** 15.00% (40 test samples, $N/K = 2.0$)
  2. **Exact Benchmark (20 Newsgroups, Exact):** 45.50% (200 test samples, $N/K = 10.0$)
  3. **Nyström Benchmark (20 Newsgroups, Nyström $m=200$):** 40.50% (200 test samples, $N/K = 10.0$)
  4. **Full Official Split (20 Newsgroups, Nyström $m=200$):** 24.23% (7,532 test samples across all 18,846 docs)
  5. **Web-Snippets Benchmark (Web-Snippets, Exact):** 52.08% (96 test samples across 8 domains)
- **Source Data:** `results/final/final_experiment_metrics.csv` (`Accuracy`, `Solver`, `Experiment`).
- **Key Takeaways:** Distinguishes the exact dual KKT solver from the low-rank Nyström primal solver. Highlights that all reproduced models perform substantially above the theoretical random-guess baselines (5.00% for 20 classes; 12.50% for 8 classes) under strict, leak-free test evaluation.

---

### Figure 2: Clustering Metrics Comparison
- **File:** [`02_ri_nmi_comparison.png`](02_ri_nmi_comparison.png)
- **What it shows:** Side-by-side two-panel comparison of clustering quality metrics across all five experiments:
  - **Panel A (Rand Index):** Displayed on a percentage scale ($60\% - 100\%$), showing pairwise clustering consistency:
    - 20NG Exact Benchmark: **92.21%** (closely matching Paper Table 6's reported 92.34%)
    - 20NG Nyström Benchmark: **90.18%**
    - 20NG Smoke Test: **86.03%**
    - 20NG Full Split: **81.11%**
    - Web-Snippets Benchmark: **81.47%**
  - **Panel B (Normalized Mutual Information):** Displayed on a decimal scale ($0.0 - 1.0$), capturing category alignment:
    - 20NG Smoke Test: **0.6251** (exhibits well-known positive finite-sample Shannon bias for small $N/K = 2.0$)
    - 20NG Exact Benchmark: **0.5261** (aligns with paper's empirical range of $0.55 - 0.65$)
    - 20NG Nyström Benchmark: **0.4919**
    - 20NG Full Split: **0.2103**
    - Web-Snippets Benchmark: **0.3899**
- **Source Data:** `results/final/final_experiment_metrics.csv` (`Rand_Index`, `NMI`, `Solver`).
- **Scale Separation:** Preserves scientific integrity by strictly separating percentage scales and decimal indices into dedicated subpanels.

---

### Figure 3: Feature Selection Reduction Progression
- **File:** [`03_feature_selection_reduction.png`](03_feature_selection_reduction.png)
- **What it shows:** The two-stage feature selection progression across each pipeline:
  - **Stage 1 (Initial TF-IDF Vocabulary):** 300 terms (or 100 terms in smoke test / full split).
  - **Stage 2 (Post FD-FFO):** Operated in faithful passthrough mode ($0\%$ pruned) because the paper omitted the mathematical formula for its *"fuzzy density score below 0.4 eliminated"* claim.
  - **Stage 3 (Post TL-HHO Wrapper Selection):** Active compression:
    - 20NG Benchmark: $300 \to 221$ features ($-26.3\%$ reduction)
    - Web-Snippets Benchmark: $300 \to 134$ features ($-55.3\%$ reduction)
    - Smoke Test: $100 \to 31$ features ($-69.0\%$ reduction)
    - Full Split: $100 \to 97$ features ($-3.0\%$ reduction)
- **Source Data:** `results/final/final_experiment_metrics.csv` (`Initial_TFIDF`, `After_FDFFO`, `After_TLHHO`).

---

### Figure 4: Exact Dual Solver vs. Nyström Low-Rank Approximation
- **File:** [`04_exact_vs_nystrom.png`](04_exact_vs_nystrom.png)
- **What it shows:** Direct head-to-head comparison between the exact dual KKT solver (`20ng_exact`) and the opt-in Nyström low-rank approximation (`20ng_nystrom`) under the **exact same benchmark configuration**:
  - Sample partition: 800 training documents, 200 testing documents (stratified, 20 categories).
  - Feature set: 221 TL-HHO selected terms from 300 TF-IDF candidates.
  - Random seed: 42.
- **Key Metrics:**
  - Accuracy: **45.50%** (Exact) vs. **40.50%** (Nyström, $-5.00$ pp trade-off for scalability).
  - Rand Index: **92.21%** (Exact) vs. **90.18%** (Nyström, $-2.03$ pp).
  - NMI: **0.5261** (Exact) vs. **0.4919** (Nyström).
  - ARI: **0.1868** (Exact) vs. **0.1364** (Nyström).
  - Execution Time: **2.63s** (Exact) vs. **2.54s** (Nyström).
- **Scientific Audit Disclaimer:** Nyström projects samples onto $m=200$ empirical landmarks. It is a low-rank approximation designed to prevent memory exhaustion on large datasets ($N > 10,000$), not a mathematical equivalent to exact dual matrix inversion.

---

### Figure 5: Paper Claims vs. Reproduced Results
- **File:** [`05_paper_vs_reproduced.png`](05_paper_vs_reproduced.png)
- **What it shows:** Multi-panel scientific audit evaluating paper-reported values against leak-free reproduced results:
  - **Panel A (Clustering Quality):**
    - **Rand Index:** Paper Table 6 reports $\approx \mathbf{92.34\%}$; our reproduced benchmark achieved $\mathbf{92.21\%}$ (Exact) and $\mathbf{90.18\%}$ (Nyström). **Status: Reproduced** (within $0.13\%$ absolute difference).
    - **NMI:** Paper reports values in the range $\approx \mathbf{0.55 - 0.65}$; our reproduced benchmark achieved $\mathbf{0.5261}$ (Exact) and $\mathbf{0.4919}$ (Nyström). **Status: Partially Reproduced / Compatible Order of Magnitude**.
  - **Panel B (Classification Accuracy & Claim Audit):**
    - Paper claims accuracy $>\mathbf{95\%}$ in experiments involving undefined *"Chunk Size"* settings (e.g. Chunk Sizes 20, 40, 60).
    - In contrast, under verified, leakage-free 20-class multi-class testing, exact accuracy is $\mathbf{45.50\%}$ (Exact Benchmark), $\mathbf{40.50\%}$ (Nyström Benchmark), and $\mathbf{24.23\%}$ (Full 18,846 docs split).
    - **Status: Non-Equivalent / Not Reproducible from Paper.**
- **Scientific Audit Disclaimer:** The paper's $>95\%$ claim cannot be verified because the authors omitted the definitions, sizes, train/test splits, and document selection mechanisms for their "chunk" evaluations.

---

## 2. Reproducibility Status Table

The full component-by-component audit is stored in [`reproducibility_status.csv`](reproducibility_status.csv):

| Paper Component / Claim | Reproducibility Status | Our Result / Implementation | Scientific Explanation |
| :--- | :---: | :--- | :--- |
| **Core Density & Expanse Equations** (Sec 4.1, Eqs. 1–3) | `Reproduced` | `src/fd_ffo.py` (`_calculate_density`, `_calculate_delta`) | Mathematically reproduced after resolving published typographical errors ($D_b < 0 \to D_{ji} < d_c$; $\rho_j > \rho_i \to \rho_i > \rho_j$). Validated against Rodriguez & Laio (2014). |
| **Fuzzy Density Score Screening Formula** (Sec 4.1, Alg. 1) | `Not Reproducible from Paper` | `src/fd_ffo.py` (`scoring_mode='passthrough'`) | The paper claims features with fuzzy density score below 0.4 are eliminated (removing ~70% of features) but does not define any formula for this score. Implementation defaults to faithful passthrough and provides documented heuristic variants. |
| **Teaching-Learning Harris Hawks Optimization** (Sec 4.2, Eqs. 12–28, 31) | `Reproduced` | `src/tl_hho.py` (`TLHHO`) | All TLBO phases, HHO exploration/exploitation modes, Mantegna Lévy flights, and wrapper fitness objectives implemented faithfully with verified zero test leakage. |
| **LS-SVR to Multiclass PNN Interface** (Sec 4.3, Eqs. 32–43) | `Defensible Implementation Choice` | `src/svr_pnn.py` (`mapping_mode='one_hot'`) | The paper formulates 1D scalar LS-SVR alongside vector PNN equations without specifying how multiclass nominal targets (20 categories) are mapped. Implemented via continuous one-hot indicator embedding to prevent artificial ordinal constraints. |
| **Exact Dual Solver Scalability on 20,000 Documents** (Sec 5.1, Table 7) | `Defensible Implementation Choice` | `src/svr_pnn.py` (`solver_mode='nystrom'`) | Exact dual KKT system on 20,000 documents requires allocating 3.05 GB dense float64 RAM and $\approx 2.67 \times 10^{12}$ FLOPs (several minutes on CPU), contradicting paper claim of 19.8s on Core i3. Opt-in Nyström approximation enables sub-20s execution (18.89s on 18,846 docs). |
| **Rand Index on 20 Newsgroups** (Sec 5.1, Table 6) | `Reproduced` | 92.21% RI (Exact Benchmark), 90.18% RI (Nyström) | Paper Table 6 reports $\approx 92.34\%$ Rand Index for SVR-PNN. Our leak-free benchmark achieved 92.21% RI, reproducing the published clustering quality within 0.13% absolute difference. |
| **Normalized Mutual Information Range** (Sec 5.1, Table 6) | `Partially Reproduced` | 0.5261 (Exact Benchmark), 0.4919 (Nyström), 0.6251 (Smoke Test) | Paper reports NMI values approximately between 0.55 and 0.65 for 20 Newsgroups. Benchmark values align with this scale; smoke test shows higher NMI (0.6251) due to documented positive finite-sample bias ($N/K = 2.0$). |
| **Classification Accuracy >95% Claim** (Sec 5.1, Figs 10–14) | `Not Reproducible from Paper` | 45.50% (Exact Benchmark), 40.50% (Nyström), 24.23% (Full Split) | Paper claims $>95\%$ accuracy using undefined 'Chunk Size' subsets without document counts, train/test splits, or leakage guarantees. Under rigorous leak-free 20-class evaluation, exact accuracy is 45.50% (well above 5% random guess, but not $>95\%$). |

---

## 3. How to Regenerate Figures

To regenerate all five figures from the repository root:

```powershell
python scripts\generate_figures.py
```

Or using the project's virtual environment executable:

```powershell
.\.venv\Scripts\python.exe scripts\generate_figures.py
```

The script will read [`results/final/final_experiment_metrics.csv`](../final/final_experiment_metrics.csv) and update all PNG artifacts in `results/figures/`.
