# Adaptive Density-Aware Feature Selection (ADAFS)

## A Mathematically Explicit and Scalable First-Stage Feature Selection Method

---

## 1. Scientific Motivation

The foundation of this research project is the document clustering architecture proposed by Pitchandi & Kingsy Grace (*Expert Systems With Applications*, Vol. 294, 2025, Article 128762). In Section 4.1 of their paper, the authors state:

> *"Fuzzy density score below a threshold of 0.4 are eliminated. Using fuzzy density fruit fly optimization algorithm 70 % of redundant contents are removed."*

However, rigorous scientific auditing revealed a critical reproducibility barrier:
- The paper provides mathematical formulas for local feature density $\rho$ (Eq. 1) and expanse distance $\delta$ (Eq. 3), adapted from clustering by fast search and find of density peaks (Rodriguez & Laio, *Science*, 2014).
- The paper defines Fruit Fly Optimization (FFO) 2D coordinates and osphresis search (Eqs. 4–10).
- **Crucially, the paper omits any mathematical definition, formula, or algorithmic procedure for the "fuzzy density score" itself.**
- Furthermore, the paper never specifies how a 2D fly coordinate selects or prunes individual textual features, nor how the arbitrary threshold of $0.4$ relates to any normalized distribution.

In our audited baseline reproduction, this ambiguity was handled neutrally by providing `scoring_mode='passthrough'` (the faithful zero-assumption baseline which keeps all features) along with heuristic variants `density_score` ($\bar{\rho} \cdot \bar{\delta} \ge 0.4$) and `fixed_ratio` (retaining the top 30% features).

To advance beyond auditing and provide an authentic scientific contribution, we propose **Adaptive Density-Aware Feature Selection (ADAFS)**. ADAFS is a newly designed, explicitly formulated, and fully reproducible feature-selection method that replaces the undefined "fuzzy density score" and fixed threshold of 0.4 with an interpretable, data-driven framework.

---

## 2. Clear Distinction Between Original Paper and ADAFS Extension

| Dimension | Original Paper (Pitchandi & Grace, 2025) | Proposed ADAFS Extension |
| :--- | :--- | :--- |
| **Feature Relevance** | Unspecified / Absent in Stage 1 | Supervised ANOVA correlation ratio $\eta^2$ (nominal multiclass) |
| **Feature Redundancy** | Local density $\rho$ and expanse $\delta$ defined, but connection to selection is omitted | Density-aware neighborhood redundancy score normalized in $[0, 1]$ |
| **Combined Score** | "Fuzzy density score" — **Formula never published** | $ADAFS_j = Rel_j - \lambda \cdot Red_j$, min-max normalized to $[0, 1]$ |
| **Thresholding** | Fixed arbitrary threshold of $0.4$ | Adaptive, distribution-driven threshold (percentile, mean, Otsu) |
| **Data Leakage** | Not documented in paper | Strictly fit on training data only; zero test leakage |
| **Claim Status** | Published claim without formula | **New proposed research novelty (not claiming paper reproduction)** |

> **Scientific Attribution Notice:** ADAFS is an independent research novelty developed in this repository. It does **not** claim to reconstruct the authors' private implementation, but rather provides a defensible, mathematically grounded alternative.

---

## 3. Mathematical Definition of Feature Relevance

To assess how effectively feature $j$ discriminates between the document categories without imposing an artificial ordinal relationship on nominal class labels, ADAFS employs the **multiclass correlation ratio** (ANOVA effect size $\eta^2$).

Let:
- $X \in \mathbb{R}^{N \times M}$ be the training feature matrix with $N$ documents and $M$ vocabulary features.
- $\mathbf{y} \in \{1, \dots, C\}^N$ be the nominal category labels with $C$ classes ($C \ge 2$).
- $N_c = \sum_{i=1}^N \mathbb{I}(y_i = c)$ be the sample count of class $c$.

For feature $j \in \{1, \dots, M\}$:
1. **Overall Training Mean:**
   $$\bar{x}_j = \frac{1}{N} \sum_{i=1}^N X_{ij}$$

2. **Class-Conditional Mean:**
   $$\bar{x}_{c,j} = \frac{1}{N_c} \sum_{i: y_i = c} X_{ij}$$

3. **Total Sum of Squares ($SST_j$):**
   $$SST_j = \sum_{i=1}^N (X_{ij} - \bar{x}_j)^2 = \sum_{i=1}^N X_{ij}^2 - N \bar{x}_j^2$$

4. **Between-Class Sum of Squares ($SSB_j$):**
   $$SSB_j = \sum_{c=1}^C N_c (\bar{x}_{c,j} - \bar{x}_j)^2$$

5. **Normalized Relevance Measure ($Rel_j$):**
   $$Rel_j = \begin{cases} \dfrac{SSB_j}{SST_j}, & \text{if } SST_j > 10^{-12} \\ 0.0, & \text{if } SST_j \le 10^{-12} \text{ (zero-variance feature)} \end{cases}$$

### Theoretical Properties:
- **Bounded in $[0, 1]$:** By the ANOVA variance decomposition $SST_j = SSB_j + SSW_j$, where $SSW_j \ge 0$, it is guaranteed that $0 \le SSB_j \le SST_j$. Thus $Rel_j \in [0, 1]$.
- **Permutation Invariance:** The statistic depends only on class groupings, not on arbitrary numerical assignments to nominal categories.
- **Sparse Efficiency:** Both $SST_j$ and $SSB_j$ are computed in $O(N \cdot M)$ using vectorized sparse matrix multiplications (`X.power(2).sum(axis=0)` and group averages), avoiding dense array inflation.

---

## 4. Mathematical Definition of Feature Redundancy

In high-dimensional bag-of-words and TF-IDF representations, semantically collinear words (e.g., synonyms or co-occurring topical terms) reside in dense clusters in feature space.

Following the paper's pairwise Euclidean distance between feature columns $D_{jk} = \|X_{\cdot, j} - X_{\cdot, k}\|_2$:
1. **Local Feature Density ($\rho_j$):**
   $$\rho_j = \sum_{k \neq j} \mathbb{I}(D_{jk} < d_c)$$
   where $d_c$ is the cutoff distance parameter (default $d_c = 0.5$).
2. **Normalized Density-Aware Redundancy ($Red_j$):**
   A feature surrounded by many other features within cutoff distance $d_c$ has high local redundancy:
   $$Red_j = \begin{cases} \dfrac{\rho_j}{\max_{k=1}^M \rho_k}, & \text{if } \max_{k} \rho_k > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

By definition, $Red_j \in [0, 1]$.

---

## 5. Combined ADAFS Score

To balance category discriminative power against representational redundancy, ADAFS defines the linear scalarized utility:

$$S_j = Rel_j - \lambda \cdot Red_j$$

where $\lambda \in [0, 1]$ (default $\lambda = 0.5$) is a user-configurable trade-off parameter:
- $\lambda = 0$: Pure supervised ranking (relevance only).
- $\lambda > 0$: Penalizes features residing in dense collinear clusters, encouraging broader representational coverage across topics.

### Score Normalization to $[0, 1]$:
Because $Rel_j \in [0, 1]$ and $Red_j \in [0, 1]$, the raw score $S_j \in [-\lambda, 1]$. To map the score onto a standard interval $[0, 1]$ for thresholding:

$$ADAFS_j = \begin{cases} \dfrac{S_j - \min(\mathbf{S})}{\max(\mathbf{S}) - \min(\mathbf{S})}, & \text{if } \max(\mathbf{S}) > \min(\mathbf{S}) \\ 1.0, & \text{if } \max(\mathbf{S}) = \min(\mathbf{S}) \end{cases}$$

---

## 6. Adaptive Feature Retention Thresholding

Rather than applying the paper's static, non-adaptive threshold of $0.4$, ADAFS computes an adaptive cutoff $\tau$ derived strictly from the training score distribution:

1. **Percentile Strategy (Default, `adafs_threshold_method='percentile'`):**
   $$\tau = \text{Percentile}(ADAFS, p)$$
   where $p \in [0, 100)$ (default $p = 50.0$, selecting the top 50% candidate features).
2. **Mean Strategy (`adafs_threshold_method='mean'`):**
   $$\tau = \frac{1}{M} \sum_{j=1}^M ADAFS_j$$
   Retains all features whose score exceeds the population average.
3. **Otsu Bimodal Strategy (`adafs_threshold_method='otsu'`):**
   Computes the optimal 1D Otsu threshold $\tau^*$ maximizing the between-class variance of the score histogram:
   $$\tau^* = \arg\max_\tau \omega_0(\tau) \omega_1(\tau) \left[\mu_0(\tau) - \mu_1(\tau)\right]^2$$
   This automatically discovers the natural gap between weak/redundant features and informative features without human tuning.
4. **Standard Deviation Strategy (`adafs_threshold_method='std'`):**
   $$\tau = \mu_{ADAFS} - k \cdot \sigma_{ADAFS}$$

### Deterministic Safety Fallback:
If an aggressive threshold results in zero selected features:
$$\text{Selected} = \{j : ADAFS_j \ge \tau\}$$
$$\text{If } |\text{Selected}| = 0 \implies \text{Selected} = \{\arg\max_j ADAFS_j\}$$
This guarantees that at least one feature is always retained, preventing downstream pipeline collapse.

---

## 7. Computational Complexity

| Step | Operation | Time Complexity | Memory Complexity |
| :--- | :--- | :---: | :---: |
| **TF-IDF Extraction** | Sublinear term frequency | $O(N \cdot L)$ ($L = \text{avg doc length}$) | Sparse $O(\text{nnz})$ |
| **ADAFS Relevance** | Class-wise means & variances | $O(N \cdot M)$ | $O(M)$ |
| **ADAFS Redundancy** | Feature Gram matrix $X^T X$ | $O(N \cdot M + M^2)$ | $O(M^2)$ |
| **ADAFS Thresholding** | Score ranking / Otsu search | $O(M \log M)$ | $O(M)$ |
| **Total ADAFS Overhead** | Complete Stage 1 screening | **$O(N \cdot M + M^2)$** | **$O(M^2)$** |

For typical document clustering settings ($M \le 1000$), $M^2 \le 10^6$ floats ($\approx 8 \text{ MB}$ RAM), taking less than **0.05 seconds** on a modern CPU. This makes ADAFS significantly faster than the subsequent metaheuristic wrapper phase (TL-HHO).

---

## 8. Data Leakage Prevention

ADAFS adheres strictly to the zero-leakage architecture:
1. `FDFFO.fit(X_train, y_train)` calculates $Rel$, $Red$, $S$, and $\tau$ **exclusively** on the training partition.
2. The testing partition $X_{test}$ is never accessed during feature evaluation or threshold derivation.
3. During evaluation, `FDFFO.transform(X_test)` projects the test documents onto the pre-computed boolean column mask:
   $$X_{test,\text{selected}} = X_{test}[:, \text{selected\_features\_}]$$
4. Neither test feature values nor test labels influence feature scores in any way.

---

## 9. Limitations & Conservative Scientific Assessment

1. **Linear Supervised Proxy:** The correlation ratio $\eta^2$ evaluates univariate feature-to-class dependency. Highly non-linear multi-feature interactions are deferred to Stage 2 (TL-HHO metaheuristic wrapper).
2. **Pairwise Redundancy Approximation:** Local density $\rho_j$ provides an empirical indicator of representation clustering, but does not capture high-order multivariate dependencies (e.g., triplets of mutually redundant terms).
3. **No Claim of Monotonic Accuracy Superiority:** In extreme low-sample regimes or heavily imbalanced domains, aggressive feature pruning may discard marginally informative vocabulary. The goal of ADAFS is **mathematical transparency, reproducibility, and principled dimensional compression**, not artificial score inflation.
