"""
Figure Generation Script for Document Clustering Research Project
Paper: Pitchandi & Kingsy Grace, Expert Systems With Applications, 294 (2025), 128762

This script programmatically generates publication-quality figures from existing
experimental results archived in 'results/final/final_experiment_metrics.csv'.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# -----------------------------------------------------------------------------
# Configuration and Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_CSV = BASE_DIR / "results" / "final" / "final_experiment_metrics.csv"
OUTPUT_DIR = BASE_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

# Color palette
COLOR_EXACT = "#1f77b4"     # Deep Blue
COLOR_NYSTROM = "#ff7f0e"   # Warm Orange
COLOR_PAPER = "#2ca02c"     # Forest Green
COLOR_BASELINE = "#d62728"  # Crimson Red
COLOR_MUTED = "#7f7f7f"     # Slate Gray


def load_metrics():
    """Load and validate authoritative experimental metrics from CSV."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing master metrics CSV: {RESULTS_CSV}")
    
    df = pd.read_csv(RESULTS_CSV)
    expected_cols = [
        "Experiment", "Dataset", "Mode", "Solver", "Landmarks",
        "Train_Samples", "Test_Samples", "Classes", "Initial_TFIDF",
        "After_FDFFO", "After_TLHHO", "Accuracy", "Precision_Weighted",
        "Recall_Weighted", "F1_Weighted", "Rand_Index", "ARI", "NMI", "Runtime_Sec"
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {RESULTS_CSV}")
    return df


# -----------------------------------------------------------------------------
# Figure 1: Experimental Performance Comparison (Classification Accuracy)
# -----------------------------------------------------------------------------
def generate_figure_1(df):
    """
    Figure 1: Classification Accuracy Comparison across all 5 final experiments.
    Shows exact percentages and distinguishes solver modes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Map experiment names to human-readable publication labels
    display_names = {
        "smoke_test": "Smoke Test\n(20NG, Exact dual, 40 test)",
        "20ng_exact": "20NG Benchmark\n(Exact dual, 200 test)",
        "20ng_nystrom": "20NG Benchmark\n(Nyström m=200, 200 test)",
        "20ng_full_nystrom": "20NG Full Split\n(Nyström m=200, 7.5k test)",
        "web_snippets": "Web-Snippets Benchmark\n(Exact dual, 96 test)",
    }

    # Ordering for logical comparison: Smoke -> 20NG Exact -> 20NG Nystrom -> 20NG Full -> Web-Snippets
    order = ["smoke_test", "20ng_exact", "20ng_nystrom", "20ng_full_nystrom", "web_snippets"]
    df_ordered = df.set_index("Experiment").reindex(order).reset_index()

    labels = [display_names.get(x, x) for x in df_ordered["Experiment"]]
    accuracies_pct = df_ordered["Accuracy"] * 100.0
    solvers = df_ordered["Solver"].tolist()

    colors = [COLOR_EXACT if s == "exact" else COLOR_NYSTROM for s in solvers]
    bars = ax.bar(labels, accuracies_pct, color=colors, width=0.55, edgecolor="#222222", linewidth=1.2, zorder=3)

    # Add value annotations
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add random-guess theoretical baselines
    ax.axhline(5.0, color="#d62728", linestyle=":", linewidth=1.5, zorder=2,
               label="20-Class Random Guess Baseline (5.00%)")
    ax.axhline(12.5, color="#9467bd", linestyle=":", linewidth=1.5, zorder=2,
               label="8-Class Random Guess Baseline (12.50%)")

    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax.set_title("Figure 1: Experimental Classification Accuracy across Pipelines\n(Leakage-Free Held-Out Test Evaluation)", pad=15)
    ax.set_ylim(0, 65)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    # Custom legend for solver types and baselines
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_EXACT, edgecolor="#222222", label="Exact Dual KKT Solver"),
        Patch(facecolor=COLOR_NYSTROM, edgecolor="#222222", label="Nyström Primal Landmark Solver (m=200)"),
        plt.Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.5, label="20-Class Guess Baseline (5.0%)"),
        plt.Line2D([0], [0], color="#9467bd", linestyle=":", linewidth=1.5, label="8-Class Guess Baseline (12.5%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.95)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "01_accuracy_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Generated] {out_path}")


# -----------------------------------------------------------------------------
# Figure 2: Clustering Metrics Comparison (Rand Index & NMI)
# -----------------------------------------------------------------------------
def generate_figure_2(df):
    """
    Figure 2: Clustering Metrics Comparison (Rand Index and NMI) across experiments.
    Uses two clean subpanels to avoid mixing percentage scale [0-100%] and decimal scale [0-1].
    """
    order = ["smoke_test", "20ng_exact", "20ng_nystrom", "20ng_full_nystrom", "web_snippets"]
    df_ordered = df.set_index("Experiment").reindex(order).reset_index()

    short_labels = [
        "Smoke Test\n(20NG Exact)",
        "20NG Exact\n(Benchmark)",
        "20NG Nyström\n(Benchmark)",
        "20NG Full\n(Nyström)",
        "Web-Snippets\n(Exact)",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # --- Panel A: Rand Index (Percentage Scale: 0 - 100%) ---
    ri_pct = df_ordered["Rand_Index"] * 100.0
    colors_ri = [COLOR_EXACT if s == "exact" else COLOR_NYSTROM for s in df_ordered["Solver"]]
    bars1 = ax1.bar(short_labels, ri_pct, color=colors_ri, width=0.55, edgecolor="#222222", linewidth=1.2, zorder=3)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_ylabel("Rand Index (%)", fontweight="bold")
    ax1.set_title("(A) Rand Index (Clustering Pair Agreement)", pad=12)
    ax1.set_ylim(60, 100)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    # Reference line for Paper Table 6 RI claim (92.34%)
    ax1.axhline(92.34, color=COLOR_PAPER, linestyle="--", linewidth=1.8, zorder=4,
                label="Paper Table 6 Reported RI (92.34%)")
    ax1.legend(loc="lower right", framealpha=0.95)

    # --- Panel B: Normalized Mutual Information (Decimal Scale: 0.0 - 1.0) ---
    nmi_vals = df_ordered["NMI"]
    colors_nmi = ["#2b83ba" if s == "exact" else "#fdae61" for s in df_ordered["Solver"]]
    bars2 = ax2.bar(short_labels, nmi_vals, color=colors_nmi, width=0.55, edgecolor="#222222", linewidth=1.2, zorder=3)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Paper NMI typical range band [0.55 - 0.65]
    ax2.axhspan(0.55, 0.65, color=COLOR_PAPER, alpha=0.18, zorder=1,
                label="Paper Reported NMI Range (0.55–0.65)")
    ax2.axhline(0.55, color=COLOR_PAPER, linestyle=":", linewidth=1.2, zorder=2)
    ax2.axhline(0.65, color=COLOR_PAPER, linestyle=":", linewidth=1.2, zorder=2)

    ax2.set_ylabel("Normalized Mutual Information (Decimal [0, 1])", fontweight="bold")
    ax2.set_title("(B) Normalized Mutual Information (NMI)", pad=12)
    ax2.set_ylim(0.0, 0.80)
    ax2.legend(loc="upper right", framealpha=0.95)

    # Note on finite-sample bias for smoke test
    ax2.annotate("Finite-sample bias\n(N/K = 2.0)",
                 xy=(0, df_ordered.loc[df_ordered["Experiment"] == "smoke_test", "NMI"].values[0]),
                 xytext=(0.4, 0.72),
                 arrowprops=dict(facecolor="#333333", shrink=0.08, width=1, headwidth=6),
                 fontsize=8.5, ha="center", style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

    fig.suptitle("Figure 2: Clustering Quality Evaluation across Experiments\n(Rand Index in Percentage Scale vs. NMI in Decimal Scale)", y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "02_ri_nmi_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {out_path}")


# -----------------------------------------------------------------------------
# Figure 3: Feature Selection Reduction (TF-IDF -> FD-FFO -> TL-HHO)
# -----------------------------------------------------------------------------
def generate_figure_3(df):
    """
    Figure 3: Feature Selection Progression: Initial TF-IDF -> FD-FFO -> TL-HHO.
    Shows the exact number of features remaining at each stage across experiments.
    """
    order = ["smoke_test", "20ng_exact", "20ng_nystrom", "20ng_full_nystrom", "web_snippets"]
    df_ordered = df.set_index("Experiment").reindex(order).reset_index()

    experiment_labels = [
        "Smoke Test\n(20NG)",
        "20NG Exact\n(Benchmark)",
        "20NG Nyström\n(Benchmark)",
        "20NG Full\n(Nyström)",
        "Web-Snippets\n(Benchmark)",
    ]

    x = np.arange(len(experiment_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    f1 = df_ordered["Initial_TFIDF"].tolist()
    f2 = df_ordered["After_FDFFO"].tolist()
    f3 = df_ordered["After_TLHHO"].tolist()

    rects1 = ax.bar(x - width, f1, width, label="Stage 1: Initial TF-IDF Terms",
                    color="#4575b4", edgecolor="#222222", linewidth=1.0, zorder=3)
    rects2 = ax.bar(x, f2, width, label="Stage 2: Post FD-FFO (Passthrough Screening)",
                    color="#74add1", edgecolor="#222222", linewidth=1.0, zorder=3)
    rects3 = ax.bar(x + width, f3, width, label="Stage 3: Post TL-HHO (Selected Subset)",
                    color="#f46d43", edgecolor="#222222", linewidth=1.0, zorder=3)

    # Label heights
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            h = bar.get_height()
            ax.annotate(f"{int(h)}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add reduction % labels on Stage 3 bars
    for i, (b1, b3) in enumerate(zip(f1, f3)):
        reduction_pct = (1.0 - (b3 / b1)) * 100.0
        ax.annotate(f"-{reduction_pct:.1f}%",
                    xy=(x[i] + width, b3 / 2),
                    ha="center", va="center", color="white", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#d73027", edgecolor="none", alpha=0.9))

    ax.set_ylabel("Number of Features / Terms", fontweight="bold")
    ax.set_title("Figure 3: Two-Stage Feature Selection Progression (TF-IDF → FD-FFO → TL-HHO)\nShowing Feature Counts and Active TL-HHO Compression Ratios", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_labels)
    ax.set_ylim(0, 360)
    ax.legend(loc="upper right", framealpha=0.95)

    # Annotation about FD-FFO passthrough mode
    ax.text(0.02, 0.95,
            "Note: FD-FFO Stage 2 operated in faithful passthrough mode because\n"
            "the paper omitted the mathematical formula for 'fuzzy density score < 0.4'.\n"
            "TL-HHO Stage 3 performed active metaheuristic wrapper dimensionality reduction.",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffbf", edgecolor="#999999", alpha=0.85))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "03_feature_selection_reduction.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[Generated] {out_path}")


# -----------------------------------------------------------------------------
# Figure 4: Exact vs. Nyström Comparison on 20 Newsgroups Benchmark
# -----------------------------------------------------------------------------
def generate_figure_4(df):
    """
    Figure 4: Exact vs. Nyström Solvers on the Comparable 20 Newsgroups Benchmark.
    Compares Accuracy, Rand Index, ARI, NMI, and Runtime under identical configuration.
    """
    exact_row = df[df["Experiment"] == "20ng_exact"].iloc[0]
    nystrom_row = df[df["Experiment"] == "20ng_nystrom"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.2, 1.0]})

    # --- Panel A: Quality Metrics ---
    metrics = ["Accuracy (%)", "Rand Index (%)", "NMI (×100)", "ARI (×100)"]
    exact_vals = [
        exact_row["Accuracy"] * 100.0,
        exact_row["Rand_Index"] * 100.0,
        exact_row["NMI"] * 100.0,
        exact_row["ARI"] * 100.0,
    ]
    nystrom_vals = [
        nystrom_row["Accuracy"] * 100.0,
        nystrom_row["Rand_Index"] * 100.0,
        nystrom_row["NMI"] * 100.0,
        nystrom_row["ARI"] * 100.0,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars_exact = ax1.bar(x - width/2, exact_vals, width, label="Exact Dual KKT Solver",
                         color=COLOR_EXACT, edgecolor="#222222", linewidth=1.2, zorder=3)
    bars_nystrom = ax1.bar(x + width/2, nystrom_vals, width, label="Nyström Primal Landmark (m=200)",
                           color=COLOR_NYSTROM, edgecolor="#222222", linewidth=1.2, zorder=3)

    for bar in bars_exact:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    for bar in bars_nystrom:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_ylabel("Metric Value (Normalized Scale 0 - 100)", fontweight="bold")
    ax1.set_title("(A) Classification & Clustering Quality Metrics", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", framealpha=0.95)

    # --- Panel B: Runtime (Seconds) ---
    runtimes = [exact_row["Runtime_Sec"], nystrom_row["Runtime_Sec"]]
    solvers = ["Exact Dual\nKKT", "Nyström\n(m=200)"]
    bars_time = ax2.bar(solvers, runtimes, color=[COLOR_EXACT, COLOR_NYSTROM], width=0.5,
                        edgecolor="#222222", linewidth=1.2, zorder=3)

    for bar in bars_time:
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}s",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Execution Time (Seconds)", fontweight="bold")
    ax2.set_title("(B) Total Pipeline Runtime", pad=12)
    ax2.set_ylim(0, 3.5)

    # Scientific disclaimer caption
    fig.suptitle("Figure 4: Exact Dual Solver vs. Nyström Low-Rank Approximation\n(Identical 20 Newsgroups Benchmark: 800 Train / 200 Test Samples, 20 Classes, Seed 42)", y=1.02)

    fig.text(0.5, -0.05,
             "Scientific Audit Note: Both configurations evaluated identical features (221 terms) and stratified splits.\n"
             "Nyström is an opt-in low-rank empirical approximation (m=200 landmarks), not mathematically identical to exact dual inversion.",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "04_exact_vs_nystrom.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {out_path}")


# -----------------------------------------------------------------------------
# Figure 5: Paper Claims vs. Reproduced Results
# -----------------------------------------------------------------------------
def generate_figure_5(df):
    """
    Figure 5: Paper Reported Claims vs. Reproduced Experimental Results.
    Scientifically audited comparison distinguishing:
      - Directly reproduced (Rand Index: Table 6 reports ~92.34%, reproduced 92.21%)
      - Partially reproduced / compatible order of magnitude (NMI range ~0.55-0.65)
      - Not reproducible from paper (>95% accuracy claim under undefined 'chunk' configurations)
    """
    exact_row = df[df["Experiment"] == "20ng_exact"].iloc[0]
    nystrom_row = df[df["Experiment"] == "20ng_nystrom"].iloc[0]
    full_row = df[df["Experiment"] == "20ng_full_nystrom"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: Clustering Metrics (Rand Index & NMI) ---
    categories_a = ["Rand Index (%)\n[Directly Comparable]", "NMI (Decimal)\n[Order of Mag. Comparable]"]
    
    # Values
    paper_ri = 92.34
    repo_ri_exact = exact_row["Rand_Index"] * 100.0
    repo_ri_nystrom = nystrom_row["Rand_Index"] * 100.0

    paper_nmi_mid = 0.60  # Center of 0.55 - 0.65 range
    repo_nmi_exact = exact_row["NMI"]
    repo_nmi_nystrom = nystrom_row["NMI"]

    x_a = np.array([0, 1.2])
    bar_w = 0.22

    # Plot RI and NMI separately with twin axis on ax1
    b_p1 = ax1.bar(x_a[0] - bar_w, paper_ri, bar_w, color=COLOR_PAPER, edgecolor="#222222", label="Paper Table 6 Reported")
    b_r1 = ax1.bar(x_a[0], repo_ri_exact, bar_w, color=COLOR_EXACT, edgecolor="#222222", label="Reproduced (Exact Benchmark)")
    b_rn1 = ax1.bar(x_a[0] + bar_w, repo_ri_nystrom, bar_w, color=COLOR_NYSTROM, edgecolor="#222222", label="Reproduced (Nyström Benchmark)")

    ax1_twin = ax1.twinx()
    b_p2 = ax1_twin.bar(x_a[1] - bar_w, paper_nmi_mid, bar_w, color=COLOR_PAPER, edgecolor="#222222", alpha=0.85)
    b_r2 = ax1_twin.bar(x_a[1], repo_nmi_exact, bar_w, color=COLOR_EXACT, edgecolor="#222222", alpha=0.85)
    b_rn2 = ax1_twin.bar(x_a[1] + bar_w, repo_nmi_nystrom, bar_w, color=COLOR_NYSTROM, edgecolor="#222222", alpha=0.85)

    # Errorbar / range on paper NMI
    ax1_twin.errorbar(x_a[1] - bar_w, paper_nmi_mid, yerr=[[0.05], [0.05]], fmt="none", ecolor="#1b7837", elinewidth=2.5, capsize=5)

    # Annotate values
    ax1.annotate(f"{paper_ri:.2f}%", xy=(x_a[0] - bar_w, paper_ri), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax1.annotate(f"{repo_ri_exact:.2f}%", xy=(x_a[0], repo_ri_exact), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax1.annotate(f"{repo_ri_nystrom:.2f}%", xy=(x_a[0] + bar_w, repo_ri_nystrom), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax1_twin.annotate(f"0.55-0.65", xy=(x_a[1] - bar_w, 0.65), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax1_twin.annotate(f"{repo_nmi_exact:.4f}", xy=(x_a[1], repo_nmi_exact), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax1_twin.annotate(f"{repo_nmi_nystrom:.4f}", xy=(x_a[1] + bar_w, repo_nmi_nystrom), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x_a)
    ax1.set_xticklabels(categories_a, fontweight="bold")
    ax1.set_ylabel("Rand Index (%) [Left Axis]", fontweight="bold")
    ax1_twin.set_ylabel("Normalized Mutual Information [Right Axis]", fontweight="bold")
    ax1.set_ylim(70, 102)
    ax1_twin.set_ylim(0.0, 0.85)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1_twin.grid(False)
    ax1.set_title("(A) Successfully & Partially Reproduced Clustering Metrics\n[Rand Index matches Table 6 within 0.13%]", pad=12)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    # --- Panel B: Supervised Accuracy Discrepancy (Scientific Limitation) ---
    acc_labels = [
        "Paper Claimed\n(Undefined 'Chunks')",
        "Exact Benchmark\n(Leak-Free 20-Class)",
        "Nyström Benchmark\n(Leak-Free 20-Class)",
        "Nyström Full Split\n(18.8k Docs 20-Class)",
        "Random Guess\nBaseline (1/20)",
    ]
    acc_values = [
        95.0,  # Paper claim >95%
        exact_row["Accuracy"] * 100.0,
        nystrom_row["Accuracy"] * 100.0,
        full_row["Accuracy"] * 100.0,
        5.0,
    ]
    acc_colors = ["#b2df8a", COLOR_EXACT, COLOR_NYSTROM, "#e31a1c", COLOR_BASELINE]

    bars_acc = ax2.bar(acc_labels, acc_values, color=acc_colors, width=0.55, edgecolor="#222222", linewidth=1.2, zorder=3)

    for bar, val in zip(bars_acc, acc_values):
        txt = "> 95.0%" if val == 95.0 else f"{val:.2f}%"
        ax2.annotate(txt,
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add red cross / warning annotation on the paper accuracy claim
    ax2.annotate("UNVERIFIED CLAIM\n(Undefined chunks,\nnon-reproducible)",
                 xy=(0, 95.0), xytext=(0.8, 85.0),
                 arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=8.5, fontweight="bold", color="#d62728", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee8c8", edgecolor="#e34a33"))

    ax2.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax2.set_title("(B) Classification Accuracy Divergence\n[Paper Claim vs. Rigorous Leakage-Free Evaluation]", pad=12)
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    fig.suptitle("Figure 5: Scientific Audit — Paper Claims vs. Rigorous Reproduction\n(Pitchandi & Grace, ESWA 2025 vs. Independent Leak-Free Reproduction)", y=1.02)

    fig.text(0.5, -0.06,
             "Scientific Audit Summary:\n"
             "• Rand Index (92.21%) reproduces paper Table 6 (92.34%) closely under valid pairwise clustering evaluation.\n"
             "• Paper's >95% accuracy claim relies on undefined 'Chunk Size' subsets; under verified 20-class test partitions, accuracy is 45.50% (Exact) and 40.50% (Nyström).\n"
             "• The paper's 'fuzzy density score' formula is omitted from the publication and cannot be mathematically reproduced without heuristics.",
             ha="center", fontsize=8.5, style="normal", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#bbbbbb"))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "05_paper_vs_reproduced.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {out_path}")


# -----------------------------------------------------------------------------
# Main Execution Entry Point
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Generating Publication-Quality Research Artifact Figures...")
    print(f"Master CSV Source : {RESULTS_CSV}")
    print(f"Output Directory   : {OUTPUT_DIR}")
    print("=" * 70)

    df = load_metrics()
    print(f"Loaded metrics for {len(df)} experiments: {', '.join(df['Experiment'].tolist())}\n")

    generate_figure_1(df)
    generate_figure_2(df)
    generate_figure_3(df)
    generate_figure_4(df)
    generate_figure_5(df)

    print("\nAll 5 research presentation figures generated successfully!")


if __name__ == "__main__":
    main()
