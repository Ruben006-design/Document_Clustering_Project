"""
Controlled Experiment Runner & Figure Generator for
Adaptive Density-Aware Feature Selection (ADAFS)

Compares four Stage 1 feature selection configurations on the 20 Newsgroups benchmark:
  1. passthrough (Faithful baseline, keeps all features)
  2. density_score (Heuristic decision graph norm_rho * norm_delta >= 0.4)
  3. fixed_ratio (Fixed 70% removal heuristic)
  4. adaptive_density (Proposed ADAFS research extension)

Outputs:
  - results/adaptive_density/adaptive_density_comparison.csv
  - results/adaptive_density/adaptive_density_comparison.png
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import run_experiment

RESULTS_DIR = PROJECT_ROOT / "results" / "adaptive_density"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_all_modes():
    modes = [
        ("passthrough", "Faithful Baseline (Passthrough)"),
        ("density_score", "Density-Delta Heuristic (rho * delta >= 0.4)"),
        ("fixed_ratio", "Fixed Ratio Heuristic (70% Drop)"),
        ("adaptive_density", "Proposed ADAFS (Adaptive Density-Aware)"),
    ]

    records = []

    for mode_name, label in modes:
        mode_dir = RESULTS_DIR / mode_name
        print("\n" + "=" * 70)
        print(f"RUNNING CONTROLLED EXPERIMENT: {label} [{mode_name}]")
        print("=" * 70)

        t0 = time.time()
        run_experiment(
            mode="benchmark",
            dataset_name="20newsgroups",
            fdffo_mode=mode_name,
            mapping_mode="one_hot",
            solver_mode="exact",
            max_features=300,
            pop_size=10,
            max_iter=5,
            gamma=1.0,
            sigma=1.0,
            seed=42,
            results_dir=str(mode_dir),
            adafs_lambda=0.5,
            adafs_threshold_method="percentile",
            adafs_threshold_value=50.0
        )
        elapsed = round(time.time() - t0, 2)

        # Load metrics from written artifacts
        with open(mode_dir / "test_metrics.json", "r") as f:
            test_metrics = json.load(f)
        with open(mode_dir / "feature_selection_summary.json", "r") as f:
            feat_summary = json.load(f)

        records.append({
            "Mode": mode_name,
            "Display_Name": label,
            "Initial_TFIDF": feat_summary["initial_tfidf_features"],
            "After_Stage1": feat_summary["after_fd_ffo_features"],
            "Stage1_Removal_Rate": feat_summary["fd_ffo_removal_rate"],
            "After_TLHHO": feat_summary["after_tl_hho_features"],
            "TLHHO_Selection_Ratio": feat_summary["tl_hho_selection_ratio"],
            "Accuracy": test_metrics["accuracy"],
            "Precision_Weighted": test_metrics["precision_weighted"],
            "Recall_Weighted": test_metrics["recall_weighted"],
            "F1_Weighted": test_metrics["f1_weighted"],
            "Precision_Macro": test_metrics["precision_macro"],
            "Recall_Macro": test_metrics["recall_macro"],
            "F1_Macro": test_metrics["f1_macro"],
            "Rand_Index": test_metrics["rand_index"],
            "ARI": test_metrics["adjusted_rand_index"],
            "NMI": test_metrics["normalized_mutual_info"],
            "Runtime_Sec": test_metrics["execution_time_seconds"]
        })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "adaptive_density_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Comparison table: {csv_path}")
    return df


def generate_comparison_figure(df):
    """
    Generate publication-quality comparison figure for ADAFS vs baseline modes.
    Visualizes:
      (A) Classification Accuracy (%)
      (B) Clustering Metrics: Rand Index (%) & NMI
      (C) Two-Stage Dimensionality Reduction Progression
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--"
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5))

    mode_labels = [
        "Faithful Baseline\n(passthrough)",
        "Density Heuristic\n(density_score)",
        "Fixed Ratio (70%)\n(fixed_ratio)",
        "Proposed ADAFS\n(adaptive_density)"
    ]

    # Colors: baseline muted, proposed highlighted
    colors = ["#1f77b4", "#aec7e8", "#c7c7c7", "#2ca02c"]
    edge_colors = ["#111111", "#111111", "#111111", "#00441b"]
    line_widths = [1.0, 1.0, 1.0, 1.8]

    # --- Panel A: Classification Accuracy ---
    accs = df["Accuracy"] * 100.0
    bars1 = ax1.bar(mode_labels, accs, color=colors, edgecolor=edge_colors, linewidth=line_widths, width=0.55, zorder=3)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.axhline(5.0, color="#d62728", linestyle=":", linewidth=1.4, label="20-Class Guess Baseline (5.0%)")
    ax1.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax1.set_title("(A) Supervised Classification Accuracy", pad=12)
    ax1.set_ylim(0, 55)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    # --- Panel B: Clustering Quality (Rand Index & NMI) ---
    x = np.arange(len(mode_labels))
    w = 0.35
    ri_vals = df["Rand_Index"] * 100.0
    nmi_vals = df["NMI"] * 100.0  # Scale NMI to 0-100 for side-by-side grouped visualization

    b_ri = ax2.bar(x - w/2, ri_vals, w, label="Rand Index (%)", color="#386cb0", edgecolor="#222222", zorder=3)
    b_nmi = ax2.bar(x + w/2, nmi_vals, w, label="NMI (×100)", color="#7fc97f", edgecolor="#222222", zorder=3)

    for bar in b_ri:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in b_nmi:
        h = bar.get_height()
        ax2.annotate(f"{h/100:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_labels, fontsize=8.5)
    ax2.set_ylabel("Metric Value (Normalized Scale 0 - 100)", fontweight="bold")
    ax2.set_title("(B) Clustering Quality (RI % vs. NMI)", pad=12)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower left", framealpha=0.95, fontsize=8.5)

    # --- Panel C: Feature Progression ---
    f_init = df["Initial_TFIDF"].tolist()
    f_stg1 = df["After_Stage1"].tolist()
    f_stg2 = df["After_TLHHO"].tolist()

    w_c = 0.25
    b_init = ax3.bar(x - w_c, f_init, w_c, label="Initial TF-IDF", color="#9ecae1", edgecolor="#222222", zorder=3)
    b_stg1 = ax3.bar(x, f_stg1, w_c, label="Post Stage 1 (FD-FFO/ADAFS)", color="#fdae6b", edgecolor="#222222", zorder=3)
    b_stg2 = ax3.bar(x + w_c, f_stg2, w_c, label="Post Stage 2 (TL-HHO)", color="#e6550d", edgecolor="#222222", zorder=3)

    for bar in b_stg1:
        h = bar.get_height()
        ax3.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in b_stg2:
        h = bar.get_height()
        ax3.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(mode_labels, fontsize=8.5)
    ax3.set_ylabel("Feature Count", fontweight="bold")
    ax3.set_title("(C) Two-Stage Feature Count Progression", pad=12)
    ax3.set_ylim(0, 360)
    ax3.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    fig.suptitle("Controlled Empirical Comparison: Proposed ADAFS vs. Baseline Screening Modes\n(Identical 20 Newsgroups Benchmark: 800 Train / 200 Test, 20 Classes, Seed 42)", y=1.03)

    fig.text(0.5, -0.07,
             "Scientific Research Extension Notice:\n"
             "• 'Proposed ADAFS' is an independent research extension combining supervised ANOVA relevance with density-aware redundancy.\n"
             "• Baseline modes (passthrough, density_score, fixed_ratio) represent the original audited paper reproduction framework.\n"
             "• All modes evaluated strictly under zero test-set leakage.",
             ha="center", fontsize=8.5, style="normal",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#bbbbbb"))

    plt.tight_layout()
    out_png = RESULTS_DIR / "adaptive_density_comparison.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"[Generated] Novelty figure: {out_png}")


if __name__ == "__main__":
    df = run_all_modes()
    generate_comparison_figure(df)
    print("\nControlled experiment and figure generation completed successfully!")
