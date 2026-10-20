import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    rand_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)

from src.data_loader import load_20_newsgroups, get_stratified_sample, load_web_snippets
from src.preprocessing import preprocess_documents
from src.feature_extraction import get_tfidf_vectorizer
from src.fd_ffo import FDFFO
from src.tl_hho import TLHHO
from src.svr_pnn import SVRPNN


def run_experiment(
    mode="smoke_test",
    dataset_name="20newsgroups",
    fdffo_mode="passthrough",
    mapping_mode="one_hot",
    solver_mode="exact",
    n_components=200,
    max_features=100,
    pop_size=10,
    max_iter=5,
    gamma=1.0,
    sigma=1.0,
    seed=42,
    results_dir="results"
):
    """
    Execute the document clustering / classification pipeline with
    rigorous train/test separation to eliminate data leakage.

    Pipeline Steps:
    1. Dataset loading & stratified sampling (if in sample mode)
    2. Establish Train / Test split BEFORE feature extraction & selection
    3. Text Preprocessing
    4. TF-IDF feature extraction (fit on train ONLY, transform test)
    5. FD-FFO feature redundancy reduction (fit on train ONLY, transform test)
    6. TL-HHO metaheuristic feature selection (fit on train ONLY, transform test)
    7. SVR-PNN training (fit on train ONLY)
    8. Comprehensive evaluation on unseen test data
    9. Save metrics and configuration to disk
    """
    start_time = time.time()
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 65)
    log("  DOCUMENT CLUSTERING RESEARCH PIPELINE - EXPERIMENTAL RUN")
    log("  Paper: ESWA 294 (2025) 128762 (Pitchandi & Grace)")
    log("=" * 65)
    log(f"Execution Mode    : {mode}")
    log(f"Dataset           : {dataset_name}")
    log(f"Random Seed       : {seed}")
    log(f"FD-FFO Mode       : {fdffo_mode}")
    log(f"SVR-PNN Mapping   : {mapping_mode}")
    log(f"SVR Solver Mode   : {solver_mode}")
    if solver_mode == "nystrom":
        log(f"Nystrom Landmarks : {n_components}")
    log(f"Max TF-IDF Terms  : {max_features}")
    log(f"TL-HHO Pop / Iter : {pop_size} / {max_iter}")
    log("-" * 65)

    # ==========================================================
    # 1. LOAD DATASET
    # ==========================================================
    log("\n[Stage 1/7] Loading Dataset...")

    if dataset_name == "20newsgroups":
        if mode == "full":
            log("Loading official 20 Newsgroups train and test splits...")
            train_docs, train_labels, target_names = load_20_newsgroups(subset="train")
            test_docs, test_labels, _ = load_20_newsgroups(subset="test")
            raw_train_docs, y_train = train_docs, train_labels
            raw_test_docs, y_test = test_docs, test_labels
            is_pre_split = True
        elif mode == "benchmark":
            log("Loading 20 Newsgroups and sampling 50 docs/class across all 20 classes (1,000 docs)...")
            docs, labels, target_names = load_20_newsgroups(subset="all")
            raw_docs, all_labels = get_stratified_sample(
                docs, labels, n_samples_per_class=50, random_state=seed
            )
            is_pre_split = False
        else:  # smoke_test
            log("Loading 20 Newsgroups and sampling 10 docs/class across all 20 classes (200 docs)...")
            docs, labels, target_names = load_20_newsgroups(subset="all")
            raw_docs, all_labels = get_stratified_sample(
                docs, labels, n_samples_per_class=10, random_state=seed
            )
            is_pre_split = False

    elif dataset_name == "web_snippets":
        if mode == "full":
            raw_train_docs, y_train, target_names = load_web_snippets(subset="train")
            raw_test_docs, y_test, _ = load_web_snippets(subset="test")
            is_pre_split = True
        else:
            docs, labels, target_names = load_web_snippets(subset="all")
            n_per = 15 if mode == "smoke_test" else 60
            raw_docs, all_labels = get_stratified_sample(
                docs, labels, n_samples_per_class=n_per, random_state=seed
            )
            is_pre_split = False
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ==========================================================
    # 2. TRAIN / TEST SPLIT (PREVENT DATA LEAKAGE)
    # ==========================================================
    log("\n[Stage 2/7] Establishing Train/Test Partition (Zero Data Leakage)...")

    if not is_pre_split:
        raw_train_docs, raw_test_docs, y_train, y_test = train_test_split(
            raw_docs,
            all_labels,
            test_size=0.2,
            random_state=seed,
            stratify=all_labels
        )

    log(f"Training Documents : {len(raw_train_docs)}")
    log(f"Testing Documents  : {len(raw_test_docs)}")
    log(f"Unique Categories  : {len(target_names)}")

    # Check computational feasibility for exact SVR-PNN
    resource_info = SVRPNN.estimate_resource_requirements(len(raw_train_docs))
    log(f"Estimated Kernel RAM (Train) : {resource_info['kernel_memory_mb']} MB")
    log(f"Estimated FLOPs (Solve)      : {resource_info['approx_flops']}")
    if not resource_info["is_feasible_exact"]:
        log("WARNING: Training set size exceeds conservative threshold for exact KKT solve!")
        if mode == "full":
            if solver_mode == "exact":
                log("ABORTING full exact solve to avoid system memory exhaustion. Use --solver_mode nystrom or benchmark mode.")
                return None
            else:
                log(f"Proceeding with scalable Nyström low-rank approximation (landmarks: {n_components}).")

    # ==========================================================
    # 3. TEXT PREPROCESSING
    # ==========================================================
    log("\n[Stage 3/7] Preprocessing Documents...")
    train_cleaned = preprocess_documents(raw_train_docs)
    test_cleaned = preprocess_documents(raw_test_docs)

    # ==========================================================
    # 4. TF-IDF VECTORIZATION (FIT ON TRAIN ONLY)
    # ==========================================================
    log("\n[Stage 4/7] TF-IDF Feature Extraction (Fit on Train ONLY)...")
    vectorizer = get_tfidf_vectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(train_cleaned).toarray()
    X_test_tfidf = vectorizer.transform(test_cleaned).toarray()
    feature_names = vectorizer.get_feature_names_out()

    log(f"TF-IDF Vocabulary Size : {X_train_tfidf.shape[1]}")

    # ==========================================================
    # 5. FD-FFO FEATURE REDUNDANCY REDUCTION
    # ==========================================================
    log("\n[Stage 5/7] Running FD-FFO on Training Set...")
    fdffo = FDFFO(
        cutoff_distance=0.5,
        density_threshold=0.4,
        scoring_mode=fdffo_mode,
        max_iterations=10,
        random_state=seed
    )
    X_train_fdffo = fdffo.fit_transform(X_train_tfidf)
    X_test_fdffo = fdffo.transform(X_test_tfidf)

    log(f"Features Before FD-FFO : {fdffo.n_features_in_}")
    log(f"Features After FD-FFO  : {fdffo.n_features_out_}")
    log(f"FD-FFO Removal Rate    : {fdffo.removal_rate_ * 100:.1f}%")

    # ==========================================================
    # 6. TL-HHO FEATURE SELECTION
    # ==========================================================
    log("\n[Stage 6/7] Running TL-HHO on Training Set...")
    tlhho = TLHHO(
        population_size=pop_size,
        max_iterations=max_iter,
        alpha=0.9,
        random_state=seed
    )
    X_train_tlhho = tlhho.fit_transform(X_train_fdffo, y_train)
    X_test_tlhho = tlhho.transform(X_test_fdffo)

    n_tlhho_selected = int(np.sum(tlhho.selected_features_))
    log(f"Features Selected by TL-HHO : {n_tlhho_selected} / {X_train_fdffo.shape[1]}")
    log(f"TL-HHO Best Fitness         : {tlhho.best_fitness:.6f}")

    # ==========================================================
    # 7. SVR-PNN TRAINING & EVALUATION
    # ==========================================================
    log("\n[Stage 7/7] Training SVR-PNN and Evaluating on Held-Out Test Set...")
    model = SVRPNN(
        gamma=gamma,
        sigma=sigma,
        mapping_mode=mapping_mode,
        solver_mode=solver_mode,
        n_components=n_components,
        random_state=seed
    )
    model.fit(X_train_tlhho, y_train)

    predictions = model.predict(X_test_tlhho)
    probabilities = model.predict_proba(X_test_tlhho)

    # Supervised classification metrics
    accuracy = float(accuracy_score(y_test, predictions))
    precision_w = float(precision_score(y_test, predictions, average="weighted", zero_division=0))
    recall_w = float(recall_score(y_test, predictions, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_test, predictions, average="weighted", zero_division=0))

    precision_m = float(precision_score(y_test, predictions, average="macro", zero_division=0))
    recall_m = float(recall_score(y_test, predictions, average="macro", zero_division=0))
    f1_m = float(f1_score(y_test, predictions, average="macro", zero_division=0))

    # Clustering metrics
    ari = float(adjusted_rand_score(y_test, predictions))
    nmi = float(normalized_mutual_info_score(y_test, predictions))
    ri = float(rand_score(y_test, predictions))

    cm = confusion_matrix(y_test, predictions).tolist()

    elapsed_time = round(time.time() - start_time, 2)

    # ==========================================================
    # SUMMARY & REPORTING
    # ==========================================================
    log("\n" + "=" * 65)
    log("                       FINAL RESULTS")
    log("=" * 65)
    log(f"Execution Time       : {elapsed_time}s")
    log(f"Total Samples (Tr/Te): {len(y_train)} / {len(y_test)}")
    log(f"Features: TF-IDF -> FD-FFO -> TL-HHO : {X_train_tfidf.shape[1]} -> {X_train_fdffo.shape[1]} -> {n_tlhho_selected}")
    log("-" * 65)
    log("Classification Metrics (Supervised):")
    log(f"  Accuracy           : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    log(f"  Precision (weighted): {precision_w:.4f} ({precision_w * 100:.2f}%)")
    log(f"  Recall    (weighted): {recall_w:.4f} ({recall_w * 100:.2f}%)")
    log(f"  F1-Score  (weighted): {f1_w:.4f} ({f1_w * 100:.2f}%)")
    log(f"  Precision (macro)   : {precision_m:.4f}")
    log(f"  Recall    (macro)   : {recall_m:.4f}")
    log(f"  F1-Score  (macro)   : {f1_m:.4f}")
    log("-" * 65)
    log("Clustering Metrics (Partition Agreement):")
    log(f"  Rand Index (RI)    : {ri:.4f} ({ri * 100:.2f}%)")
    log(f"  Adjusted Rand Index: {ari:.4f}")
    log(f"  NMI (Norm Mut Info): {nmi:.4f}")
    log("=" * 65)

    # Save artifacts
    config_dict = {
        "mode": mode,
        "dataset_name": dataset_name,
        "random_seed": seed,
        "fdffo_mode": fdffo_mode,
        "mapping_mode": mapping_mode,
        "solver_mode": solver_mode,
        "n_components": n_components,
        "max_features": max_features,
        "tlhho_population": pop_size,
        "tlhho_iterations": max_iter,
        "gamma": gamma,
        "sigma": sigma,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "num_classes": len(target_names)
    }

    feature_summary = {
        "initial_tfidf_features": int(X_train_tfidf.shape[1]),
        "after_fd_ffo_features": int(X_train_fdffo.shape[1]),
        "fd_ffo_removal_rate": float(fdffo.removal_rate_),
        "after_tl_hho_features": int(n_tlhho_selected),
        "tl_hho_selection_ratio": float(n_tlhho_selected / max(1, X_train_fdffo.shape[1])),
        "tl_hho_best_fitness": float(tlhho.best_fitness)
    }

    metrics_dict = {
        "execution_time_seconds": elapsed_time,
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
        "rand_index": ri,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "confusion_matrix": cm
    }

    with open(results_path / "experiment_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(results_path / "feature_selection_summary.json", "w") as f:
        json.dump(feature_summary, f, indent=2)

    with open(results_path / "test_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(results_path / "execution_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    log(f"\nAll experimental artifacts successfully written to: {results_path.resolve()}")
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Document Clustering Pipeline Runner")
    parser.add_argument("--mode", type=str, default="smoke_test", choices=["smoke_test", "benchmark", "full"],
                        help="Experiment mode: smoke_test, benchmark, or full")
    parser.add_argument("--dataset", type=str, default="20newsgroups", choices=["20newsgroups", "web_snippets"],
                        help="Dataset to evaluate on")
    parser.add_argument("--fdffo_mode", type=str, default="passthrough", choices=["passthrough", "density_score", "fixed_ratio"],
                        help="FD-FFO scoring mode")
    parser.add_argument("--mapping_mode", type=str, default="one_hot", choices=["one_hot", "ordinal"],
                        help="SVR multiclass mapping mode")
    parser.add_argument("--solver_mode", type=str, default="exact", choices=["exact", "nystrom"],
                        help="SVR solver mode: exact (standard paper formulation) or nystrom (scalable approximation)")
    parser.add_argument("--n_components", type=int, default=200,
                        help="Number of Nystrom landmark components")
    parser.add_argument("--max_features", type=int, default=100,
                        help="Max TF-IDF terms")
    parser.add_argument("--pop_size", type=int, default=10,
                        help="TL-HHO population size")
    parser.add_argument("--max_iter", type=int, default=5,
                        help="TL-HHO iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Output directory for results and logs")

    args = parser.parse_args()

    # Automatically set appropriate defaults for benchmark mode if not overridden
    if args.mode == "benchmark" and args.max_features == 100:
        args.max_features = 300

    run_experiment(
        mode=args.mode,
        dataset_name=args.dataset,
        fdffo_mode=args.fdffo_mode,
        mapping_mode=args.mapping_mode,
        solver_mode=args.solver_mode,
        n_components=args.n_components,
        max_features=args.max_features,
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        seed=args.seed,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
