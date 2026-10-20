import os
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_20newsgroups, load_files


def load_20_newsgroups(
    subset="all",
    data_home=None,
    download_if_missing=False,
    categories=None,
    remove=()
):
    """
    Load the 20 Newsgroups dataset robustly.

    Priority:
    1. If local sklearn cache exists (e.g. ~/scikit_learn_data), loads offline.
    2. If raw directory structure exists (~/Downloads/20news-bydate-*), loads via load_files.
    3. If online access is permitted and data is missing, downloads via fetch_20newsgroups.

    Args:
        subset: 'all', 'train', or 'test'
        data_home: Optional path to scikit_learn_data cache directory
        download_if_missing: Whether to download if not present locally
        categories: Optional list of category names to filter
        remove: Tuple of elements to strip ('headers', 'footers', 'quotes')

    Returns:
        documents: List of text documents (str)
        labels: 1D numpy array of integer class labels
        target_names: List of category name strings
    """

    # Check local raw directory structure first for backward compatibility
    data_root = Path.home() / "Downloads"
    train_path = data_root / "20news-bydate-train"
    test_path = data_root / "20news-bydate-test"

    if train_path.exists() and test_path.exists():
        print(f"Loading 20 Newsgroups from local folders: {train_path} / {test_path}")
        train = load_files(
            str(train_path),
            categories=categories,
            encoding="latin1",
            decode_error="ignore"
        )
        test = load_files(
            str(test_path),
            categories=categories,
            encoding="latin1",
            decode_error="ignore"
        )

        if subset == "train":
            documents = train.data
            labels = np.asarray(train.target, dtype=int)
            target_names = list(train.target_names)
        elif subset == "test":
            documents = test.data
            labels = np.asarray(test.target, dtype=int)
            target_names = list(test.target_names)
        else:
            documents = list(train.data) + list(test.data)
            labels = np.concatenate([train.target, test.target]).astype(int)
            target_names = list(train.target_names)

        print(f"Loaded {len(documents)} documents across {len(target_names)} categories.")
        return documents, labels, target_names

    # Use fetch_20newsgroups (utilizing existing local scikit_learn_data cache)
    try:
        bunch = fetch_20newsgroups(
            data_home=data_home,
            subset=subset,
            categories=categories,
            shuffle=False,
            random_state=42,
            remove=remove,
            download_if_missing=download_if_missing
        )
        documents = list(bunch.data)
        labels = np.asarray(bunch.target, dtype=int)
        target_names = list(bunch.target_names)

        print("20 Newsgroups dataset loaded successfully via scikit-learn cache!")
        print(f"Subset              : {subset}")
        print(f"Number of documents : {len(documents)}")
        print(f"Number of categories: {len(target_names)}")

        return documents, labels, target_names

    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to load 20 Newsgroups dataset. Neither local directory '{train_path}' "
            f"nor scikit-learn offline cache could be loaded ({exc}). "
            f"To enable download, set download_if_missing=True."
        ) from exc


def get_stratified_sample(
    documents,
    labels,
    n_samples_per_class=None,
    total_samples=None,
    random_state=42
):
    """
    Extract a balanced, stratified subset across all present classes.

    Ensures that every class is proportionally represented without
    alphabetical ordering artifacts.

    Args:
        documents: List of document strings
        labels: Array-like of integer class labels
        n_samples_per_class: Exact number of samples per class (if specified)
        total_samples: Total number of samples desired across all classes
        random_state: Integer seed for reproducible sampling

    Returns:
        sampled_docs: List of sampled document strings
        sampled_labels: 1D numpy array of sampled integer labels
    """
    labels = np.asarray(labels)
    unique_classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_classes)
    rng = np.random.default_rng(random_state)

    if n_samples_per_class is None:
        if total_samples is None:
            raise ValueError("Either n_samples_per_class or total_samples must be provided.")
        n_samples_per_class = max(1, total_samples // num_classes)

    selected_indices = []
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        actual_samples = min(n_samples_per_class, len(cls_indices))
        chosen = rng.choice(cls_indices, size=actual_samples, replace=False)
        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    sampled_docs = [documents[i] for i in selected_indices]
    sampled_labels = labels[selected_indices]

    return sampled_docs, sampled_labels


def load_web_snippets(zip_path=None, subset="all"):
    """
    Load the Web-Snippets dataset from local archive.
    Paper reference: 8 domain categories, 12,340 snippets.

    Args:
        zip_path: Optional path to ENwebsnippets.zip
        subset: 'all', 'train', or 'test'

    Returns:
        documents: List of text snippets
        labels: 1D numpy array of integer labels
        target_names: List of category strings
    """
    import zipfile

    if zip_path is None:
        potential_paths = [
            Path("data") / "ENwebsnippets.zip",
            Path.home() / "Downloads" / "ENwebsnippets.zip",
        ]
        for p in potential_paths:
            if p.exists():
                zip_path = p
                break

    if zip_path is None or not Path(zip_path).exists():
        raise FileNotFoundError(f"Web-Snippets archive not found at {zip_path}")

    target_domains = [
        "education-science",
        "culture-arts-entertainment",
        "business",
        "computers",
        "politics-society",
        "sports",
        "health",
        "engineering"
    ]
    domain_to_idx = {dom: i for i, dom in enumerate(target_domains)}

    documents = []
    labels = []

    files_to_read = []
    if subset in ("all", "train"):
        files_to_read.append("EN-web-snippets/train.txt")
    if subset in ("all", "test"):
        files_to_read.append("EN-web-snippets/test.txt")

    with zipfile.ZipFile(zip_path) as z:
        for fname in files_to_read:
            lines = z.read(fname).decode("latin1", errors="ignore").splitlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                label_candidate = parts[-1]
                if label_candidate in domain_to_idx:
                    text = " ".join(parts[:-1])
                    documents.append(text)
                    labels.append(domain_to_idx[label_candidate])

    return documents, np.asarray(labels, dtype=int), target_domains
