from sklearn.datasets import load_files
from pathlib import Path


def load_20_newsgroups():
    """
    Load the 20 Newsgroups dataset from local files.

    Returns:
        documents: List of text documents
        labels: Numerical category labels
        target_names: Names of the categories
    """

    data_root = Path.home() / "Downloads"

    train_path = data_root / "20news-bydate-train"
    test_path = data_root / "20news-bydate-test"

    train = load_files(
        train_path,
        encoding="latin1",
        decode_error="ignore"
    )

    test = load_files(
        test_path,
        encoding="latin1",
        decode_error="ignore"
    )

    documents = train.data + test.data
    labels = list(train.target) + list(test.target)
    target_names = train.target_names

    print("20 Newsgroups dataset loaded successfully!")
    print(f"Number of documents : {len(documents)}")
    print(f"Number of categories: {len(target_names)}")

    return documents, labels, target_names
