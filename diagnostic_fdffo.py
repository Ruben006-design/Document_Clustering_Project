import numpy as np
from pathlib import Path
from sklearn.datasets import load_files

from src.preprocessing import preprocess_documents
from src.feature_extraction import create_tfidf_matrix
from src.fd_ffo import FDFFO

train_path = Path.home() / "Downloads" / "20news-bydate-train"

data = load_files(
    train_path,
    encoding="latin1",
    decode_error="ignore"
)

documents = data.data[:200]

print("Loaded diagnostic documents:", len(documents))

clean = preprocess_documents(documents)

tfidf, features, _ = create_tfidf_matrix(clean)

X = tfidf[:, :50].toarray()

print("Diagnostic matrix:", X.shape)
print()
print("FD-FFO cutoff-distance diagnostic")
print("=" * 70)
print("Cutoff | Min Density | Max Density | Mean Density | Zero Density")
print("-" * 70)

for cutoff in [0.05, 0.10, 0.15, 0.20, 0.25,
               0.30, 0.35, 0.40, 0.45, 0.50]:

    model = FDFFO(
        cutoff_distance=cutoff,
        max_iterations=10,
        random_state=42
    )

    model.fit(X)

    density = model.density_

    print(
        f"{cutoff:6.2f} | "
        f"{density.min():11.0f} | "
        f"{density.max():11.0f} | "
        f"{density.mean():12.2f} | "
        f"{np.sum(density == 0):11d}"
    )
