
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data_loader import load_20_newsgroups
from src.preprocessing import preprocess_documents
from src.feature_extraction import create_tfidf_matrix
from src.fd_ffo import FDFFO
from src.tl_hho import TLHHO
from src.svr_pnn import SVRPNN


def main():

    # ==========================================================
    # 1. LOAD 20 NEWSGROUPS DATASET
    # ==========================================================

    documents, labels, target_names = load_20_newsgroups()

    print("\nDataset:")
    print("Number of documents :", len(documents))
    print("Number of categories:", len(target_names))


    # ==========================================================
    # 2. TEXT PREPROCESSING
    # ==========================================================

    cleaned_documents = preprocess_documents(documents)

    print("\nText preprocessing completed successfully!")
    print(
        "Number of documents processed:",
        len(cleaned_documents)
    )


    # ==========================================================
    # 3. TF-IDF FEATURE EXTRACTION
    # ==========================================================

    tfidf_matrix, feature_names, vectorizer = (
        create_tfidf_matrix(cleaned_documents)
    )

    print("\nTF-IDF feature extraction completed successfully!")
    print("Number of documents :", tfidf_matrix.shape[0])
    print("Number of features  :", tfidf_matrix.shape[1])


    # ==========================================================
    # 4. SMALL DATASET FOR PIPELINE INTEGRATION TEST
    # ==========================================================
    #
    # We are currently using a small subset because the current
    # SVR-PNN implementation creates a kernel matrix.
    #
    # Once the complete pipeline is verified, we can optimize
    # the implementation and increase the dataset size.
    #

    number_of_documents = 200
    number_of_features = 50

    X = tfidf_matrix[
        :number_of_documents,
        :number_of_features
    ].toarray()

    y = labels[:number_of_documents]

    selected_feature_names = feature_names[
        :number_of_features
    ]


    print("\nPipeline test matrix:")
    print("Documents:", X.shape[0])
    print("Features :", X.shape[1])


    # ==========================================================
    # 5. FD-FFO FEATURE REDUNDANCY REDUCTION
    # ==========================================================

    print("\nRunning FD-FFO...")

    fd_ffo = FDFFO(
        cutoff_distance=0.5,
        density_threshold=0.4
    )

    X_fdffo = fd_ffo.fit_transform(X)

    selected_fdffo_names = selected_feature_names[
        fd_ffo.selected_features_
    ]

    print("\nFD-FFO completed!")

    print("Original feature count :",
          X.shape[1])

    print("Remaining feature count:",
          X_fdffo.shape[1])

    print("Removed feature count  :",
          X.shape[1] - X_fdffo.shape[1])


    # ==========================================================
    # 6. TL-HHO FEATURE SELECTION
    # ==========================================================

    print("\nRunning TL-HHO...")

    optimizer = TLHHO(
        population_size=10,
        max_iterations=10,
        alpha=0.9,
        random_state=42
    )

    # IMPORTANT:
    # TL-HHO receives the FD-FFO output,
    # not the original TF-IDF matrix.

    selected_mask = optimizer.fit(
        X_fdffo,
        y
    )

    selected_features = X_fdffo[
        :,
        selected_mask
    ]

    selected_names = selected_fdffo_names[
        selected_mask
    ]


    print("\nTL-HHO completed!")

    print("Features after FD-FFO :",
          X_fdffo.shape[1])

    print("Features selected by TL-HHO:",
          len(selected_names))

    print("\nSelected features:")

    for feature in selected_names:
        print(" -", feature)


    # ==========================================================
    # 7. TRAIN / TEST SPLIT
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        selected_features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test split:")
    print("Training samples:", X_train.shape[0])
    print("Testing samples :", X_test.shape[0])


    # ==========================================================
    # 8. SVR-PNN
    # ==========================================================

    print("\nRunning SVR-PNN...")

    model = SVRPNN(
        gamma=1.0,
        sigma=1.0
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )


    # ==========================================================
    # 9. EVALUATION
    # ==========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    # ==========================================================
    # 10. RESULTS
    # ==========================================================

    print("\n======================================")
    print("        PIPELINE RESULTS")
    print("======================================")

    print(f"Original TF-IDF features : {X.shape[1]}")
    print(f"After FD-FFO             : {X_fdffo.shape[1]}")
    print(f"After TL-HHO             : {len(selected_names)}")

    print("--------------------------------------")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F-Measure : {f1:.4f}")

    print("--------------------------------------")

    print("Probability matrix shape:",
          probabilities.shape)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
