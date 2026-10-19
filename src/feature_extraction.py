from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(documents):
    """
    Convert cleaned documents into a TF-IDF feature matrix.
    """

    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    print("TF-IDF feature extraction completed successfully!")
    print(f"Number of documents : {tfidf_matrix.shape[0]}")
    print(f"Number of features  : {tfidf_matrix.shape[1]}")

    return tfidf_matrix, feature_names, vectorizer