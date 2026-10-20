from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.95,
    stop_words="english"
):
    """
    Construct a TF-IDF vectorizer with standard document preprocessing parameters.
    """
    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words
    )


def create_tfidf_matrix(documents, max_features=1000, vectorizer=None):
    """
    Convert cleaned documents into a TF-IDF feature matrix.

    Args:
        documents: List of cleaned document strings
        max_features: Maximum number of vocabulary features
        vectorizer: Optional pre-fitted or custom TfidfVectorizer instance

    Returns:
        tfidf_matrix: Scipy sparse matrix of TF-IDF features
        feature_names: Array of feature token strings
        vectorizer: The fitted TfidfVectorizer
    """
    if vectorizer is None:
        vectorizer = get_tfidf_vectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(documents)
    else:
        tfidf_matrix = vectorizer.transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    print("TF-IDF feature extraction completed successfully!")
    print(f"Number of documents : {tfidf_matrix.shape[0]}")
    print(f"Number of features  : {tfidf_matrix.shape[1]}")

    return tfidf_matrix, feature_names, vectorizer