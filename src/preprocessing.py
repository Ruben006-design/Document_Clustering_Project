import re


def clean_text(text):
    """
    Clean a document before feature extraction.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Keep alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_documents(documents):
    """
    Clean all documents.

    Args:
        documents: List of raw text documents

    Returns:
        cleaned_documents: List of cleaned documents
    """

    cleaned_documents = [clean_text(doc) for doc in documents]

    print("Text preprocessing completed successfully!")
    print(f"Number of documents processed: {len(cleaned_documents)}")

    return cleaned_documents