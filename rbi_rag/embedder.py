"""Turn text into embedding vectors (lists of numbers that capture meaning)."""

import numpy as np
from sentence_transformers import SentenceTransformer

from rbi_rag.chunker import Chunk

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model() -> SentenceTransformer:
    """Load the model once, then reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def chunk_to_text(chunk: Chunk) -> str:
    """The text we embed: labels plus paragraph, so the heading's meaning helps search."""
    return f"{chunk.doc_title} | {chunk.chapter} | {chunk.heading}\n{chunk.text}"


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed many texts at once (used for chunks)."""
    return get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )


def embed_query(question: str) -> np.ndarray:
    """Embed one user question (with the special search prefix)."""
    return get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)


if __name__ == "__main__":
    sentences = [
        "Banks must maintain a cash reserve with the RBI.",
        "The CRR has to be kept with the Reserve Bank.",
        "Credit cards must not be issued without customer consent.",
    ]
    vectors = embed_texts(sentences)

    print("Shape:", vectors.shape)
    print("First 5 numbers of sentence 1:", vectors[0][:5].round(3))
    print("\nSimilarity 1 vs 2 (same meaning):", round(float(vectors[0] @ vectors[1]), 3))
    print("Similarity 1 vs 3 (different topic):", round(float(vectors[0] @ vectors[2]), 3))