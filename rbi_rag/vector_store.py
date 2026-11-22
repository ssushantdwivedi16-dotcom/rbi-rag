"""Store chunk embeddings in ChromaDB and search them."""

import sys
from pathlib import Path

import chromadb

from rbi_rag.chunker import Chunk, chunk_document
from rbi_rag.embedder import chunk_to_text, embed_query, embed_texts
from rbi_rag.pdf_reader import list_pdfs

DB_PATH = "chroma_db"
COLLECTION_NAME = "rbi_directions"
BATCH_SIZE = 500


def get_client():
    """Open (or create) the database folder on disk."""
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection():
    """Get the table that holds our chunks, using cosine similarity."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def build_index(pdf_folder: Path) -> int:
    """Chunk all PDFs, embed them, and save everything in ChromaDB."""
    chunks: list[Chunk] = []
    for pdf_path in list_pdfs(pdf_folder):
        chunks.extend(chunk_document(pdf_path))
    print("Chunks to index:", len(chunks))

    try:
        get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = get_collection()

    vectors = embed_texts([chunk_to_text(c) for c in chunks])

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors[start : start + BATCH_SIZE].tolist(),
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                    "page": c.page,
                    "chapter": c.chapter,
                    "heading": c.heading,
                }
                for c in batch
            ],
        )
    return collection.count()


def search(question: str, k: int = 5, doc_id: str | None = None) -> list[dict]:
    """Return the k closest chunks; optionally only from one document."""
    results = get_collection().query(
        query_embeddings=[embed_query(question).tolist()],
        n_results=k,
        where={"doc_id": doc_id} if doc_id else None,
    )
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "chunk_id": results["ids"][0][i],
                "score": round(1 - results["distances"][0][i], 3),
                "text": results["documents"][0][i],
                **results["metadatas"][0][i],
            }
        )
    return hits


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print("Stored in ChromaDB:", build_index(Path("data/raw_pdfs")))

    questions = [
        "When does a loan account become a non-performing asset?",
        "Can a bank issue a credit card without the customer's consent?",
        "What documents are needed to open an account for an individual?",
    ]
    for q in questions:
        print(f"\n=== {q}")
        for hit in search(q, k=3):
            print(f"{hit['score']}  {hit['doc_title']} | p{hit['page']} | {hit['heading']}")
            print("      ", hit["text"][:150])