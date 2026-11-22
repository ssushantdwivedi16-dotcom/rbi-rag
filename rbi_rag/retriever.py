"""Two-stage retrieval: vector search finds candidates, a cross-encoder reranks them."""

import re

from sentence_transformers import CrossEncoder

from rbi_rag.vector_store import search

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 100

ABBREVIATIONS = {
    "non-performing asset": "NPA",
    "know your customer": "KYC",
    "customer due diligence": "CDD",
    "cash reserve ratio": "CRR",
    "statutory liquidity ratio": "SLR",
    "special mention account": "SMA",
    "priority sector lending": "PSL",
    "anti-money laundering": "AML",
}

_reranker = None


def get_reranker() -> CrossEncoder:
    """Load the reranker once, then reuse it."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_NAME)
    return _reranker


def expand_query(question: str) -> str:
    """Add the abbreviation or the full form, so both wordings are searched."""
    lower = question.lower()
    words = set(re.findall(r"[a-z]+", lower))
    extra = []
    for full, short in ABBREVIATIONS.items():
        if full in lower and short.lower() not in words:
            extra.append(short)
        elif short.lower() in words and full not in lower:
            extra.append(full)
    return f"{question} ({', '.join(extra)})" if extra else question


def retrieve(question: str, k: int = 5, doc_id: str | None = None) -> list[dict]:
    """Return the k best chunks for the question, best first."""
    expanded = expand_query(question)
    candidates = search(expanded, k=CANDIDATES, doc_id=doc_id)
    if not candidates:
        return []

    pairs = [(expanded, f"{c['heading']}\n{c['text']}") for c in candidates]
    scores = get_reranker().predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = round(float(score), 3)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:k]


if __name__ == "__main__":
    questions = [
        "When does a loan account become a non-performing asset?",
        "Can a bank issue a credit card without the customer's consent?",
        "What documents are needed to open an account for an individual?",
    ]
    for q in questions:
        print(f"\n=== {expand_query(q)}")
        for h in retrieve(q, k=3):
            print(f"{h['rerank_score']:>7}  (vector {h['score']})  {h['chunk_id']} | {h['doc_title']} | p{h['page']} | {h['heading']}")
            print("         ", h["text"][:150])