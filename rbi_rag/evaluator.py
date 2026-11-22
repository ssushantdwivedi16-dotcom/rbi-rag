"""Measure retrieval quality on test questions whose correct answers we know."""

import json
from pathlib import Path

from rbi_rag.retriever import retrieve
from rbi_rag.vector_store import get_collection, search

EVAL_FILE = Path("eval/questions.json")
K = 5


def load_questions() -> list[dict]:
    """Read the test questions from the JSON file."""
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))


def check_phrases(questions: list[dict]) -> bool:
    """Make sure every expected phrase really exists in its document."""
    collection = get_collection()
    all_ok = True
    for q in questions:
        found = collection.get(
            where={"doc_id": q["doc_id"]},
            where_document={"$contains": q["phrase"]},
        )
        if not found["ids"]:
            print(f"WARNING: not found in {q['doc_id']}: {q['phrase']!r}")
            all_ok = False
    return all_ok


def rank_of_answer(hits: list[dict], q: dict) -> int | None:
    """Position (1, 2, 3...) of the first correct hit, or None if missing."""
    for position, hit in enumerate(hits, start=1):
        if hit["doc_id"] == q["doc_id"] and q["phrase"] in hit["text"]:
            return position
    return None


def evaluate(mode: str, questions: list[dict]) -> None:
    """Run every question through one retrieval mode and print the scores."""
    ranks = []
    for q in questions:
        if mode == "vector":
            hits = search(q["question"], k=K)
        else:
            hits = retrieve(q["question"], k=K)
        rank = rank_of_answer(hits, q)
        ranks.append(rank)
        print(f"  {('#' + str(rank)) if rank else 'miss':>5}  {q['question']}")

    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hitk = sum(1 for r in ranks if r)
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"  => Hit@1 {hit1}/{n} | Hit@{K} {hitk}/{n} | MRR {mrr:.2f}")


if __name__ == "__main__":
    questions = load_questions()
    print(f"Questions: {len(questions)}")
    if not check_phrases(questions):
        print("Fix the warnings above, or those questions can never pass.\n")

    for mode in ["vector", "rerank"]:
        print(f"\n--- {mode}")
        evaluate(mode, questions)