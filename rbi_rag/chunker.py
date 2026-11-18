"""Split cleaned RBI documents into labelled chunks (one numbered paragraph each)."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rbi_rag.pdf_reader import list_pdfs, read_pdf
from rbi_rag.text_cleaner import clean_document

DOC_TITLES = {
    "150MD": "CRR and SLR Directions, 2025",
    "151MD": "Prudential Norms on Capital Adequacy Directions, 2025",
    "152MD": "Resource Raising Norms Directions, 2025",
    "153MD": "Interest Rate on Deposits Directions, 2025",
    "154MD": "Credit Facilities Directions, 2025",
    "155MD": "Credit Cards and Debit Cards Directions, 2025",
    "156MD": "Credit Information Reporting Directions, 2025",
    "157MD": "Credit Risk Management Directions, 2025",
    "158MD": "Concentration Risk Management Directions, 2025",
    "159MD": "Transfer and Distribution of Credit Risk Directions, 2025",
    "160MD": "Securitisation Transactions Directions, 2025",
    "161MD": "Interest Rates on Advances Directions, 2025",
    "162MD": "Investment Portfolio Directions, 2025",
    "163MD": "Asset Liability Management Directions, 2025",
    "164MD": "IRACP (NPA) Directions, 2025",
    "165MD": "Resolution of Stressed Assets Directions, 2025",
    "166MD": "Wilful Defaulters and Large Defaulters Directions, 2025",
    "167MD": "Financial Statements Directions, 2025",
    "169MD": "Know Your Customer Directions, 2025",
    "170MD": "Responsible Business Conduct Directions, 2025",
    "171MD": "Managing Risks in Outsourcing Directions, 2025",
    "172MD": "Climate Finance and Climate Risk Directions, 2025",
    "174MD": "Miscellaneous Directions, 2025",
}

CHAPTER = re.compile(r"^chapter\s+[ivxlc]+\b", re.IGNORECASE)
HEADING = re.compile(r"^[A-Z]\.\s")
PARAGRAPH = re.compile(r"^\d+(\.\d+)*\.\s")

MAX_WORDS = 300


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    chapter: str
    heading: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_document(pdf_path: Path) -> list[Chunk]:
    """Turn one PDF into labelled chunks."""
    doc_id = pdf_path.stem
    doc_title = DOC_TITLES.get(doc_id, doc_id)
    pages = clean_document(read_pdf(pdf_path))

    chunks: list[Chunk] = []
    chapter, heading = "", ""
    current: list[str] = []
    start_page = 1

    def close_chunk():
        """Save the lines collected so far as one or more chunks(max MAX_WORDS each), then start fresh."""
        if current:
            words = " ".join(current).split()
            for start in range(0, len(words), MAX_WORDS):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{len(chunks) + 1}",
                        doc_id=doc_id,
                        doc_title=doc_title,
                        page=start_page,
                        chapter=chapter,
                        heading=heading,
                        text=" ".join(words[start : start + MAX_WORDS]),
                    )
                )
            current.clear()

    for page_number, page_text in enumerate(pages, start=1):
        for line in page_text.splitlines():
            if CHAPTER.match(line):
                close_chunk()
                chapter, heading = line, ""
            elif HEADING.match(line):
                close_chunk()
                heading = line
            else:
                too_long = sum(len(part.split()) for part in current) > MAX_WORDS
                if PARAGRAPH.match(line) or too_long:
                    close_chunk()
                if not current:
                    start_page = page_number
                current.append(line)

    close_chunk()
    return chunks


if __name__ == "__main__":
    all_chunks: list[Chunk] = []
    for pdf_path in list_pdfs(Path("data/raw_pdfs")):
        all_chunks.extend(chunk_document(pdf_path))

    sizes = [len(c.text.split()) for c in all_chunks]
    print("Total chunks:", len(all_chunks))
    print("Average words per chunk:", sum(sizes) // len(sizes))
    print("Largest chunk (words):", max(sizes))

    for c in all_chunks[3:6]:
        print(f"\n--- {c.chunk_id} | {c.doc_title} | page {c.page}")
        print("Chapter:", c.chapter)
        print("Heading:", c.heading)
        print(c.text[:300])