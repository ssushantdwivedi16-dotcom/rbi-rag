"""Clean Raw pdf page text: remove page numbers and repeated headers/footers, 
rejoin broken links"""

import re
from collections import Counter

PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$",re.IGNORECASE)
NEW_BLOCK = re.compile(
    r"^(chapter\s+[ivxlc]+\b|annex|[a-z]\.\s|\d+(\.\d+)*\.\s|\([a-z0-9]+\)\s)",
    re.IGNORECASE
) 

def find_repeated_lines(pages:list[str], min_share:float=0.5)->set[str]:
    """Return lines that appear on at least half the pages (headers/footers)"""
    if len(pages)<3:
        return set()
    counts = Counter()
    for page in pages:
        counts.update({line.strip() for line in page.splitlines() if line.strip()})
    limit = len(pages)*min_share
    return {line for line , n in counts.items() if n >=limit}

def join_broken_lines(lines:list[str])-> str:
    """Join lines into paragraph; start a new one at chapters, heading and numbered points."""
    paragraphs = []
    for line in lines:
        if paragraphs and not NEW_BLOCK.match(line):
            paragraphs[-1]+=" "+ line
        else:
            paragraphs.append(line)
    return "\n".join(paragraphs)

def clean_page(text:str,repeated:set[str])->str:
    """Remove empty lines , page numbers and headers/footers , then rejoin lines."""
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line or PAGE_NUMBER.match(line) or line in repeated:
            continue
        kept.append(line)
    return join_broken_lines(kept)

def clean_document(pages:list[str])->list[str]:
    """Clean every page of one pdf"""
    repeated= find_repeated_lines(pages)
    return [clean_page(page, repeated) for page in pages]

if __name__=="__main__":
    from pathlib import Path

    from rbi_rag.pdf_reader import list_pdfs, read_pdf

    pdf_path = list_pdfs(Path("data/raw_pdfs"))[0]
    pages = read_pdf(pdf_path)
    cleaned  = clean_document(pages)

    print("Repeated lines removed:", find_repeated_lines(pages))
    print("\n---BEFORE(page 2)---")
    print(pages[1][:600])
    print("\n---AFTER(page 2)---")
    print(cleaned[1][:600])
