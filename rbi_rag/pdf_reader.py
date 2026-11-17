"""Read RBI PDF files and extract their text.page by page."""

from pathlib import Path
import pymupdf

def list_pdfs(folder:Path)->list[Path]:
    """Return all PDF files in the folder, sorted by name."""
    return sorted(folder.glob("*.pdf"))

def read_pdf(pdf_path:Path)->list[str]:
    """Read a list with the text of each page(index 0 = page 1)."""
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]

if __name__ == "__main__":
    pdf_files=list_pdfs(Path("data/raw_pdfs"))
    total_pages = 0 

    for pdf_path in pdf_files:
        pages = read_pdf(pdf_path)
        total_pages += len(pages)
        print(f"{pdf_path.name}: {len(pages)} pages")

    print("Files:", len(pdf_files))
    print("Total pages:", total_pages)
    if pdf_files:
        first_pages = read_pdf(pdf_files[0])
        print("\n---Page 2 of the first PDF---")
        print(first_pages[min(1, len(first_pages)-1)][:800])
        
              