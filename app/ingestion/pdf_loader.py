from pypdf import PdfReader
from pathlib import Path

def load_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)
