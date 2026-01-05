import hashlib
from typing import List, Dict

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict]:
    words = text.split()
    chunks = []

    start = 0
    idx = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunk_id = _hash(chunk_text)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "index": idx,
        })

        idx += 1
        start = end - overlap

    return chunks
