from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import tempfile

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text

from app.ingestion.embedder import EmbeddingService


from app.retrieval.hnsw import HNSWIndex
import numpy as np


router = APIRouter(prefix="/ingest", tags=["ingestion"])
hnsw_index = HNSWIndex(dim=384)

@router.post("/pdfs")
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    results = []
    embedder = EmbeddingService(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    for file in files:
        if not file.filename.endswith(".pdf"):
            results.append({
                "file": file.filename,
                "status": "skipped",
                "reason": "not a pdf"
            })
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        try:
            text = load_pdf(tmp_path)

            if not text.strip():
                results.append({
                    "file": file.filename,
                    "status": "failed",
                    "reason": "empty pdf"
                })
                continue

            chunks = chunk_text(text)
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(chunk_texts)
            vectors = np.vstack(embeddings)

            metadatas = [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "file": file.filename,
                }
                for c in chunks
            ]

            hnsw_index.add(vectors, metadatas)

            results.append({
                "file": file.filename,
                "status": "ingested",
                "num_chunks": len(chunks)
            })

        except Exception as e:
            results.append({
                "file": file.filename,
                "status": "failed",
                "reason": str(e)
            })

        finally:
            tmp_path.unlink(missing_ok=True)

        
    return {
        "total_files": len(files),
        "results": results
    }
