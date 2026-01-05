from fastapi import FastAPI
from app.core.logging import setup_logging
from app.api import ingest, query

setup_logging()

app = FastAPI(title="Production RAG System")

app.include_router(ingest.router)
app.include_router(query.router)

@app.get("/health")
async def health():
    return {"status": "ok"}
