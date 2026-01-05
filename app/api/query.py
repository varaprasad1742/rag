from fastapi import APIRouter
from app.ingestion.embedder import EmbeddingService
from app.retrieval.hnsw import HNSWIndex
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from app.generation.llm import GeminiLLM
import time 

router = APIRouter(prefix="/query", tags=["query"])

embedder = EmbeddingService(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
index = HNSWIndex(dim=384)
retriever = Retriever(embedder, index)
reranker = Reranker()
llm = GeminiLLM()

@router.post("/")
async def query_rag(payload: dict):
    query = payload.get("query")
    if not query:
        return {"error": "query is required"}
    start = time.time()
    ann_results = retriever.retrieve(query)
    reranked = reranker.rerank(query, ann_results)
    answer = llm.generate(query, reranked)
    print(time.time()-start)
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "file": c["file"],
                "chunk_id": c["chunk_id"],
            }
            for c in reranked
        ],
    }
