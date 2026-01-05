import json
import numpy as np
from typing import List, Dict

from app.core.cache import RedisClient
from app.ingestion.embedder import EmbeddingService
from app.retrieval.hnsw import HNSWIndex
from app.retrieval.utils import hash_query

class Retriever:
    def __init__(
        self,
        embedder: EmbeddingService,
        index: HNSWIndex,
        top_k: int = 20,
    ):
        self.embedder = embedder
        self.index = index
        self.top_k = top_k
        self.redis = RedisClient.get_client()

    def _cache_key(self, query: str) -> str:
        qh = hash_query(query)
        return f"ann:{self.embedder.model_name}:{qh}:{self.top_k}"

    def retrieve(self, query: str) -> List[Dict]:
        cache_key = self._cache_key(query)

        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # 1. Embed query (single vector)
        query_vec = self.embedder.embed_texts([query])[0]

        # 2. ANN search
        results = self.index.search(
            np.array(query_vec, dtype="float32"),
            k=self.top_k,
        )

        # 3. Cache ANN result (short TTL)
        self.redis.setex(
            cache_key,
            300,  # 5 min TTL
            json.dumps(results),
        )

        return results
