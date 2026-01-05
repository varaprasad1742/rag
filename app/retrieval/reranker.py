import json
from typing import List, Dict
from sentence_transformers import CrossEncoder

from app.core.cache import RedisClient
from app.retrieval.utils import hash_query

class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ):
        self.model_name = model_name
        self.top_n = top_n
        self.model = CrossEncoder(model_name)
        self.redis = RedisClient.get_client()

    def _cache_key(self, query: str, k: int) -> str:
        qh = hash_query(query)
        return f"rerank:{self.model_name}:{qh}:{k}:{self.top_n}"

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []

        cache_key = self._cache_key(query, len(candidates))
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["score"] = float(s)

        reranked = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True,
        )[: self.top_n]

        self.redis.setex(
            cache_key,
            300,  # 5 min TTL
            json.dumps(reranked),
        )

        return reranked
