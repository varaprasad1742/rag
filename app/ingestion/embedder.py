import hashlib
import json
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.cache import RedisClient

class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.redis = RedisClient.get_client()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str) -> str:
        return f"embed:{self.model_name}:{self._hash(text)}"

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = [None] * len(texts)
        missing_texts = []
        missing_indices = []

        # 1. Check cache
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self.redis.get(key)

            if cached:
                embeddings[i] = np.array(json.loads(cached))
            else:
                missing_texts.append(text)
                missing_indices.append(i)

        # 2. Compute missing embeddings in batch
        if missing_texts:
            new_embeddings = self.model.encode(
                missing_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            for idx, emb in zip(missing_indices, new_embeddings):
                key = self._cache_key(texts[idx])
                self.redis.set(
                    key,
                    json.dumps(emb.tolist()),
                )
                embeddings[idx] = emb

        return embeddings
