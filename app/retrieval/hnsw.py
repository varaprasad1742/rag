import faiss
import json
import os
import threading
from typing import List, Dict
import numpy as np

INDEX_PATH = "data/faiss.index"
META_PATH = "data/metadata.jsonl"

class HNSWIndex:
    def __init__(
        self,
        dim: int,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        self.dim = dim
        self.lock = threading.Lock()

        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.index = faiss.IndexHNSWFlat(dim, m)
            self.index.hnsw.efConstruction = ef_construction

        self.index.hnsw.efSearch = ef_search

        os.makedirs("data", exist_ok=True)

    def add(self, vectors: np.ndarray, metadatas: List[Dict]):
        assert vectors.shape[1] == self.dim
        assert len(vectors) == len(metadatas)

        with self.lock:
            start_id = self.index.ntotal
            self.index.add(vectors)

            with open(META_PATH, "a") as f:
                for i, meta in enumerate(metadatas):
                    meta["_vector_id"] = start_id + i
                    f.write(json.dumps(meta) + "\n")

            faiss.write_index(self.index, INDEX_PATH)

    def search(self, query_vector: np.ndarray, k: int = 10):
        with self.lock:
            distances, indices = self.index.search(
                query_vector.reshape(1, -1), k
            )

        results = []
        if indices[0][0] == -1:
            return results

        needed_ids = set(indices[0].tolist())

        with open(META_PATH, "r") as f:
            for line in f:
                meta = json.loads(line)
                if meta["_vector_id"] in needed_ids:
                    results.append(meta)

        return results
