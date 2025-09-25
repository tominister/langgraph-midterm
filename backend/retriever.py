from typing import List

from .embedder import Embedder
from .vectorstore import QdrantStore


class Retriever:
    def __init__(self, embed_model: str = None, collection: str = None):
        self.embedder = Embedder(embed_model or 'all-MiniLM-L6-v2')
        self.store = QdrantStore(collection=collection) if collection else QdrantStore()

    def retrieve(self, query: str, top_k: int = 3):
        qvec = self.embedder.embed_text(query)
        hits = self.store.search(qvec, top_k=top_k)
        # convert hits to simple dicts
        results = []
        for h in hits:
            results.append({
                'id': h.id,
                'score': h.score,
                'payload': h.payload,
            })
        return results
