from typing import List


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def _load(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError("sentence-transformers not installed") from e
            self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._load()
        embs = self.model.encode(texts, show_progress_bar=False)
        return embs.tolist() if hasattr(embs, 'tolist') else embs

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
