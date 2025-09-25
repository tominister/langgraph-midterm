from typing import Dict
from .retriever import Retriever
from .llm_client import LLMClient
from .config import CHUNK_SIZE, CHUNK_OVERLAP


class RAGPipeline:
    def __init__(self, collection: str = None):
        self.retriever = Retriever(collection=collection)
        self.llm = LLMClient()

    def build_prompt(self, question: str, contexts: list) -> str:
        parts = ["You are a helpful assistant. Use the provided context to answer the question. Cite sources by [id].\n"]
        parts.append("CONTEXTS:\n")
        for c in contexts:
            parts.append(f"[{c.get('id')}] {c.get('payload', {}).get('text','')[:1000]}\n---\n")
        parts.append("QUESTION:\n" + question + "\n\nAnswer:" )
        return "\n".join(parts)

    def answer(self, question: str, top_k: int = 3) -> Dict:
        hits = self.retriever.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, hits)
        answer = self.llm.generate(prompt)
        # Sanitize hits for user-facing output: keep id, score, file_id, index and a short snippet
        sanitized = []
        for h in hits:
            payload = h.get('payload', {}) if isinstance(h, dict) else {}
            snippet = (payload.get('text') or '')[:400].replace('\n', ' ')
            sanitized.append({
                'id': h.get('id'),
                'score': h.get('score'),
                'file_id': payload.get('file_id'),
                'index': payload.get('index'),
                'snippet': snippet,
            })

        return {
            'answer': answer,
            'sources': sanitized,
        }
