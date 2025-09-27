import os
import uuid
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RAG:
    def __init__(self, model_name="google/flan-t5-base"):
        self.collection_name = "docs"
        self.client = QdrantClient(host="localhost", port=6333)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def index_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        text = ""
        if ext == ".pdf":
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"[RAG] Unsupported file type: {ext}")
            return

        if not text.strip():
            print(f"[RAG] No text found in {filepath}")
            return

        self.add_document(text, metadata={"source": os.path.basename(filepath)})
        print(f"[RAG] Indexed file: {filepath}")

    def add_document(self, text, metadata=None):
        chunks = self.text_splitter.split_text(text)
        embeddings = self.embedder.encode(chunks).tolist()
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk, **(metadata or {})}
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"[RAG] Added {len(points)} chunks from {metadata.get('source') if metadata else 'unknown'}")

    def retrieve(self, query, top_k=5, file_filter=None):
        query_embedding = self.embedder.encode(query).tolist()
        if file_filter:
            flt = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=file_filter))]
            )
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=flt
            )
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
        return [hit.payload["text"] for hit in results]

    def generate_answer(self, query, context_chunks):
        context = "\n".join(context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def is_indexed(self, filename):
        result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filename))]
            ),
            limit=1
        )
        return len(result) > 0

    def count_points_for_file(self, filename):
        result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filename))]
            ),
            limit=10000
        )
        return len(result)

    def delete_file(self, filename):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=filename))]
            )
        )
        print(f"[RAG] Deleted vectors for file: {filename}")
