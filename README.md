# Document RAG Agent

Independent end-to-end RAG application covering document ingestion, embedding, vector retrieval and grounded generation. It is intentionally compact and documents the authentication, access-control and evaluation work required before production use.

A compact retrieval-augmented generation application for asking questions about uploaded PDF and text documents.

## How it works

1. The Flask app accepts a PDF or TXT upload.
2. Text is extracted and split into retrievable chunks.
3. `all-MiniLM-L6-v2` creates embeddings.
4. Qdrant stores and retrieves the most relevant chunks.
5. Flan-T5 generates an answer grounded in the retrieved context.

## Stack

- Python and Flask
- Qdrant vector search
- Sentence Transformers embeddings
- `google/flan-t5-xl`
- PDF and plain-text ingestion

## Run locally

```bash
git clone https://github.com/tominister/document-rag-agent.git
cd document-rag-agent
pip install -r requirements.txt
```

Start Qdrant locally:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

Then run:

```bash
python app.py
```

Open `http://127.0.0.1:5000`, upload a document, and ask a question.

## Limitations

- Answer quality depends on document extraction, chunking, retrieval quality, and the selected generation model.
- The current app is a compact demonstration and does not include production authentication, access controls, or evaluation monitoring.
