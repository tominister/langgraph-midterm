RAG demo scaffold

This repository contains a minimal scaffold for a Retrieval-Augmented Generation (RAG) system using:
- FastAPI backend
- Qdrant vector store
- SentenceTransformers embeddings
- A pluggable LLM client (configure endpoint in .env)

Quick start

1. Create and activate a virtualenv

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
```

2. Install requirements

```powershell
pip install -r requirements.txt
```

3. Start Qdrant (Docker)

```powershell
./scripts/start_qdrant_docker.ps1
```

4. Run the backend

```powershell
uvicorn backend.app:app --reload --port 8000
```

5. Open the frontend

Navigate to `http://localhost:8000/frontend/index.html` (or serve the frontend folder with any static server). The simple UI uploads a document, ingests it, and lets you ask queries.

Notes
- Configure environment variables using `.env` or export them before running.
- The `llm_client` is a stub; set `LLM_ENDPOINT` and `LLM_API_KEY` in env to enable real LLM calls.
- This scaffold is intended to be a starting point. Improve chunking, error handling, and security before production.
