import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .storage import save_upload
from .extractor import extract_text
from .chunker import chunk_text
from .embedder import Embedder
from .vectorstore import QdrantStore
from .rag import RAGPipeline
from .config import CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_COLLECTION
from .llm_client import LLMClient

app = FastAPI(title="RAG Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve the frontend static files at /frontend
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


class IngestRequest(BaseModel):
    file_id: str
    path: str
    collection: str = QDRANT_COLLECTION


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_id, path = save_upload(file, file.filename)
    return {"file_id": file_id, "path": path}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        # extract
        if not os.path.exists(req.path):
            raise HTTPException(status_code=404, detail="file not found")
        text = extract_text(req.path)
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        # embed
        embedder = Embedder()
        texts = [c['text'] for c in chunks]
        embs = embedder.embed_texts(texts)
        # upsert to qdrant
        store = QdrantStore(collection=req.collection)
        import uuid

        points = []
        for c, v in zip(chunks, embs):
            # Qdrant requires point IDs to be either unsigned ints or UUIDs.
            # Use a fresh UUID for the Qdrant point id, and keep the original
            # chunk id inside the payload so we can reference it later.
            pid = uuid.uuid4()
            payload = {"text": c['text'], "file_id": req.file_id, "index": c['index'], "chunk_id": c['id']}
            points.append({
                "id": pid,
                "vector": v,
                "payload": payload,
            })
        # qdrant upload: prefer using the library model objects but delegate
        # normalization to QdrantStore.upsert so we never pass raw dicts
        # directly into low-level client methods (which can expect .id attributes).
        try:
            from qdrant_client.models import PointStruct
            points_structs = [PointStruct(id=p['id'], vector=p['vector'], payload=p['payload']) for p in points]
            store.upsert(points_structs)
        except Exception as e:
            # If constructing PointStructs or the preferred path fails,
            # fall back to letting QdrantStore.upsert normalize dicts itself.
            logging.debug(f"PointStruct path failed, falling back to dict upsert: {e}")
            store.upsert(points)
        return {"ingested": True, "num_chunks": len(chunks)}
    except Exception as e:
        logging.exception("Error during ingest")
        tb = traceback.format_exc()
        # Return error details (trim long tracebacks) to help debugging in dev
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb[:2000]})


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    collection: str = QDRANT_COLLECTION


@app.post("/query")
async def query(req: QueryRequest):
    rag = RAGPipeline(collection=req.collection)
    result = rag.answer(req.query, top_k=req.top_k)
    return result


class LLMTestRequest(BaseModel):
    prompt: str


@app.post('/llm_test')
async def llm_test(req: LLMTestRequest):
    client = LLMClient()
    try:
        out = client.generate(req.prompt)
        return {'ok': True, 'response': out}
    except Exception as e:
        return JSONResponse(status_code=500, content={'ok': False, 'error': str(e)})
