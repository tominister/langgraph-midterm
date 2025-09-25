import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List

from .config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBED_DIM


class QdrantStore:
    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT, collection: str = QDRANT_COLLECTION):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        # Ensure collection exists. qdrant-client API versions vary; be tolerant of differences.
        try:
            coll = self.client.get_collection(collection_name=collection)
            exists = bool(coll)
        except Exception:
            exists = False

        if not exists:
            # create/recreate collection
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def upsert(self, points: List[PointStruct]):
        # qdrant-client upload_points expects PointStruct-like objects with .id/.vector/.payload
        # Log types and sample content to help debug issues where dicts slip through.
        logging.debug(f"Upsert called with {len(points)} points. Sample types: {[type(p) for p in points[:3]]}")
        for i, p in enumerate(points[:3]):
            try:
                if isinstance(p, dict):
                    logging.debug(f"point[{i}] dict keys: {list(p.keys())}")
                else:
                    logging.debug(f"point[{i}] attrs: id={getattr(p,'id',None)} payload_keys={list(getattr(p,'payload',{}).keys()) if getattr(p,'payload',None) else None}")
            except Exception:
                logging.debug(f"point[{i}] preview failed: {p}")

        # Validate and normalize vectors
        normalized = []
        for p in points:
            if isinstance(p, dict):
                pid = p.get('id')
                vec = p.get('vector')
                payload = p.get('payload')
            else:
                pid = getattr(p, 'id', None)
                vec = getattr(p, 'vector', None)
                payload = getattr(p, 'payload', None)

            # validate vector shape
            if vec is None:
                raise ValueError(f"Point {pid} has no vector")
            if not hasattr(vec, '__len__'):
                raise ValueError(f"Point {pid} vector is not a sequence: {type(vec)}")
            if len(vec) != EMBED_DIM:
                raise ValueError(f"Point {pid} vector length {len(vec)} does not match EMBED_DIM={EMBED_DIM}")

            # construct fresh PointStruct objects to avoid cross-import isinstance issues
            try:
                # If pid is a UUID object, cast to string to satisfy some client versions
                if hasattr(pid, 'hex') or hasattr(pid, 'urn'):
                    pid_for_struct = str(pid)
                else:
                    pid_for_struct = pid
                normalized.append(PointStruct(id=pid_for_struct, vector=list(vec), payload=payload))
            except Exception as e:
                logging.exception("Failed to construct PointStruct; passing raw dict as fallback")
                normalized.append({"id": str(pid), "vector": list(vec), "payload": payload})

        # finally call upload
        try:
            self.client.upload_points(collection_name=self.collection, points=normalized)
        except Exception:
            logging.exception("upload_points failed")
            raise

    def search(self, vector: List[float], top_k: int = 5):
        hits = self.client.search(collection_name=self.collection, query_vector=vector, limit=top_k)
        return hits
