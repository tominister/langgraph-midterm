import uuid
from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[dict]:
    """Chunk text into overlapping windows of approx chunk_size."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    text_len = len(text)
    idx = 0
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append({
            "id": f"chunk_{idx}_{uuid.uuid4().hex[:8]}",
            "text": chunk,
            "start": start,
            "end": end,
            "index": idx,
        })
        idx += 1
        if end == text_len:
            break
        start = end - overlap
    return chunks
