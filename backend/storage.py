import os
import uuid
from pathlib import Path

from .config import UPLOAD_DIR

os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload(file_obj, filename: str = None):
    """Save fastapi UploadFile to disk and return file_id and path."""
    file_id = str(uuid.uuid4())
    filename = filename or getattr(file_obj, 'filename', file_id)
    dest = Path(UPLOAD_DIR) / f"{file_id}_{filename}"
    with open(dest, "wb") as f:
        content = file_obj.file.read()
        f.write(content)
    return file_id, str(dest)
