from pathlib import Path


def extract_text_from_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        raise RuntimeError("pypdf not installed; install pypdf or pypdf2")
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)


def extract_text_from_docx(path: str) -> str:
    try:
        import docx
    except Exception:
        raise RuntimeError("python-docx not installed; install python-docx")
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".pdf",):
        return extract_text_from_pdf(path)
    if ext in (".docx", ".doc"):
        return extract_text_from_docx(path)
    # fallback: read as text
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
