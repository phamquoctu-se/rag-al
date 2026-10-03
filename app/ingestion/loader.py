from pathlib import Path
from dataclasses import dataclass


@dataclass
class RawDocument:
    source_file: str   # basename, e.g. "ky_thuat_nuoi_tom.pdf"
    content: str       # full extracted text
    metadata: dict     # e.g. {"pages": 42, "format": "pdf"}


def load_pdf(file_path: Path) -> RawDocument:
    """Extract text from a PDF file using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    content = "\n\n".join(pages_text)
    return RawDocument(
        source_file=file_path.name,
        content=content,
        metadata={"pages": len(reader.pages), "format": "pdf"},
    )


def load_docx(file_path: Path) -> RawDocument:
    """Extract text from a DOCX file using python-docx."""
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    content = "\n\n".join(paragraphs)
    return RawDocument(
        source_file=file_path.name,
        content=content,
        metadata={"paragraphs": len(paragraphs), "format": "docx"},
    )


def load_markdown(file_path: Path) -> RawDocument:
    """Load plain markdown / text file."""
    content = file_path.read_text(encoding="utf-8")
    return RawDocument(
        source_file=file_path.name,
        content=content,
        metadata={"format": "markdown"},
    )


def load_document(file_path: Path) -> RawDocument:
    """Auto-detect file type and load accordingly."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        return load_docx(file_path)
    elif suffix in (".md", ".txt"):
        return load_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_directory(directory: Path) -> list[RawDocument]:
    """
    Load all supported documents from a directory (non-recursive).
    Supported formats: .pdf, .docx, .md, .txt
    """
    supported = {".pdf", ".docx", ".doc", ".md", ".txt"}
    docs = []
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            doc = load_document(file_path)
            docs.append(doc)
    return docs
