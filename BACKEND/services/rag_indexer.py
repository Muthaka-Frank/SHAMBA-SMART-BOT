"""
RAG Indexer — One-time script to embed seed documents into a local vector store.
Run this once before starting the Flask app: python services/rag_indexer.py

Uses numpy-based vector store instead of ChromaDB — zero extra dependencies needed.
"""
import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(kb_path: str) -> list[dict]:
    """Load all .txt and .pdf files from the knowledge base directory."""
    docs = []
    kb_dir = Path(kb_path)
    if not kb_dir.exists():
        logger.error(f"Knowledge base path not found: {kb_path}")
        return docs

    for file_path in kb_dir.glob("**/*"):
        if file_path.suffix == ".txt":
            text = file_path.read_text(encoding="utf-8")
            docs.append({"text": text, "source": file_path.name})
            logger.info(f"Loaded TXT: {file_path.name} ({len(text)} chars)")

        elif file_path.suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                docs.append({"text": text, "source": file_path.name})
                logger.info(f"Loaded PDF: {file_path.name} ({len(text)} chars)")
            except Exception as e:
                logger.warning(f"Could not read PDF {file_path.name}: {e}")

    return docs


def _split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Simple recursive text splitter — no langchain needed."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []

    # Try each separator, starting with the most meaningful
    for sep in separators:
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return chunks

        parts = text.split(sep)
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = part.strip()
        if current.strip():
            chunks.append(current.strip())

        if chunks:
            return chunks

    # Fallback: just slice by character
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into ~500-char chunks."""
    chunks = []
    for doc in docs:
        parts = _split_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, part in enumerate(parts):
            chunks.append({
                "text": part,
                "id": f"{doc['source']}_chunk_{i}",
                "source": doc["source"],
            })
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


def index_to_store(chunks: list[dict], store_path: str):
    """Embed chunks and save to a local numpy vector store."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    sources = [c["source"] for c in chunks]
    ids = [c["id"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save everything to disk
    os.makedirs(store_path, exist_ok=True)

    # Save embeddings as numpy array
    np.save(os.path.join(store_path, "embeddings.npy"), embeddings)

    # Save metadata as JSON
    metadata = {
        "texts": texts,
        "sources": sources,
        "ids": ids,
        "model": EMBED_MODEL,
        "count": len(texts),
    }
    with open(os.path.join(store_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Indexed {len(texts)} chunks into vector store at {store_path}")


if __name__ == "__main__":
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "../DATABASE/knowledge_base")
    store_path = os.getenv("VECTOR_STORE_PATH", "../DATABASE/vector_store")

    logger.info("=== Shamba-Smart RAG Indexer ===")
    logger.info(f"Knowledge base: {kb_path}")
    logger.info(f"Vector store:   {store_path}")

    docs = load_documents(kb_path)
    if not docs:
        logger.error("No documents found. Exiting.")
        sys.exit(1)

    chunks = chunk_documents(docs)
    index_to_store(chunks, store_path)
    logger.info("🌱 Indexing complete. You can now start the Flask app.")
