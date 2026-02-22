"""
RAG-powered Crop Advisor
Retrieves relevant agricultural knowledge chunks from numpy vector store,
then uses Groq LLM to generate a grounded answer in the farmer's language.
"""
import os
import json
import logging
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = logging.getLogger(__name__)

_groq_client = None
_embedding_model = None
_vector_store = None

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading multilingual embedding model...")
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embedding_model


def _load_vector_store():
    """Load the numpy vector store from disk."""
    global _vector_store
    if _vector_store is None:
        store_path = os.getenv("VECTOR_STORE_PATH", "../DATABASE/vector_store")
        emb_file = os.path.join(store_path, "embeddings.npy")
        meta_file = os.path.join(store_path, "metadata.json")

        if not os.path.exists(emb_file) or not os.path.exists(meta_file):
            logger.warning("Vector store not found! Run services/rag_indexer.py first.")
            return None

        embeddings = np.load(emb_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        _vector_store = {
            "embeddings": embeddings,
            "texts": metadata["texts"],
            "sources": metadata["sources"],
            "count": metadata["count"],
        }
        logger.info(f"Vector store loaded: {_vector_store['count']} chunks")
    return _vector_store


def _retrieve_chunks(query: str, n_results: int = 3) -> list[str]:
    """Embed query and find top-N most similar chunks via cosine similarity."""
    store = _load_vector_store()
    if store is None or store["count"] == 0:
        return []

    model = _get_embedding_model()
    query_embedding = model.encode(query)

    # Cosine similarity: dot product of normalized vectors
    embeddings = store["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normalized = embeddings / norms
    query_norm = query_embedding / (np.linalg.norm(query_embedding) or 1)

    similarities = normalized @ query_norm
    top_indices = np.argsort(similarities)[-n_results:][::-1]

    chunks = [store["texts"][i] for i in top_indices]
    logger.info(f"Retrieved {len(chunks)} RAG chunks for query: '{query[:60]}'")
    return chunks


SWAHILI_SYSTEM = """Wewe ni mtaalamu wa kilimo Kenya. Unajibu maswali ya wakulima kwa Kiswahili.
Jibu kwa ufupi na kwa uwazi. Toa jibu la vitendo ambalo mkulima anaweza kutumia mara moja.
Tumia habari zilizotolewa hapa chini kama msingi wa jibu lako.
Ikiwa habari hazitoshi, tumia ujuzi wako wa kilimo wa Kenya."""

KIKUYU_SYSTEM = """Wee nĩwe njũrũrĩri wa ũrimũ wa Kenya. Ũkĩũra maciira ma arĩmĩ na Gĩkũyũ.
Tũma ũhoro ũrĩa ũrĩ haha thĩ ũgĩe nĩguo ũhure.]
Nĩ ũgĩe ũhure nĩ ũũru na na kĩndũ gĩkia arĩmĩ angĩkĩra gĩtĩkĩo gaaku."""


def get_crop_advice(transcript: str, lang: str) -> str:
    """
    Retrieve relevant agricultural chunks, then ask Groq LLM for grounded advice.

    Args:
        transcript: Farmer's message about their crop problem.
        lang: 'sw' (Swahili) or 'ki' (Kikuyu).

    Returns:
        Expert advice string in the farmer's language.
    """
    try:
        chunks = _retrieve_chunks(transcript)
        context = "\n\n---\n\n".join(chunks) if chunks else "Hakuna habari maalum."
        system_prompt = KIKUYU_SYSTEM if lang == "ki" else SWAHILI_SYSTEM

        user_message = f"""Habari kutoka kwa mkulima:
\"{transcript}\"

Habari za kilimo (RAG):
{context}

Toa jibu la kitaalamu la vitendo."""

        client = _get_groq()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"[Crop Advisor] Answer ({lang}): '{answer[:100]}'")
        return answer

    except Exception as e:
        logger.error(f"Crop advisor error: {e}")
        if lang == "ki":
            return "Samahani, nĩ na thĩna. Jaribu tena baadaye."
        return "Samahani, kuna tatizo. Jaribu tena baadaye."
