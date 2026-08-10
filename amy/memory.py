"""
ChromaDB-backed episodic memory for Amy.

Public API (called sequentially at session end by companion.py):
    update_memory(transcript: list[dict]) -> None
    query_memory(query: str, n: int = 5) -> list[dict]
"""

import json
import os
from datetime import datetime

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

_DB_PATH = os.path.join(os.path.dirname(__file__), ".chroma")
_COLLECTION_NAME = "amy_episodes"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    if not _CHROMA_AVAILABLE:
        return None
    _client = chromadb.PersistentClient(path=_DB_PATH)
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def update_memory(transcript: list[dict]) -> None:
    """
    Persist a session transcript to ChromaDB as episodic memory.

    Each user/assistant turn is stored as a separate document so
    semantic retrieval works at turn granularity.
    """
    col = _get_collection()
    if col is None:
        return

    session_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    documents, metadatas, ids = [], [], []

    for i, turn in enumerate(transcript):
        role = turn.get("role", "unknown")
        content = turn.get("content", "").strip()
        if not content:
            continue
        documents.append(content)
        metadatas.append({"role": role, "session_id": session_id, "turn": i})
        ids.append(f"{session_id}_{i}")

    if documents:
        col.add(documents=documents, metadatas=metadatas, ids=ids)


def query_memory(query: str, n: int = 5) -> list[dict]:
    """
    Retrieve the n most semantically relevant memory turns for a query.

    Returns a list of dicts with keys: content, role, session_id, distance.
    """
    col = _get_collection()
    if col is None:
        return []

    results = col.query(query_texts=[query], n_results=n)
    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"content": doc, "distance": dist, **meta})
    return out
