"""Rebuild OpenSearch keyword index from persisted Chroma chunks."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings  # noqa: E402
from app.retrieval.opensearch_store import ensure_index, get_opensearch_client, index_documents  # noqa: E402


def main() -> int:
    if not settings.opensearch_enabled:
        print("OPENSEARCH_ENABLED is false")
        return 1

    client = get_opensearch_client()
    if client is None:
        print("OpenSearch client unavailable")
        return 1

    index_name = settings.opensearch_index_name
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    if not ensure_index():
        print("OpenSearch index init failed")
        return 1

    chunks = read_chroma_chunks()

    indexed_count = index_documents(chunks)
    print(f"Indexed {indexed_count}/{len(chunks)} chunks into {index_name}")
    return 0 if indexed_count == len(chunks) else 1


def read_chroma_chunks() -> list:
    import chromadb
    from langchain_core.documents import Document

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    chunks = []
    for collection in client.list_collections():
        result = collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for content, metadata in zip(documents, metadatas):
            safe_metadata = metadata if isinstance(metadata, dict) else {}
            if not safe_metadata.get("knowledge_base_type"):
                safe_metadata = {**safe_metadata, "knowledge_base_type": "enterprise"}
            chunks.append(Document(page_content=content or "", metadata=safe_metadata))
    return chunks


if __name__ == "__main__":
    raise SystemExit(main())
