# vector_db.py
import os
import json
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from config import PROCESSED_DATA_PATH, OPENAI_API_KEY

# =========================
# Configuration
# =========================
DB_PATH = "data/vector_db"
COLLECTION_NAME = "rag_collection"

# =========================
# Initialize Chroma Client
# =========================
client = PersistentClient(path=DB_PATH)

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large"
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# =========================
# Helpers
# =========================
def safe_metadata(chunk: dict, chunk_id: str) -> dict:
    """
    Flattens and sanitizes metadata for ChromaDB.
    Converts lists to comma-separated strings.
    """
    metadata = {}

    # Copy all existing metadata fields
    if isinstance(chunk.get("metadata"), dict):
        for k, v in chunk["metadata"].items():
            # Convert lists to comma-separated strings
            if isinstance(v, list):
                metadata[k] = ", ".join(map(str, v))
            else:
                metadata[k] = v

    # Add standard fields for filtering
    metadata.update({
        "id": chunk_id,
        "company_id": chunk.get("metadata", {}).get("company_id", "stryker_bd"),
        "type": chunk.get("metadata", {}).get("type") or chunk.get("metadata", {}).get("category"),
        "category": chunk.get("metadata", {}).get("category"),
        "language": chunk.get("metadata", {}).get("language", "en"),
    })

    # Remove None values
    return {k: v for k, v in metadata.items() if v is not None}

# =========================
# Ingestion Logic
# =========================
def ingest_all_json():
    json_files = [
        f for f in os.listdir(PROCESSED_DATA_PATH)
        if f.endswith(".json")
    ]

    if not json_files:
        print("❌ No JSON files found to ingest.")
        return

    total_chunks = 0
    skipped_chunks = 0

    for json_file in json_files:
        file_path = os.path.join(PROCESSED_DATA_PATH, json_file)

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                chunks = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in {json_file}: {e}")
                continue

        if not isinstance(chunks, list):
            print(f"❌ {json_file} is not a list. Skipping.")
            continue

        print(f"\n📄 Ingesting {json_file} ({len(chunks)} chunks)")

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id") or f"{json_file}_chunk_{i}"

            # Accept both 'document' and 'content'
            content = chunk.get("document") or chunk.get("content")
            if not content or not isinstance(content, str):
                print(f"⚠️  Skipping chunk without content: {chunk_id}")
                skipped_chunks += 1
                continue

            metadata = safe_metadata(chunk, chunk_id)

            try:
                collection.add(
                    ids=[chunk_id],
                    documents=[content],
                    metadatas=[metadata]
                )
                total_chunks += 1
            except Exception as e:
                print(f"❌ Failed to add chunk {chunk_id}: {e}")
                skipped_chunks += 1

        print(f"✅ {json_file} ingested successfully")

    print("\n==============================")
    print("🎉 INGESTION COMPLETE")
    print(f"📦 Total chunks ingested: {total_chunks}")
    print(f"⏭️  Chunks skipped: {skipped_chunks}")
    print(f"📊 Collection size: {collection.count()}")
    print("==============================\n")

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    ingest_all_json()
