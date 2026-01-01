from chromadb import PersistentClient

DB_PATH = "data/vector_db"
client = PersistentClient(path=DB_PATH)

collection = client.get_collection("rag_collection")

all_items = collection.get(include=["documents", "metadatas"])
print("Number of documents:", len(all_items.get("documents", [])))

for i, (doc, meta) in enumerate(zip(all_items.get("documents", []), all_items.get("metadatas", []))):
    print(f"\nChunk {i}")
    print("ID:", meta.get("id"))
    print("Source:", meta.get("source"))
    print("Content:", doc[:200], "...")
