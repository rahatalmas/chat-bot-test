from chromadb import PersistentClient

client = PersistentClient(path="data/vector_db")
collection = client.get_or_create_collection(name="rag_collection")

# Check document count
print(f"Documents in collection: {collection.count()}")