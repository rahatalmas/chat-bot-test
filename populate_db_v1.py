"""
Script to populate ChromaDB with Stryker data
Run this once to initialize your vector database
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from chromadb import PersistentClient
from openai import OpenAI
from config import OPENAI_API_KEY, DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def generate_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Generate embeddings for texts in batches"""
    embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    return embeddings

def clean_metadata(metadata: dict) -> dict:
    """Clean metadata to ensure ChromaDB compatibility"""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            cleaned[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned

def populate_database(data: List[Dict]):
    """Populate ChromaDB with data"""
    logger.info("Initializing clients...")
    chroma_client = PersistentClient(path=DB_PATH)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Stryker Bangladesh RAG Collection"}
        )
        logger.info(f"✓ Collection '{COLLECTION_NAME}' ready")
    except Exception as e:
        logger.error(f"✗ Failed to create collection: {e}")
        return
    
    existing_count = collection.count()
    if existing_count > 0:
        logger.warning(f"Collection has {existing_count} existing documents")
        response = input("Clear existing data? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            collection.delete(where={})
            logger.info("✓ Cleared existing data")
    
    ids = []
    documents = []
    metadatas = []
    
    for item in data:
        ids.append(item['id'])
        documents.append(item['document'])
        metadatas.append(clean_metadata(item['metadata']))
    
    logger.info(f"Preparing to insert {len(documents)} documents...")
    logger.info("Generating embeddings (this may take a few minutes)...")
    embeddings = generate_embeddings(documents, openai_client)
    
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Successfully inserted {len(documents)} documents")
    except Exception as e:
        logger.error(f"✗ Failed to insert documents: {e}")
        return
    
    # Verify insertion
    final_count = collection.count()
    logger.info(f"Database now contains {final_count} documents")
    
    # Test query
    logger.info("\nTesting database with sample query...")
    test_query = "What products do you have?"
    test_embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=test_query
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=3,
        where={"company_id": {"$eq": "stryker_bd"}}
    )
    
    logger.info("Sample query results:")
    for i, doc in enumerate(results['documents'][0], 1):
        logger.info(f"{i}. {doc[:100]}...")
    
    logger.info("\nDatabase population complete!")

def main():
    """Main function"""
    print("=" * 70)
    print("Stryker ChromaDB Population Script")
    print("=" * 70)
    print()
    
    data_file = "json/chunks/feed.json"  # Change this to your JSON file location
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please update the data_file path in the script to point to your JSON file")
        logger.info(f"Current path: {Path(data_file).absolute()}")
        return
    
    data = load_json_data(data_file)
    
    if not data:
        logger.error("No data to populate")
        return
    
    required_keys = ['id', 'document', 'metadata']
    for idx, item in enumerate(data):
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            logger.error(f"Item {idx} missing keys: {missing_keys}")
            return
    
    logger.info("Data validation passed")
    
    populate_database(data)

if __name__ == "__main__":
    main()