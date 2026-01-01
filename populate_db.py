"""
Enhanced ChromaDB Population Script
Production-ready with better error handling and validation
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
    """Load and validate data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

def generate_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Generate embeddings in batches with progress tracking"""
    embeddings = []
    batch_size = 50 
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            logger.info(f"Processing batch {batch_num}/{total_batches}...")
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            logger.info(f"Batch {batch_num}/{total_batches} complete")
        except Exception as e:
            logger.error(f"Failed on batch {batch_num}: {e}")
            raise
    
    return embeddings

def clean_metadata(metadata: dict) -> dict:
    """Clean and validate metadata for ChromaDB"""
    cleaned = {}
    
    for key, value in metadata.items():
        # Convert lists to comma-separated strings
        if isinstance(value, list):
            cleaned[key] = ", ".join(str(v) for v in value)
        # Convert boolean to string for consistency
        elif isinstance(value, bool):
            cleaned[key] = str(value)
        # Keep compatible types
        elif isinstance(value, (str, int, float)) or value is None:
            cleaned[key] = value
        # Convert everything else to string
        else:
            cleaned[key] = str(value)
    
    return cleaned

def validate_data(data: List[Dict]) -> bool:
    """Validate data structure"""
    required_keys = ['id', 'document', 'metadata']
    
    for idx, item in enumerate(data):
        # Check required keys
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            logger.error(f"✗ Item {idx} missing keys: {missing_keys}")
            return False
        
        # Validate types
        if not isinstance(item['id'], str):
            logger.error(f"Item {idx}: 'id' must be string")
            return False
        
        if not isinstance(item['document'], str):
            logger.error(f"Item {idx}: 'document' must be string")
            return False
        
        if not isinstance(item['metadata'], dict):
            logger.error(f"Item {idx}: 'metadata' must be dict")
            return False
        
        # Validate company_id exists in metadata
        if 'company_id' not in item['metadata']:
            logger.warning(f"Item {idx}: 'company_id' missing in metadata, adding default")
            item['metadata']['company_id'] = 'stryker_bd'
    
    logger.info("✓ Data validation passed")
    return True

def populate_database(data: List[Dict]):
    """Populate ChromaDB with validated data"""
    
    # Initialize clients
    logger.info("Initializing clients...")
    try:
        chroma_client = PersistentClient(path=DB_PATH)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✓ Clients initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize clients: {e}")
        return False
    
    # Create or get collection
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Stryker Bangladesh RAG Collection"}
        )
        logger.info(f"✓ Collection '{COLLECTION_NAME}' ready")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False
    
    # Handle existing data
    existing_count = collection.count()
    if existing_count > 0:
        logger.warning(f"Collection has {existing_count} existing documents")
        response = input("Clear existing data? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            try:
                collection.delete(where={})
                logger.info("Cleared existing data")
            except Exception as e:
                logger.error(f"Failed to clear data: {e}")
                return False
    
    # Prepare data
    ids = []
    documents = []
    metadatas = []
    
    for item in data:
        ids.append(item['id'])
        documents.append(item['document'])
        metadatas.append(clean_metadata(item['metadata']))
    
    logger.info(f"Preparing to insert {len(documents)} documents...")
    
    # Generate embeddings
    logger.info("Generating embeddings (this may take a few minutes)...")
    try:
        embeddings = generate_embeddings(documents, openai_client)
        logger.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return False
    
    # Insert into ChromaDB
    try:
        logger.info("Inserting documents into database...")
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"Successfully inserted {len(documents)} documents")
    except Exception as e:
        logger.error(f"Failed to insert documents: {e}")
        return False
    
    # Verify insertion
    final_count = collection.count()
    logger.info(f"Database now contains {final_count} documents")
    
    # Test query
    logger.info("\nTesting database with sample queries...")
    test_queries = [
        "What products do you have?",
        "How much does it cost?",
        "Are you hiring?"
    ]
    
    for query in test_queries:
        try:
            test_embedding = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            ).data[0].embedding
            
            results = collection.query(
                query_embeddings=[test_embedding],
                n_results=2,
                where={"company_id": {"$eq": "stryker_bd"}}
            )
            
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"Results: {len(results['documents'][0])} documents found")
            for i, doc in enumerate(results['documents'][0], 1):
                logger.info(f"  {i}. {doc[:80]}...")
        except Exception as e:
            logger.error(f"Test query failed: {e}")
    
    logger.info("\n" + "="*70)
    logger.info("Database population complete!")
    logger.info("="*70)
    return True

def main():
    """Main execution function"""
    print("=" * 70)
    print("Stryker ChromaDB Population Script v2.0")
    print("=" * 70)
    print()
    
    # Define data file path
    data_file = "json/chunks/feed.json"
    
    # Check if file exists
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info(f"Current directory: {Path.cwd()}")
        logger.info(f"Looking for: {Path(data_file).absolute()}")
        logger.info("\nPlease ensure the JSON file exists at the specified path")
        return
    
    # Load data
    data = load_json_data(data_file)
    
    if not data:
        logger.error("No data to populate")
        return
    
    # Validate data structure
    if not validate_data(data):
        logger.error("Data validation failed")
        return
    
    # Show summary
    logger.info(f"\n Summary:")
    logger.info(f"   Documents: {len(data)}")
    logger.info(f"   Estimated time: ~{len(data) * 0.5:.0f} seconds")
    
    # Confirm
    response = input("\nProceed with population? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        logger.info("Operation cancelled")
        return
    
    # Populate
    success = populate_database(data)
    
    if success:
        logger.info("\n All done! Your RAG system is ready to use.")
    else:
        logger.error("\n Population failed. Please check the errors above.")

if __name__ == "__main__":
    main()