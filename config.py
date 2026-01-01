# import os

# OPENAI_API_KEY = ""  
# # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# #RAW_DATA_PATH = "data/raw/pdf_data"
# PROCESSED_DATA_PATH = "json/chunks"

# CHUNK_SIZE = 400       # tokens per chunk
# CHUNK_OVERLAP = 80     # token overlap
#new  code -----
"""
Configuration file for Stryker RAG ChatBot
"""

import os


# OpenAI API Configuration
OPENAI_API_KEY = ""

# Server Configuration
SERVER_HOST =  "0.0.0.0"
SERVER_PORT = 8000
DEBUG_MODE = "False"

# Database Configuration
DB_PATH = "data/vector_db"
COLLECTION_NAME = "rag_collection"

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4-turbo-preview"

# Performance Configuration
MAX_CONTEXT_CHUNKS =  int(10)
TIMEOUT_SECONDS = int(30)
MAX_CONVERSATION_HISTORY =  int(6)

# WebSocket Configuration
WS_PING_INTERVAL = int(20)
WS_PING_TIMEOUT = int(20)

# Company Configuration
COMPANY_ID = "stryker_bd"
COMPANY_NAME = "Stryker Bangladesh"
SUPPORT_EMAIL = "info@drinkstryker.com"
SUPPORT_PHONE = "+8801770375731"
SUPPORT_WHATSAPP = "https://wa.me/8801770375731"

# CORS Configuration
ALLOWED_ORIGINS ="*"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'