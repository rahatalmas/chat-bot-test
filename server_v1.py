"""
Production-Ready WebSocket RAG Chatbot Server
Optimized for Stryker Bangladesh - High Performance & Scalability
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from chromadb import PersistentClient
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

# ==================== CONFIGURATION ====================
DB_PATH = "data/vector_db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4-turbo-preview"
MAX_CONTEXT_CHUNKS = 10
TIMEOUT_SECONDS = 30
MAX_CONVERSATION_HISTORY = 6

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== INITIALIZE APP ====================
app = FastAPI(
    title="Stryker RAG ChatBot API",
    description="Production WebSocket-based RAG System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
try:
    chroma_client = PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("✓ ChromaDB and OpenAI clients initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize: {e}")
    raise

# ==================== INTENT & PATTERNS ====================
INTENT_KEYWORDS = {
    "product": ["product", "flavor", "flavour", "drink", "bottle", "available", "stock", "taste"],
    "pricing": ["price", "cost", "how much", "expensive", "cheap", "afford"],
    "order": ["order", "buy", "purchase", "bulk", "wholesale", "quantity"],
    "career": ["job", "career", "hiring", "position", "work", "employ", "vacancy", "apply"],
    "contact": ["contact", "reach", "phone", "email", "whatsapp", "call", "message"],
    "brand": ["about", "company", "who", "what is stryker", "tell me about"],
    "investment": ["invest", "partner", "collaboration", "business", "funding"],
    "support": ["help", "support", "agent", "human", "talk to someone", "representative"]
}

HANDOVER_TRIGGERS = [
    "could not find", "don't have information", "unable to", 
    "not sure", "unclear", "apologise"
]

# ==================== CONNECTION MANAGER ====================
class ConnectionManager:
    """Manages WebSocket connections with session management"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.user_sessions[client_id] = {
            "conversation": [],
            "connected_at": datetime.now().isoformat(),
            "message_count": 0
        }
        logger.info(f"✓ Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.user_sessions:
            del self.user_sessions[client_id]
        logger.info(f"✗ Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Send error for {client_id}: {e}")
    
    def add_to_history(self, client_id: str, role: str, content: str):
        if client_id in self.user_sessions:
            self.user_sessions[client_id]["conversation"].append({
                "role": role,
                "content": content
            })
            # Keep last N messages
            if len(self.user_sessions[client_id]["conversation"]) > MAX_CONVERSATION_HISTORY:
                self.user_sessions[client_id]["conversation"] = \
                    self.user_sessions[client_id]["conversation"][-MAX_CONVERSATION_HISTORY:]
    
    def get_history(self, client_id: str) -> List[Dict]:
        return self.user_sessions.get(client_id, {}).get("conversation", [])

manager = ConnectionManager()

# ==================== CORE FUNCTIONS ====================
async def get_embedding(text: str) -> List[float]:
    """Generate embeddings with error handling"""
    try:
        response = await asyncio.wait_for(
            openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:8000]  # Truncate if too long
            ),
            timeout=TIMEOUT_SECONDS
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def detect_intent(question: str) -> str:
    """Detect primary intent from question"""
    q_lower = question.lower()
    
    # Count keyword matches for each intent
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q_lower)
        if score > 0:
            intent_scores[intent] = score
    
    if intent_scores:
        return max(intent_scores, key=intent_scores.get)
    return "general"

def format_context(chunks: List[str], metadata: List[dict]) -> tuple[str, List[dict], str]:
    """Build optimized context with metadata extraction"""
    
    products = []
    info_sections = []
    contact_info = {}
    career_info = []
    
    for chunk, meta in zip(chunks, metadata):
        doc_type = meta.get("type", "info")
        
        if doc_type == "product":
            # Handle both boolean and string values for sugar_free
            sugar_free = meta.get("sugar_free", True)
            if isinstance(sugar_free, str):
                sugar_free = sugar_free.lower() == "true"
            
            products.append({
                "name": meta.get("category", "Electrolyte Drink"),
                "flavor": meta.get("flavor", "N/A"),
                "price": meta.get("price_bdt", "N/A"),
                "currency": meta.get("currency", "BDT"),
                "url": meta.get("product_url", "#"),
                "sugar_free": sugar_free
            })
        elif doc_type == "contact":
            contact_info = {
                "email": meta.get("email", "info@drinkstryker.com"),
                "phone": meta.get("phone", "+8801770375731")
            }
            info_sections.append(f"[{doc_type.upper()}]\n{chunk}")
        elif doc_type == "career":
            career_info.append({
                "category": meta.get("category", ""),
                "content": chunk,
                "salary": meta.get("salary_bdt"),
                "apply_url": meta.get("apply_url")
            })
            info_sections.append(f"[{doc_type.upper()} - {meta.get('category', '')}]\n{chunk}")
        else:
            info_sections.append(f"[{doc_type.upper()}]\n{chunk}")
    
    # Build context text
    context = "\n\n".join(info_sections)
    
    # Add formatted products
    if products:
        context += "\n\n**AVAILABLE PRODUCTS:**\n"
        for i, p in enumerate(products, 1):
            context += (
                f"{i}. {p['name']} - {p['flavor']}\n"
                f"   Price: {p['price']} {p['currency']} | Sugar-free: {'Yes' if p['sugar_free'] else 'No'}\n"
                f"   Link: {p['url']}\n"
            )
    
    # Detect context type for prompt optimization
    context_type = "general"
    if products:
        context_type = "product"
    elif career_info:
        context_type = "career"
    elif contact_info:
        context_type = "contact"
    
    return context, products, context_type

def build_system_prompt(intent: str, context_type: str, context: str) -> str:
    """Build optimized system prompt based on intent and context"""
    
    base_prompt = """You are Stryker's AI assistant - intelligent, helpful, and professional.

**BRAND VOICE:**
- Energetic yet professional
- Health-conscious and motivating
- Use British English
- Speak as "we/us/our"
- Be concise but complete

**CORE RULES:**
- Use ONLY provided context
- For products: Include name, flavor, price, and link
- For orders: Provide pricing, then offer WhatsApp contact for bulk
- For careers: Share details and application links
- If uncertain: Offer human support via WhatsApp/Email
- Never invent information

"""
    
    intent_guidance = {
        "product": """**PRODUCT FOCUS:**
- Highlight sugar-free benefits
- Mention electrolyte content for hydration
- Compare flavors if asked
- Always include pricing and links
- Suggest best flavor for their needs""",
        
        "pricing": """**PRICING FOCUS:**
- State exact prices clearly
- Mention bulk discount availability
- Provide WhatsApp contact for wholesale
- Highlight value proposition""",
        
        "order": """**ORDER FOCUS:**
- Confirm product availability
- State standard pricing
- For bulk (>20 bottles): Direct to WhatsApp +8801770375731
- Mention delivery options if in context""",
        
        "career": """**CAREER FOCUS:**
- Describe role clearly
- Mention salary range if available
- Highlight company culture
- Provide application link
- Mention perks and benefits""",
        
        "contact": """**CONTACT FOCUS:**
- Provide email: info@drinkstryker.com
- Provide WhatsApp: +8801770375731
- Mention response time expectations
- Offer social media alternatives""",
        
        "support": """**SUPPORT FOCUS:**
- Acknowledge their need for help
- Provide immediate contact options
- WhatsApp: +8801770375731 (fastest)
- Email: info@drinkstryker.com
- Set expectation for response time"""
    }
    
    prompt = base_prompt + intent_guidance.get(intent, "")
    prompt += f"\n\n**CONTEXT:**\n{context}\n"
    
    return prompt

async def generate_response(
    question: str,
    client_id: str,
    conversation_history: List[Dict]
) -> dict:
    """Generate intelligent response using RAG"""
    
    try:
        # Detect intent
        intent = detect_intent(question)
        
        # Send searching status
        await manager.send_message(client_id, {
            "type": "status",
            "status": "searching",
            "message": "Searching knowledge base..."
        })
        
        # Get embeddings and search
        query_embedding = await get_embedding(question)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=MAX_CONTEXT_CHUNKS,
            where={"company_id": {"$eq": "stryker_bd"}}
        )
        
        chunks = results["documents"][0]
        metadata = results["metadatas"][0]
        
        if not chunks:
            return {
                "type": "message",
                "content": "I apologise, but I couldn't find specific information about that. Would you like to contact our team?\n\n📱 WhatsApp: +8801770375731\n📧 Email: info@drinkstryker.com",
                "intent": intent,
                "needs_handover": True,
                "confidence": "low"
            }
        
        # Format context
        context, products, context_type = format_context(chunks, metadata)
        
        # Send thinking status
        await manager.send_message(client_id, {
            "type": "status",
            "status": "thinking",
            "message": "Generating response..."
        })
        
        # Build messages for GPT
        system_prompt = build_system_prompt(intent, context_type, context)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last N messages)
        messages.extend(conversation_history[-4:])
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Generate response
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            ),
            timeout=TIMEOUT_SECONDS
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Check if handover needed
        needs_handover = any(trigger in answer.lower() for trigger in HANDOVER_TRIGGERS)
        if intent in ["support", "investment"]:
            needs_handover = True
        
        return {
            "type": "message",
            "content": answer,
            "intent": intent,
            "products": products if products else None,
            "needs_handover": needs_handover,
            "confidence": "high" if not needs_handover else "medium"
        }
        
    except asyncio.TimeoutError:
        logger.error("Response generation timeout")
        return {
            "type": "error",
            "content": "Response took too long. Please try again.",
            "error_code": "TIMEOUT"
        }
    except Exception as e:
        logger.error(f"Response error: {e}")
        return {
            "type": "error",
            "content": "Something went wrong. Please contact support.",
            "error_code": "INTERNAL_ERROR"
        }

# ==================== WEBSOCKET ENDPOINT ====================
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        await manager.send_message(client_id, {
            "type": "system",
            "content": "Connected to Stryker AI Assistant 🚀"
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            message_type = data.get("type", "message")
            
            if message_type == "ping":
                await manager.send_message(client_id, {"type": "pong"})
                continue
            
            if message_type == "message":
                user_message = data.get("content", "").strip()
                
                if not user_message:
                    continue
                
                # Add to history
                manager.add_to_history(client_id, "user", user_message)
                
                # Generate response
                conversation = manager.get_history(client_id)
                response = await generate_response(user_message, client_id, conversation)
                
                # Add assistant response to history
                if response.get("type") == "message":
                    manager.add_to_history(client_id, "assistant", response["content"])
                
                # Send response
                await manager.send_message(client_id, response)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

# ==================== HTTP ENDPOINTS ====================
@app.get("/")
async def root():
    return {"status": "online", "service": "Stryker RAG ChatBot"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20
    )