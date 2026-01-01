"""
Production-Ready WebSocket RAG Chatbot Server
Enhanced with Intelligence + Off-Topic Detection - Stryker Bangladesh
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from chromadb import PersistentClient
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

DB_PATH = "data/vector_db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4-turbo-preview"
MAX_CONTEXT_CHUNKS = 10
TIMEOUT_SECONDS = 25
MAX_CONVERSATION_HISTORY = 6

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stryker RAG ChatBot API",
    description="Production WebSocket-based RAG System with Off-Topic Detection",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    chroma_client = PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("ChromaDB and OpenAI clients initialized")
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    raise

INTENT_KEYWORDS = {
    "product": {
        "keywords": ["product", "flavor", "flavour", "drink", "bottle", "available", "stock", "taste", "list", "show me", "what do you have", "electrolyte"],
        "short_answer": False,
        "on_topic": True
    },
    "pricing": {
        "keywords": ["price", "cost", "how much", "expensive", "cheap", "afford", "bdt", "taka", "tk"],
        "short_answer": True,
        "on_topic": True
    },
    "order": {
        "keywords": ["order", "buy", "purchase", "bulk", "wholesale", "quantity", "deliver"],
        "short_answer": False,
        "on_topic": True
    },
    "career": {
        "keywords": ["job", "career", "hiring", "position", "work", "employ", "vacancy", "apply", "salary", "recruit"],
        "short_answer": False,
        "on_topic": True
    },
    "contact": {
        "keywords": ["contact", "reach", "phone", "email", "whatsapp", "call", "message", "address", "location"],
        "short_answer": True,
        "on_topic": True
    },
    "brand": {
        "keywords": ["about", "company", "who", "what is stryker", "tell me about", "founder", "mission", "vision", "story"],
        "short_answer": False,
        "on_topic": True
    },
    "investment": {
        "keywords": ["invest", "partner", "collaboration", "business", "funding", "franchise"],
        "short_answer": False,
        "on_topic": True
    },
    "support": {
        "keywords": ["help", "support", "agent", "human", "talk to someone", "representative", "complain", "issue"],
        "short_answer": True,
        "on_topic": True
    },
    "health": {
        "keywords": ["health", "benefit", "nutrition", "fitness", "workout", "hydration", "exercise", "sugar free"],
        "short_answer": False,
        "on_topic": True
    }
}

OFF_TOPIC_KEYWORDS = {
    "general_knowledge": ["capital", "country", "president", "history", "geography", "who invented", "when was", "what is the"],
    "philosophy": ["why am i", "meaning of life", "purpose", "exist", "philosophy", "soul"],
    "technology": ["how to code", "python", "javascript", "programming", "computer", "software"],
    "science": ["physics", "chemistry", "biology", "universe", "planet", "space"],
    "entertainment": ["movie", "song", "game", "tv show", "celebrity", "actor"],
    "math": ["calculate", "solve", "equation", "math", "formula"],
    "personal": ["my life", "my problem", "advice on", "should i", "what should i do"],
    "weather": ["weather", "temperature", "rain", "forecast"],
    "news": ["news", "politics", "election", "government"],
    "other": ["joke", "story", "poem", "riddle", "translate"]
}

HANDOVER_TRIGGERS = [
    "could not find", "don't have information", "unable to", 
    "not sure", "unclear", "apologise"
]

def is_question_on_topic(question: str) -> Tuple[bool, str]:
    """
    Determines if question is related to Stryker business
    Returns: (is_on_topic: bool, category: str)
    """
    q_lower = question.lower()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "sup", "yo"]
    if any(q_lower.strip().startswith(g) for g in greetings) and len(q_lower) < 25:
        return True, "greeting"
    
    stryker_indicators = ["stryker", "drink", "electrolyte", "beverage", "your company", "your product"]
    if any(indicator in q_lower for indicator in stryker_indicators):
        return True, "company_related"
    
    for intent, config in INTENT_KEYWORDS.items():
        if any(kw in q_lower for kw in config["keywords"]):
            return True, intent
    
    for category, keywords in OFF_TOPIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return False, category
    
    general_patterns = [
        "what is the", "who is the", "when was", "where is", "how many people",
        "capital of", "population of", "why am i", "meaning of", "purpose of life"
    ]
    if any(pattern in q_lower for pattern in general_patterns):
        if not any(word in q_lower for word in ["stryker", "company", "you", "your", "here"]):
            return False, "general_knowledge"
    
    return True, "unknown"  # Default to allowing, but will be caught by RAG if irrelevant

def generate_off_topic_response(category: str, question: str) -> str:
    """Generate friendly off-topic responses"""
    
    responses = {
        "general_knowledge": "I'm here to help with questions about Stryker products, orders, and services! For general knowledge, you might want to try a search engine. 😊\n\nWhat can I help you with regarding Stryker?",
        
        "philosophy": "That's an interesting philosophical question! However, I'm specialized in helping with Stryker beverages—products, orders, and company info. 😊\n\nHow can I assist you with Stryker?",
        
        "technology": "I'm focused on Stryker's electrolyte drinks and business services rather than tech topics. 😊\n\nAny questions about our products or services?",
        
        "science": "While that's a fascinating topic, I specialize in Stryker's electrolyte beverages! 🥤\n\nWant to know about our products, flavors, or prices?",
        
        "entertainment": "I'm here to talk about Stryker drinks, not entertainment! \n\nCurious about our flavors or want to place an order?",
        
        "math": "I'm better at calculating the perfect hydration solution than math problems! \n\nLet me help you with Stryker products instead?",
        
        "personal": "I appreciate you sharing, but I'm specifically designed to help with Stryker business matters. For personal advice, consider talking to someone you trust. 💙\n\nCan I help with our products or services?",
        
        "weather": "I don't have weather info, but I can tell you our electrolyte drinks are perfect for any weather! ☀️💧\n\nWant to explore our flavors?",
        
        "news": "I focus on Stryker updates rather than general news. \n\nInterested in our latest products or company news?",
        
        "other": "That's outside my expertise! I'm here to help with Stryker beverages, orders, careers, and company info. 😊\n\nWhat would you like to know about Stryker?"
    }
    
    return responses.get(category, 
        "I'm Stryker's AI assistant, specialized in our products and services! 🥤\n\n"
        "I can help with:\n"
        "• Product information & flavors\n"
        "• Pricing & orders\n"
        "• Career opportunities\n"
        "• Contact details\n\n"
        "What can I help you with?"
    )

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
            "message_count": 0,
            "last_intent": None,
            "off_topic_count": 0
        }
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.user_sessions:
            del self.user_sessions[client_id]
        logger.info(f"Client {client_id} disconnected")
    
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
            if len(self.user_sessions[client_id]["conversation"]) > MAX_CONVERSATION_HISTORY:
                self.user_sessions[client_id]["conversation"] = \
                    self.user_sessions[client_id]["conversation"][-MAX_CONVERSATION_HISTORY:]
    
    def get_history(self, client_id: str) -> List[Dict]:
        return self.user_sessions.get(client_id, {}).get("conversation", [])
    
    def get_message_count(self, client_id: str) -> int:
        return self.user_sessions.get(client_id, {}).get("message_count", 0)
    
    def increment_message_count(self, client_id: str):
        if client_id in self.user_sessions:
            self.user_sessions[client_id]["message_count"] += 1
    
    def increment_off_topic(self, client_id: str):
        if client_id in self.user_sessions:
            self.user_sessions[client_id]["off_topic_count"] += 1
    
    def get_off_topic_count(self, client_id: str) -> int:
        return self.user_sessions.get(client_id, {}).get("off_topic_count", 0)

manager = ConnectionManager()

async def get_embedding(text: str) -> List[float]:
    """Generate embeddings with error handling"""
    try:
        response = await asyncio.wait_for(
            openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:8000]
            ),
            timeout=TIMEOUT_SECONDS
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def detect_intent_smart(question: str) -> Tuple[str, bool]:
    """
    Smart intent detection with response length determination
    Returns: (intent, short_answer)
    """
    q_lower = question.lower()
    
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "sup"]
    if any(q_lower.startswith(g) for g in greetings) and len(q_lower) < 20:
        return "greeting", True
    
    intent_scores = {}
    for intent, config in INTENT_KEYWORDS.items():
        score = sum(1 for kw in config["keywords"] if kw in q_lower)
        if score > 0:
            intent_scores[intent] = score
    
    if intent_scores:
        primary_intent = max(intent_scores, key=intent_scores.get)
        short_answer = INTENT_KEYWORDS[primary_intent]["short_answer"]
        return primary_intent, short_answer
    
    return "general", False

def should_show_products(intent: str, question: str) -> bool:
    """Decide if products should be included in response"""
    
    if intent in ["product", "pricing"]:
        return True
    
    if intent == "order":
        return True
    
    product_mentions = ["flavor", "flavour", "drink", "bottle", "blue raspberry", "mango", "strawberry", "mixed"]
    if any(mention in question.lower() for mention in product_mentions):
        return True
    
    return False

def format_context(chunks: List[str], metadata: List[dict]) -> tuple[str, List[dict], str]:
    """Build optimized context with metadata extraction"""
    
    products = []
    info_sections = []
    contact_info = {}
    career_info = []
    
    for chunk, meta in zip(chunks, metadata):
        doc_type = meta.get("type", "info")
        
        if doc_type == "product":
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
    
    # Add formatted products prominently
    if products:
        context = "**AVAILABLE PRODUCTS:**\n"
        for i, p in enumerate(products, 1):
            context += (
                f"{i}. {p['name']} - {p['flavor']}\n"
                f"   Price: {p['price']} {p['currency']} | Sugar-free: {'Yes' if p['sugar_free'] else 'No'}\n"
                f"   Link: {p['url']}\n"
            )
        context += "\n\n" + "\n\n".join(info_sections)
    
    # Detect context type
    context_type = "general"
    if products:
        context_type = "product"
    elif career_info:
        context_type = "career"
    elif contact_info:
        context_type = "contact"
    
    return context, products, context_type

def build_system_prompt(intent: str, context_type: str, context: str, short_answer: bool, message_count: int) -> str:
    """Build optimized system prompt with response length guidance"""
    
    response_style = "concise and friendly" if message_count > 3 else "warm and helpful"
    length_guidance = "1-2 sentences" if short_answer else "2-4 sentences"
    
    base_prompt = f"""You are Stryker's AI assistant - intelligent, helpful, and professional.

**RESPONSE STYLE:**
- Keep responses {length_guidance} (user expects {response_style} answers)
- Match the complexity of the user's question
- Short question = short answer, detailed question = detailed answer
- Don't repeat information from previous messages

**BRAND VOICE:**
- Energetic yet professional
- Health-conscious and motivating
- Use British English
- Speak as "we/us/our"

**CRITICAL RULES:**
- Use ONLY information from the provided context
- For products: Use EXACT names, flavors, and prices from context
- NEVER invent product names or prices
- NEVER answer questions unrelated to Stryker business
- If uncertain: Offer human support via WhatsApp/Email
- Never make assumptions

"""
    
    intent_guidance = {
        "product": """**PRODUCT FOCUS:**
- List products with exact flavor names and prices from context
- Mention sugar-free and electrolyte benefits briefly
- Include product links
- Keep it scannable""",
        
        "pricing": """**PRICING FOCUS:**
- State exact prices clearly
- One sentence about bulk discounts if relevant
- Provide WhatsApp for wholesale inquiries""",
        
        "order": """**ORDER FOCUS:**
- Confirm what they want to order
- State pricing
- Direct to WhatsApp: +8801770375731 for orders
- Keep it simple""",
        
        "career": """**CAREER FOCUS:**
- Share role details: position, salary, requirements
- Provide application link
- Mention 1-2 key benefits
- Be encouraging""",
        
        "contact": """**CONTACT FOCUS:**
- Provide WhatsApp: +8801770375731
- Provide Email: info@drinkstryker.com
- One sentence max""",
        
        "support": """**SUPPORT FOCUS:**
- Acknowledge their need
- Provide WhatsApp: +8801770375731 (fastest)
- Email: info@drinkstryker.com
- Keep it brief and helpful"""
    }
    
    prompt = base_prompt + intent_guidance.get(intent, "Answer directly from context.")
    prompt += f"\n\n**CONTEXT:**\n{context}\n"
    
    return prompt

async def generate_response(
    question: str,
    client_id: str,
    conversation_history: List[Dict]
) -> dict:
    """Generate intelligent response using RAG with off-topic detection"""
    
    try:
        is_on_topic, category = is_question_on_topic(question)
        
        if not is_on_topic:
            logger.info(f"OFF-TOPIC: {category} | Question: {question[:50]}")
            manager.increment_off_topic(client_id)
            
            off_topic_response = generate_off_topic_response(category, question)
            
            return {
                "type": "message",
                "content": off_topic_response,
                "intent": "off_topic",
                "category": category,
                "confidence": "high",
                "is_off_topic": True
            }
        
        intent, short_answer = detect_intent_smart(question)
        message_count = manager.get_message_count(client_id)
        manager.increment_message_count(client_id)
        
        logger.info(f"ON-TOPIC: {intent} | Short: {short_answer} | MsgCount: {message_count}")
        
        if intent == "greeting":
            greetings = [
                "Hello! How can I help you with Stryker today?",
                "Hi there! What can I help you with?",
                "Hey! Interested in our products or have questions?"
            ]
            return {
                "type": "message",
                "content": greetings[message_count % len(greetings)],
                "intent": intent,
                "confidence": "high"
            }
        
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
                "content": "I don't have specific information about that. Would you like to contact our team?\n\n📱 WhatsApp: +8801770375731\n📧 Email: info@drinkstryker.com",
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
        
        # Build system prompt with intelligence
        system_prompt = build_system_prompt(intent, context_type, context, short_answer, message_count)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history
        messages.extend(conversation_history[-4:])
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        # Generate response with dynamic max_tokens
        max_tokens = 300 if short_answer else 700
        
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens
            ),
            timeout=TIMEOUT_SECONDS
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Check if handover needed
        needs_handover = any(trigger in answer.lower() for trigger in HANDOVER_TRIGGERS)
        if intent in ["support", "investment"]:
            needs_handover = True
        
        # Smart product inclusion
        show_products = should_show_products(intent, question)
        
        return {
            "type": "message",
            "content": answer,
            "intent": intent,
            "products": products if (show_products and products) else None,
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
        logger.error(f"Response error: {e}", exc_info=True)
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
            "content": "Connected to Stryker AI Assistant"
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
                
                # Add assistant response to history (only for on-topic responses)
                if response.get("type") == "message" and not response.get("is_off_topic"):
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
    return {"status": "online", "service": "Stryker RAG ChatBot", "version": "2.1"}

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