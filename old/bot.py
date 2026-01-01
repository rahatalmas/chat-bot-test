# intelligent_rag_bot_v3.py
import re
from chromadb import PersistentClient
from openai import OpenAI
from config import OPENAI_API_KEY

DB_PATH = "data/vector_db"
COLLECTION_NAME = "rag_collection"

client = PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Keywords for human handover
ORDER_KEYWORDS = ["order", "bulk", "purchase", "buy"]
INVESTMENT_KEYWORDS = ["invest", "investment", "partnership", "funding"]
HUMAN_REQUEST_KEYWORDS = ["human", "agent", "support", "talk to someone"]

UNCERTAINTY_PHRASES = [
    "could not find any relevant information",
    "i'm sorry for any confusion",
    "i do not have access",
    "we apologise",
    "i'm afraid",
]

# ===== Generate embeddings =====
def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# ===== Generate intelligent answer =====
def generate_answer(user_question: str, top_k: int = 10):
    query_embedding = get_embedding(user_question)
    where_filter = {"company_id": {"$eq": "stryker_bd"}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metas = results["metadatas"][0]

    if not retrieved_chunks:
        return "We apologise, but we could not find any relevant information regarding your question."

    # Build structured context for GPT
    context_text = ""
    product_list = []
    for chunk, meta in zip(retrieved_chunks, retrieved_metas):
        if meta.get("type") == "product":
            product_list.append({
                "name": meta.get("category", "Unknown Product"),
                "flavor": meta.get("flavor", "N/A"),
                "price": meta.get("price_bdt", "N/A"),
                "currency": meta.get("currency", "BDT"),
                "url": meta.get("product_url", "N/A")
            })
        else:
            context_text += f"\n--- {meta.get('type','info')} ---\n{chunk}\n"

    if product_list:
        context_text += "\nPRODUCTS:\n"
        for idx, p in enumerate(product_list, start=1):
            context_text += f"{idx}. {p['name']} ({p['flavor']}), Price: {p['price']} {p['currency']}, [Product Link]({p['url']})\n"

    # GPT prompt engineering
    system_prompt = f"""
You are a highly intelligent customer support assistant for Stryker.
RULES:
- Use only the context provided.
- For product questions, always try to give a complete and structured answer.
- If the user asks about orders, bulk purchases, or investments, provide general guidance and only offer human handover if context is insufficient.
- If the answer is unclear, politely mention human agent availability.
- Always maintain professional, friendly, polite British English.
- Speak as 'we/us/our'.

CONTEXT:
{context_text}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# ===== Decide if human handover is needed =====
def needs_human_handover(answer: str, user_question: str) -> bool:
    answer_lower = answer.lower()
    question_lower = user_question.lower()
    # If GPT is unsure
    if any(phrase in answer_lower for phrase in UNCERTAINTY_PHRASES):
        return True
    # Explicit user request for human
    if any(word in question_lower for word in HUMAN_REQUEST_KEYWORDS):
        return True
    # Bulk/order/investment questions
    if any(word in question_lower for word in ORDER_KEYWORDS + INVESTMENT_KEYWORDS):
        return True
    return False

# ===== Main interactive loop =====
if __name__ == "__main__":
    print("Stryker Super-Intelligent RAG Bot\n")
    while True:
        user_question = input("Enter your question (or type 'exit' to quit): ").strip()
        if user_question.lower() in ["exit", "quit"]:
            break

        answer = generate_answer(user_question)
        print("\nAnswer:", answer)

        if needs_human_handover(answer, user_question):
            print("\nIt seems we may not have enough information to fully answer your question.")
            # Check if user already requested human
            if not any(word in user_question.lower() for word in HUMAN_REQUEST_KEYWORDS):
                user_response = input("Would you like to connect with a human agent? (yes/no): ").strip().lower()
                if user_response in ["yes", "y", "sure"]:
                    print("Currently, no human agent is available. Please try again later.")
                else:
                    print("Okay! Feel free to ask another question.\n")
            else:
                print("Currently, no human agent is available. Please try again later.\n")

        print("\n" + "-"*80 + "\n")
