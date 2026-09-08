import os
from dotenv import load_dotenv

load_dotenv(override=True)

google_api_key = os.getenv("GOOGLE_API_KEY")
open_router_api_key = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")

gemini_base_url = os.getenv("gemini_base_url")
open_router_base_url = os.getenv("open_router_base_url")

MODEL = 'gemini-2.5-flash-lite'
OR_MODEL = "inclusionai/ling-3.0-flash-fin:free"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# chunk_size = 200
# chunk_overlap = 50
RETRIEVE_K = 6
# RELEVANCE_THRESHOLD = 1.0
GOOD_ABSOLUTE_SCORE = 1.3
ABSOLUTE_CEILING = 2.0
RELATIVE_RATIO = 0.85

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "user_docs"

SYSTEM_PROMPT = """You are a helpful, honest assistant integrated into a document Q&A application.

The user may or may not have uploaded documents. Depending on the question, you will either be given:
1. Context retrieved from the user's uploaded documents, or
2. No context at all, meaning you should answer using your own general knowledge.

Follow these rules at all times:
- If you are given document context, base your answer primarily on it.
- If the context only partially answers the question, use it for what it covers, and clearly supplement the rest with your own general knowledge. Explicitly distinguish which part of your answer came from the document versus your own knowledge.
- If no context is given, or the context is unrelated to the question, answer using your general knowledge.
- Never fabricate facts, sources, or details that aren't grounded in either the provided context or your own reliable knowledge.
- If you are unsure or the information isn't available to you, say so plainly rather than guessing.
- Keep answers clear and reasonably concise unless the user asks for more detail.
- Maintain a friendly, conversational tone, and use the ongoing conversation history to understand follow-up questions and references to earlier turns.
"""

