"""
Module 1 - Step 2: The RAG chatbot itself.

Takes a customer question, retrieves relevant policy chunks from ChromaDB,
and asks Gemini to answer using only that retrieved context.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from src.db.database import get_session
from src.db.models import ChatLog

# --- Load environment variables (GEMINI_API_KEY) from .env ---
load_dotenv()

# --- Config (must match ingest.py so we open the SAME vector store) ---
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"
COLLECTION_NAME = "acko_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3  # how many chunks to retrieve per question

GEMINI_MODEL = "gemini-3.6-flash"  # fast + free-tier friendly

SYSTEM_PROMPT = """You are a helpful assistant for Acko Insurance customers.
Answer the customer's question using ONLY the policy context provided below.
If the answer isn't in the context, say you don't have that information and
suggest they contact Acko support - do NOT make up an answer.

Keep your answer SHORT and direct: 2-4 sentences for simple questions,
at most a short paragraph plus up to 4 bullet points for questions that
need a list (like required documents or a discount schedule). Do NOT use
headers, bold section titles, or multiple sections - plain, conversational
text only. Mention specific numbers (amounts, waiting periods, percentages)
exactly as given in the context, but skip extra caveats or tips unless
directly asked.
"""


def load_vector_store():
    """Open the existing ChromaDB collection built by ingest.py."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store


def retrieve_chunks(vector_store, question: str, k: int = TOP_K):
    """Find the k most relevant policy chunks for this question."""
    results = vector_store.similarity_search(question, k=k)
    return results  # list of Document objects


def build_prompt(question: str, chunks: list) -> str:
    """Combine the question and retrieved chunks into one prompt for Gemini."""
    context_blocks = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_blocks.append(
            f"[Context {i} - from {source}, page {page}]\n{doc.page_content}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_PROMPT}

POLICY CONTEXT:
{context_text}

CUSTOMER QUESTION:
{question}

ANSWER:"""
    return prompt


def ask_gemini(prompt: str, max_retries: int = 2) -> str:
    """Send the prompt to Gemini and return the generated answer.

    Retries automatically if the free-tier rate limit (5 requests/minute)
    is hit, waiting the amount of time Google's API tells us to.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY not set correctly in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return response.text
        except ResourceExhausted as e:
            error_text = str(e)
            if "PerDay" in error_text:
                # Daily quota exhausted - retrying won't help, fail fast
                raise RuntimeError(
                    "Gemini free-tier DAILY quota (20 requests) is used up. "
                    "This resets after 24 hours - try again tomorrow, or "
                    "reduce how many test calls you run per session."
                ) from e
            if attempt == max_retries:
                raise  # give up after final attempt
            wait_seconds = 60  # per-minute limit resets roughly every 60s
            print(f"  Rate limit hit — waiting {wait_seconds}s before retry "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_seconds)


def answer_question(vector_store, question: str, log_to_db: bool = True) -> dict:
    """Full pipeline: retrieve -> build prompt -> generate answer.

    Returns a dict with the answer plus the sources used. Also logs the
    question, answer, and response time to the chat_logs table.
    """
    start_time = time.time()

    chunks = retrieve_chunks(vector_store, question)
    prompt = build_prompt(question, chunks)
    answer = ask_gemini(prompt)

    elapsed_seconds = time.time() - start_time

    sources = [
        {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}
        for doc in chunks
    ]

    if log_to_db:
        session = get_session()
        try:
            record = ChatLog(
                question=question,
                answer=answer,
                response_time_seconds=elapsed_seconds,
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "response_time_seconds": elapsed_seconds,
    }


if __name__ == "__main__":
    # Quick manual test from the command line
    print("Loading vector store...")
    store = load_vector_store()

    test_questions = [
        "Is my bike insurance covering theft if I forgot to lock it?",
        "What is the waiting period for pre-existing diseases?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = answer_question(store, q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")