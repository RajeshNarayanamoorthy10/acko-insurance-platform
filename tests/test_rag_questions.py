"""
Test the RAG chatbot against a broader set of questions spanning all 3 policy docs.
Checks: does retrieval pull from the right document(s)? Is the answer accurate?
Is response time reasonable (<5 seconds target)?
"""

import sys
import time
from pathlib import Path

# Allow importing from src/rag/ when running this script directly
# Allow importing the src package when running this script directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.chatbot import load_vector_store, answer_question

TEST_QUESTIONS = [
    # --- Motor policy questions ---
    "Is bike insurance mandatory in India?",
    "What is NCB and how much can I save with it?",
    "What does IDV mean for my car insurance?",

    # --- Health policy questions ---
    "Is Panchakarma treatment covered under my health plan?",
    "What is the waiting period before maternity claims are covered?",
    "If I choose a hospital room above my room rent limit, what happens to my claim?",

    # --- FAQ document questions ---
    "What is Acko's claims settlement ratio?",
    "Is flood damage covered under comprehensive car insurance?",

    # --- Cross-document (should possibly pull from multiple sources) ---
    "Does Acko cover mental health hospitalisation?",

    # --- Out-of-scope (should NOT hallucinate an answer) ---
    "What's the best pizza place in Chennai?",
]


def run_tests():
    print("Loading vector store...\n")
    store = load_vector_store()

    results = []

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"{'='*70}")
        print(f"Test {i}/{len(TEST_QUESTIONS)}: {question}")

        start = time.time()
        result = answer_question(store, question, log_to_db=False)
        elapsed = time.time() - start

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")
        print(f"\nResponse time: {elapsed:.2f}s {'✅' if elapsed < 5 else '⚠️  SLOW'}")

        results.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "time_seconds": round(elapsed, 2),
        })
                # Stay comfortably under the free tier's 5 requests/minute limit
        if i < len(TEST_QUESTIONS):
            time.sleep(13)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    avg_time = sum(r["time_seconds"] for r in results) / len(results)
    slow_count = sum(1 for r in results if r["time_seconds"] >= 5)
    print(f"Total questions tested: {len(results)}")
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Questions over 5s: {slow_count}")


if __name__ == "__main__":
    run_tests()