"""
Step 5 — Manual retrieval verification.
Run: python test_retrieval.py
"""
from runners.retriever import get_retriever

QUERIES = [
    "How does JavaScript handle division by zero?",
    "What is memoization and how does Python cache function results?",
    "How do parameterized queries prevent SQL injection?",
]

def main():
    print("Initializing retriever (embedding RAG contexts)...\n")
    retriever = get_retriever()
    print(f"Indexed {len(retriever._chunks)} RAG context chunks.\n")
    print("=" * 70)

    for query in QUERIES:
        print(f"Query: {query}")
        print("-" * 70)
        results = retriever.retrieve(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  #{i} [{result['id']} / {result['task_type']}] (score: {result['score']:.4f})")
            print(f"      {result['context'][:200]}...")
            print()
        print("=" * 70)

if __name__ == "__main__":
    main()
