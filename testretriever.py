from src.retriever import Retriever

retriever = Retriever()

question = "What is the National Testing Agency?"

print("Question:", question)

results = retriever.retrieve(
    question,
    n_results=5
)

print(f"\nRetrieved results: {len(results)}")

for i, result in enumerate(results, start=1):

    print("=" * 60)
    print(f"RESULT {i}")
    print(f"SOURCE: {result['source']}")
    print(f"PAGE: {result['page']}")
    print(f"SIMILARITY: {result['similarity']:.4f}")
    print(f"TEXT: {result['text'][:500]}")
    print()