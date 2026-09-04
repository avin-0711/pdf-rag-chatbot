from src.embeddings import EmbeddingModel

model = EmbeddingModel()

texts = ["This is a test sentence for our PDF RAG chatbot."]

vectors = model.encode(texts)

print("Number of vectors:", len(vectors))
print("Embedding dimensions:", len(vectors[0]))
print("First 5 values:", vectors[0][:5])