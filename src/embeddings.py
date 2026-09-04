from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Lazy sentence-transformer wrapper used by the vector database."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
