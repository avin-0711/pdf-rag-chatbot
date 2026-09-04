from src.database import VectorDatabase
from src.embeddings import EmbeddingModel


class Retriever:
    """Retrieve relevant chunks from selected PDF documents."""

    def __init__(
        self,
        database: VectorDatabase | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.database = database or VectorDatabase()
        self.embedding_model = embedding_model or EmbeddingModel()

    def retrieve(
        self,
        question: str,
        document_ids: list[int],
        n_results: int = 5,
        min_similarity: float = 0.25,
    ) -> list[dict]:

        # ==========================================
        # Validate document IDs
        # ==========================================

        if not document_ids:
            return []

        # Remove duplicates
        document_ids = list(
            dict.fromkeys(document_ids)
        )

        query_embedding = self.embedding_model.encode([question])[0]
        rows = self.database.search_chunks(
            query_embedding=query_embedding,
            document_ids=document_ids,
            n_results=n_results,
        )
        return [
            {
                "text": row["content"],
                "source": row["filename"],
                "page": row["page_number"],
                "similarity": float(row.get("similarity", 0.0)),
                "chunk_id": row.get("id"),
            }
            for row in rows
            if float(row.get("similarity", 0.0)) >= min_similarity
        ]