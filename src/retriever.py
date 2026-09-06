import numpy as np

from src.database import VectorDatabase
from src.embeddings import EmbeddingModel


class Retriever:
    def __init__(
        self,
        database=None,
        embedding_model=None,
    ):
        self.database = database or VectorDatabase()
        self.embedding_model = embedding_model or EmbeddingModel()

    def _parse_embedding(self, value):
        """
        Convert a pgvector value such as:
        '[0.123,0.456,...]'
        into a NumPy array.
        """

        if value is None:
            return None

        try:
            if isinstance(value, str):
                value = value.strip()

                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1]

                if not value:
                    return None

                values = [
                    float(x.strip())
                    for x in value.split(",")
                    if x.strip()
                ]

                return np.array(values, dtype=np.float32)

            if isinstance(value, list):
                return np.array(value, dtype=np.float32)

            return None

        except Exception:
            return None

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:

        if not query or not query.strip():
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        query_vector = np.array(
            query_embedding,
            dtype=np.float32,
        )

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return []

        query_vector = query_vector / query_norm

        # Get all stored chunks from Supabase
        result = (
            self.database.client
            .table("chunks")
            .select(
                "id, document_id, filename, page_number, "
                "chunk_index, content, embedding"
            )
            .execute()
        )

        rows = result.data or []

        if not rows:
            return []

        scored_results = []

        for row in rows:

            stored_vector = self._parse_embedding(
                row.get("embedding")
            )

            if stored_vector is None:
                continue

            # Only compare vectors with the same dimensions
            if len(stored_vector) != len(query_vector):
                continue

            stored_norm = np.linalg.norm(stored_vector)

            if stored_norm == 0:
                continue

            stored_vector = stored_vector / stored_norm

            # Cosine similarity
            similarity = float(
                np.dot(query_vector, stored_vector)
            )

            scored_results.append(
                {
                    "text": row.get("content", ""),
                    "source": row.get("filename", "Unknown"),
                    "page": row.get("page_number", "?"),
                    "chunk_index": row.get("chunk_index", 0),
                    "similarity": similarity,
                }
            )

        # Highest similarity first
        scored_results.sort(
            key=lambda x: x["similarity"],
            reverse=True,
        )

        return scored_results[:n_results]