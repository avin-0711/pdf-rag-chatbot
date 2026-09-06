import numpy as np

from src.database import VectorDatabase
from src.embeddings import EmbeddingModel


class Retriever:
    """Retrieve relevant PDF chunks using local cosine similarity."""

    def __init__(self):
        self.database = VectorDatabase()
        self.embedding_model = EmbeddingModel()

    def retrieve(
        self,
        question: str,
        n_results: int = 5,
    ) -> list[dict]:

        # ==========================================
        # 1. Create embedding for the question
        # ==========================================

        query_embedding = self.embedding_model.encode(
            [question]
        )[0]

        query_vector = np.array(
            query_embedding,
            dtype=np.float32
        )

        if query_vector.size != 384:
            raise ValueError(
                f"Query embedding has {query_vector.size} dimensions. "
                f"Expected 384."
            )

        # Normalize query
        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return []

        query_vector = query_vector / query_norm

        # ==========================================
        # 2. Get stored chunks from Supabase
        # ==========================================

        response = (
            self.database.client
            .table("chunks")
            .select(
                "id, document_id, filename, page_number, "
                "chunk_index, content, embedding"
            )
            .limit(1000)
            .execute()
        )

        rows = response.data or []

        # ==========================================
        # 3. No documents found
        # ==========================================

        if not rows:
            return []

        valid_rows = []
        vectors = []

        # ==========================================
        # 4. Convert stored embeddings
        # ==========================================

        for row in rows:

            embedding = row.get("embedding")

            if embedding is None:
                continue

            try:

                if isinstance(embedding, str):

                    cleaned = embedding.strip()

                    if cleaned.startswith("["):
                        cleaned = cleaned[1:]

                    if cleaned.endswith("]"):
                        cleaned = cleaned[:-1]

                    values = [
                        float(value.strip())
                        for value in cleaned.split(",")
                        if value.strip()
                    ]

                    vector = np.array(
                        values,
                        dtype=np.float32
                    )

                else:

                    vector = np.array(
                        embedding,
                        dtype=np.float32
                    )

                # Make sure dimensions match
                if vector.size != 384:
                    continue

                # Normalize stored vector
                norm = np.linalg.norm(vector)

                if norm == 0:
                    continue

                vector = vector / norm

                vectors.append(vector)
                valid_rows.append(row)

            except Exception:
                continue

        # ==========================================
        # 5. No valid embeddings
        # ==========================================

        if not vectors:
            return []

        # ==========================================
        # 6. Calculate cosine similarity
        # ==========================================

        matrix = np.vstack(vectors)

        similarities = matrix @ query_vector

        # ==========================================
        # 7. Sort by similarity
        # ==========================================

        top_indices = np.argsort(
            similarities
        )[::-1][:n_results]

        # ==========================================
        # 8. Build results
        # ==========================================

        results = []

        for index in top_indices:

            row = valid_rows[index]

            results.append({
                "text": row["content"],
                "source": row["filename"],
                "page": row["page_number"],
                "similarity": float(
                    similarities[index]
                ),
            })

        return results