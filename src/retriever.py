from src.database import VectorDatabase
from src.embeddings import EmbeddingModel

import numpy as np


class Retriever:
    """Retrieve relevant chunks from one selected PDF."""

    def __init__(self):

        self.database = VectorDatabase()
        self.embedding_model = EmbeddingModel()

    def retrieve(
        self,
        question: str,
        document_id: int,
        n_results: int = 5,
    ) -> list[dict]:

        # ==========================================
        # 1. Generate question embedding
        # ==========================================

        query_embedding = self.embedding_model.encode(
            [question]
        )[0]

        query_vector = np.array(
            query_embedding,
            dtype=np.float32
        )

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return []

        query_vector = (
            query_vector / query_norm
        )

        # ==========================================
        # 2. Get ONLY chunks from selected PDF
        # ==========================================

        response = (
            self.database.client
            .table("chunks")
            .select(
                "id, document_id, filename, "
                "page_number, chunk_index, "
                "content, embedding"
            )
            .eq(
                "document_id",
                document_id
            )
            .execute()
        )

        rows = response.data or []

        if not rows:
            return []

        # ==========================================
        # 3. Convert embeddings
        # ==========================================

        valid_rows = []
        vectors = []

        for row in rows:

            embedding = row.get("embedding")

            if not embedding:
                continue

            try:

                if isinstance(
                    embedding,
                    str
                ):

                    values = (
                        embedding
                        .strip("[]")
                        .split(",")
                    )

                    vector = np.array(
                        [
                            float(x)
                            for x in values
                        ],
                        dtype=np.float32
                    )

                else:

                    vector = np.array(
                        embedding,
                        dtype=np.float32
                    )

                if len(vector) != 384:
                    continue

                norm = np.linalg.norm(vector)

                if norm == 0:
                    continue

                vector = vector / norm

                vectors.append(vector)
                valid_rows.append(row)

            except Exception:
                continue

        if not vectors:
            return []

        # ==========================================
        # 4. Cosine similarity
        # ==========================================

        matrix = np.vstack(vectors)

        similarities = (
            matrix @ query_vector
        )

        # ==========================================
        # 5. Top-K
        # ==========================================

        top_indices = np.argsort(
            similarities
        )[::-1][:n_results]

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