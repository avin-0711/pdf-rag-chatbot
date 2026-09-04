import os

import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()


class VectorDatabase:
    """Supabase database for PDF documents and vector chunks."""

    def __init__(self):

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url:
            raise ValueError(
                "SUPABASE_URL is missing from the .env file."
            )

        if not supabase_key:
            raise ValueError(
                "SUPABASE_KEY is missing from the .env file."
            )

        self.client: Client = create_client(
            supabase_url,
            supabase_key
        )

    # ==========================================
    # DOCUMENTS
    # ==========================================

    def add_document(
        self,
        filename: str,
        file_hash: str
    ) -> int:

        response = (
            self.client
            .table("documents")
            .insert({
                "filename": filename,
                "file_hash": file_hash
            })
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Failed to create document."
            )

        return response.data[0]["id"]

    # ==========================================
    # CHUNKS
    # ==========================================

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings,
        document_id: int,
        document_name: str
    ) -> int:

        rows = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            rows.append({
                "document_id": document_id,
                "filename": document_name,
                "page_number": int(
                    chunk.get("page_number", 0)
                ),
                "chunk_index": index,
                "content": str(
                    chunk["text"]
                ),
                "embedding": (
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                )
            })

        if not rows:
            return 0

        response = (
            self.client
            .table("chunks")
            .insert(rows)
            .execute()
        )

        return len(response.data or [])

    def search_chunks(
        self,
        query_embedding: list[float],
        document_ids: list[int],
        n_results: int = 5,
    ) -> list[dict]:
        """Run Top-K cosine search in Supabase through the pgvector RPC."""
        if not document_ids:
            return []

        try:
            response = self.client.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_document_ids": document_ids,
                    "match_count": n_results,
                },
            ).execute()
            return response.data or []
        except Exception as error:
            error_code = getattr(error, "code", None)
            error_text = str(error)
            missing_rpc = (
                error_code == "PGRST202"
                or "Could not find the function public.match_chunks" in error_text
                or "schema cache" in error_text
            )
            if not missing_rpc:
                raise
            return self._legacy_search_chunks(
                query_embedding,
                document_ids,
                n_results,
            )

    def _legacy_search_chunks(
        self,
        query_embedding: list[float],
        document_ids: list[int],
        n_results: int,
    ) -> list[dict]:
        """Compatibility path until the new filtered RPC is deployed."""
        response = (
            self.client
            .table("chunks")
            .select(
                "id, document_id, filename, page_number, "
                "chunk_index, content, embedding"
            )
            .in_("document_id", document_ids)
            .execute()
        )
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query /= query_norm

        scored = []
        for row in response.data or []:
            embedding = row.get("embedding")
            if isinstance(embedding, str):
                embedding = [float(value.strip()) for value in embedding.strip("[]").split(",")]
            vector = np.asarray(embedding, dtype=np.float32)
            if vector.shape != query.shape:
                continue
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            scored.append((float(np.dot(vector / norm, query)), row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                **row,
                "similarity": similarity,
            }
            for similarity, row in scored[:n_results]
        ]