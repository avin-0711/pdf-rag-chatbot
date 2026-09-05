import os

import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

from src.config import get_setting


load_dotenv()


class VectorDatabase:
    """Supabase database for PDF documents and vector chunks."""

    def __init__(self):

        supabase_url = get_setting("SUPABASE_URL")
        supabase_key = get_setting("SUPABASE_KEY")

        if not supabase_url:
            raise ValueError(
                "SUPABASE_URL is missing. Add it to .env locally or Streamlit Cloud Secrets."
            )

        if not supabase_key:
            raise ValueError(
                "SUPABASE_KEY is missing. Add it to .env locally or Streamlit Cloud Secrets."
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

        # Keep retrieval independent of the optional RPC until every deployed
        # Supabase project has the same match_chunks signature and schema cache.
        return self._legacy_search_chunks(
            query_embedding,
            document_ids,
            n_results,
        )

    def searchable_document_ids(
        self,
        document_ids: list[int],
        filenames: list[str],
    ) -> list[int]:
        """Return active IDs that have chunks, repairing stale session IDs by filename."""
        chunk_rows = (
            self.client
            .table("chunks")
            .select("document_id")
            .in_("document_id", document_ids)
            .execute()
            .data
            or []
        )
        searchable_ids = list(dict.fromkeys(row["document_id"] for row in chunk_rows))
        if searchable_ids:
            return searchable_ids

        if not filenames:
            return []

        matching_documents = (
            self.client
            .table("documents")
            .select("id")
            .in_("filename", filenames)
            .execute()
            .data
            or []
        )
        candidate_ids = [row["id"] for row in matching_documents]
        if not candidate_ids:
            return []

        populated = (
            self.client
            .table("chunks")
            .select("document_id")
            .in_("document_id", candidate_ids)
            .execute()
            .data
            or []
        )
        return list(dict.fromkeys(row["document_id"] for row in populated))

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