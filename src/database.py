import os

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
                "page_number": chunk.get(
                    "page",
                    chunk.get("page_number", 0)
                ),
                "chunk_index": chunk.get(
                    "chunk_index",
                    index
                ),
                "content": str(
                    chunk["text"]
                ),
                "embedding": embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
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