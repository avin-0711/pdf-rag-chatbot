import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.chunker import chunk_pages
from src.database import VectorDatabase
from src.embeddings import EmbeddingModel
from src.generator import generate_answer
from src.pdf_processor import extract_text_from_pdf
from src.retriever import Retriever


load_dotenv()


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")
st.caption(
    "Upload PDF files, index their contents, and ask grounded questions."
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


# ==========================================
# LOAD DATABASE
# ==========================================

@st.cache_resource
def get_database():
    return VectorDatabase()


# ==========================================
# LOAD COMPONENTS
# ==========================================

embedding_model = get_embedding_model()
database = get_database()

# IMPORTANT:
# Do NOT cache Retriever with database/embedding_model
# as arguments because Streamlit tries to hash them.
retriever = Retriever(
    database,
    embedding_model
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📚 Documents")

    uploads = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploads:
        st.caption(
            f"{len(uploads)} PDF(s) selected"
        )

    index_documents = st.button(
        "Index Documents",
        type="primary",
        disabled=not uploads
    )

    # ==========================================
    # INDEX DOCUMENTS
    # ==========================================

    if index_documents and uploads:

        progress = st.progress(
            0,
            text="Starting indexing..."
        )

        total_chunks = 0
        indexed_count = 0
        skipped_count = 0
        failed_count = 0

        for index, upload in enumerate(uploads):

            try:

                # ==========================================
                # READ PDF
                # ==========================================

                file_bytes = upload.getvalue()

                # ==========================================
                # SHA-256 FILE HASH
                # ==========================================

                file_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                # ==========================================
                # CHECK FOR DUPLICATE
                # ==========================================

                existing = (
                    database.client
                    .table("documents")
                    .select("id, filename")
                    .eq("file_hash", file_hash)
                    .execute()
                )

                if existing.data:

                    skipped_count += 1

                    st.info(
                        f"⏭️ {upload.name} is already indexed. Skipping."
                    )

                    progress.progress(
                        (index + 1) / len(uploads),
                        text=f"Skipped {upload.name}"
                    )

                    continue

                # ==========================================
                # TEMPORARY PDF FILE
                # ==========================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(file_bytes)
                    temp_path = temp_file.name

                # ==========================================
                # EXTRACT PAGES
                # ==========================================

                progress.progress(
                    index / len(uploads),
                    text=f"Extracting {upload.name}..."
                )

                pages = extract_text_from_pdf(
                    temp_path
                )

                if not pages:

                    failed_count += 1

                    st.warning(
                        f"⚠️ Could not extract text from "
                        f"{upload.name}."
                    )

                    continue

                # ==========================================
                # CHUNK PDF
                # ==========================================

                chunks = chunk_pages(pages)

                if not chunks:

                    failed_count += 1

                    st.warning(
                        f"⚠️ No text chunks found in "
                        f"{upload.name}."
                    )

                    continue

                st.write(
                    f"📄 {upload.name}: "
                    f"{len(pages)} pages → "
                    f"{len(chunks)} chunks"
                )

                # ==========================================
                # CREATE EMBEDDINGS
                # ==========================================

                progress.progress(
                    index / len(uploads),
                    text=f"Creating embeddings for {upload.name}..."
                )

                vectors = embedding_model.encode(
                    [
                        str(chunk["text"])
                        for chunk in chunks
                    ]
                )

                # ==========================================
                # SAFETY CHECK
                # ==========================================

                if not vectors:

                    failed_count += 1

                    st.warning(
                        f"⚠️ No embeddings generated for "
                        f"{upload.name}."
                    )

                    continue

                if len(vectors[0]) != 384:

                    failed_count += 1

                    st.error(
                        f"❌ Wrong embedding dimension for "
                        f"{upload.name}: "
                        f"{len(vectors[0])}. "
                        f"Expected 384."
                    )

                    continue

                # ==========================================
                # ADD DOCUMENT
                # ==========================================

                document_id = database.add_document(
                    filename=upload.name,
                    file_hash=file_hash
                )

                # ==========================================
                # ADD CHUNKS
                # ==========================================

                stored = database.add_chunks(
                    chunks=chunks,
                    embeddings=vectors,
                    document_id=document_id,
                    document_name=upload.name
                )

                total_chunks += stored
                indexed_count += 1

                progress.progress(
                    (index + 1) / len(uploads),
                    text=f"Indexed {upload.name}"
                )

                st.success(
                    f"✅ {upload.name}: "
                    f"{stored} chunks indexed"
                )

            except Exception as e:

                failed_count += 1

                st.error(
                    f"❌ Error processing "
                    f"{upload.name}: {e}"
                )

        # ==========================================
        # FINISH
        # ==========================================

        progress.progress(
            1.0,
            text="Indexing complete"
        )

        st.success(
            f"Finished indexing: "
            f"{indexed_count} new document(s), "
            f"{skipped_count} already indexed, "
            f"{failed_count} failed, "
            f"{total_chunks} new chunks."
        )


# ==========================================
# MAIN CHAT
# ==========================================

st.subheader("💬 Ask your documents")

question = st.chat_input(
    "Ask something about your uploaded PDFs..."
)


# ==========================================
# QUESTION PROCESSING
# ==========================================

if question:

    # ==========================================
    # USER MESSAGE
    # ==========================================

    with st.chat_message("user"):
        st.write(question)

    # ==========================================
    # ASSISTANT MESSAGE
    # ==========================================

    with st.chat_message("assistant"):

        matches = []

        try:

            # ------------------------------------------
            # RETRIEVE
            # ------------------------------------------

            with st.spinner(
                "🔎 Searching your documents..."
            ):

                matches = retriever.retrieve(
                    question,
                    n_results=5
                )

            # ------------------------------------------
            # SHOW RETRIEVAL DEBUG
            # ------------------------------------------

            if matches:

                st.caption(
                    f"Retrieved {len(matches)} relevant chunks."
                )

                # ------------------------------------------
                # GENERATE ANSWER
                # ------------------------------------------

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    answer = generate_answer(
                        question,
                        matches
                    )

                st.markdown(answer)

            else:

                st.warning(
                    "I couldn't find relevant information "
                    "in the indexed documents."
                )

                st.caption(
                    "The PDF may be indexed, but the "
                    "retriever returned no matching chunks."
                )

        except Exception as e:

            st.error(
                f"❌ Error while answering: {e}"
            )


    # ==========================================
    # SOURCES
    # ==========================================

    if matches:

        st.subheader("📌 Sources")

        for match in matches:

            source = match.get(
                "source",
                "Unknown document"
            )

            page = match.get(
                "page",
                "Unknown"
            )

            similarity = match.get(
                "similarity",
                0.0
            )

            with st.expander(
                f"{source} · Page {page}"
            ):

                st.write(
                    match.get(
                        "text",
                        ""
                    )
                )

                st.caption(
                    f"Similarity: {float(similarity):.4f}"
                )


# ==========================================
# STARTUP MESSAGE
# ==========================================

else:

    st.info(
        "📚 Upload and index your PDFs from the "
        "sidebar, then ask a question."
    )