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
# LOAD COMPONENTS
# ==========================================

@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def get_database():
    return VectorDatabase()


@st.cache_resource
def get_retriever(database, embedding_model):
    return Retriever(database, embedding_model)


embedding_model = get_embedding_model()
database = get_database()
retriever = get_retriever(
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

    index_documents = st.button(
        "Index Documents",
        type="primary",
        disabled=not uploads
    )

    if index_documents and uploads:

        progress = st.progress(
            0,
            text="Indexing documents..."
        )

        total_chunks = 0
        indexed_count = 0
        skipped_count = 0

        for index, upload in enumerate(uploads):

            try:

                # ==========================================
                # GET FILE BYTES
                # ==========================================

                file_bytes = upload.getvalue()

                # ==========================================
                # CREATE REAL FILE HASH
                # ==========================================

                file_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                # ==========================================
                # CHECK IF ALREADY INDEXED
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
                        f"⏭️ {upload.name} is already indexed."
                    )

                    progress.progress(
                        (index + 1) / len(uploads),
                        text=f"Skipped {upload.name}"
                    )

                    continue

                # ==========================================
                # SAVE TEMPORARY PDF
                # ==========================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(file_bytes)
                    temp_path = temp_file.name

                # ==========================================
                # EXTRACT TEXT
                # ==========================================

                pages = extract_text_from_pdf(
                    temp_path
                )

                if not pages:

                    st.warning(
                        f"⚠️ Could not extract text from {upload.name}."
                    )

                    progress.progress(
                        (index + 1) / len(uploads),
                        text=f"Failed: {upload.name}"
                    )

                    continue

                # ==========================================
                # CREATE CHUNKS
                # ==========================================

                chunks = chunk_pages(pages)

                if not chunks:

                    st.warning(
                        f"⚠️ No chunks found in {upload.name}."
                    )

                    progress.progress(
                        (index + 1) / len(uploads),
                        text=f"Failed: {upload.name}"
                    )

                    continue

                # ==========================================
                # GENERATE EMBEDDINGS
                # ==========================================

                with st.spinner(
                    f"Creating embeddings for {upload.name}..."
                ):

                    vectors = embedding_model.encode(
                        [
                            str(chunk["text"])
                            for chunk in chunks
                        ]
                    )

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
                    f"✅ {upload.name}: {stored} chunks added"
                )

            except Exception as e:

                st.error(
                    f"❌ Error processing {upload.name}: {e}"
                )

        progress.empty()

        st.success(
            f"Finished indexing: {indexed_count} new document(s), "
            f"{skipped_count} already indexed, "
            f"{total_chunks} new chunks."
        )


# ==========================================
# CHAT
# ==========================================

st.subheader("💬 Ask your documents")

question = st.chat_input(
    "Ask something about your uploaded PDFs..."
)


if question:

    # ==========================================
    # USER MESSAGE
    # ==========================================

    with st.chat_message("user"):
        st.write(question)

    # ==========================================
    # ASSISTANT
    # ==========================================

    with st.chat_message("assistant"):

        matches = []

        try:

            with st.spinner(
                "Searching the knowledge base..."
            ):

                matches = retriever.retrieve(
                    question,
                    n_results=5
                )

            # ==========================================
            # NO RESULTS
            # ==========================================

            if not matches:

                st.warning(
                    "I couldn't find relevant information "
                    "in the uploaded documents."
                )

            # ==========================================
            # GENERATE ANSWER
            # ==========================================

            else:

                answer = generate_answer(
                    question,
                    matches
                )

                st.markdown(answer)

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
                0
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
                    f"Similarity: {similarity:.4f}"
                )


# ==========================================
# INITIAL MESSAGE
# ==========================================

else:

    st.info(
        "Upload and index a PDF from the sidebar, "
        "then ask a question."
    )