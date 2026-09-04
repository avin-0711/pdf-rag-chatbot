import hashlib
import tempfile
from pathlib import Path

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
# CONFIGURATION
# ==========================================

MAX_PDFS = 50


st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")

st.caption(
    "Upload up to 50 PDF files and ask questions "
    "across their combined contents."
)


# ==========================================
# SESSION STATE
# ==========================================

if "active_document_ids" not in st.session_state:
    st.session_state.active_document_ids = []

if "active_document_names" not in st.session_state:
    st.session_state.active_document_names = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# LOAD RESOURCES
# ==========================================

@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def get_database():
    return VectorDatabase()


@st.cache_resource
def get_retriever():
    return Retriever(
        database=database,
        embedding_model=embedding_model,
    )


embedding_model = get_embedding_model()
database = get_database()
retriever = get_retriever()


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

    # ==========================================
    # 50 PDF LIMIT
    # ==========================================

    if uploads and len(uploads) > MAX_PDFS:

        st.error(
            f"You selected {len(uploads)} PDFs. "
            f"The maximum allowed is {MAX_PDFS}."
        )

        uploads = uploads[:MAX_PDFS]

        st.warning(
            f"Only the first {MAX_PDFS} PDFs will be used."
        )

    # ==========================================
    # INDEX BUTTON
    # ==========================================

    index_documents = st.button(
        "Index PDFs",
        type="primary",
        disabled=not uploads
    )

    # ==========================================
    # INDEX DOCUMENTS
    # ==========================================

    if index_documents and uploads:

        total_files = len(uploads)

        overall_progress = st.progress(
            0,
            text="Starting indexing..."
        )

        indexed_ids = []
        indexed_names = []

        total_chunks = 0

        successful_files = 0
        skipped_files = 0
        failed_files = 0

        for file_index, upload in enumerate(uploads):

            try:

                # ==================================
                # Progress
                # ==================================

                overall_progress.progress(
                    file_index / total_files,
                    text=(
                        f"Processing "
                        f"{file_index + 1}/{total_files}: "
                        f"{upload.name}"
                    )
                )

                # ==================================
                # Read file
                # ==================================

                file_bytes = upload.getvalue()

                file_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()

                # ==================================
                # Check if already indexed
                # ==================================

                existing = (
                    database.client
                    .table("documents")
                    .select(
                        "id, filename"
                    )
                    .eq(
                        "file_hash",
                        file_hash
                    )
                    .execute()
                )

                if existing.data:

                    document_id = existing.data[0]["id"]
                    document_name = existing.data[0]["filename"]

                    indexed_ids.append(
                        document_id
                    )

                    indexed_names.append(
                        document_name
                    )

                    skipped_files += 1

                    st.info(
                        f"⏭️ {document_name} "
                        f"is already indexed."
                    )

                    continue

                # ==================================
                # Save temporary PDF
                # ==================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        file_bytes
                    )

                    temp_path = temp_file.name

                # ==================================
                # Extract PDF text
                # ==================================

                try:
                    pages = extract_text_from_pdf(
                        temp_path
                    )
                finally:
                    Path(temp_path).unlink(
                        missing_ok=True
                    )

                if not pages:

                    failed_files += 1

                    st.warning(
                        f"⚠️ Could not extract text "
                        f"from {upload.name}."
                    )

                    continue

                # ==================================
                # Create chunks
                # ==================================

                chunks = chunk_pages(
                    pages
                )

                if not chunks:

                    failed_files += 1

                    st.warning(
                        f"⚠️ No text chunks found "
                        f"in {upload.name}."
                    )

                    continue

                # ==================================
                # Generate embeddings
                # ==================================

                vectors = embedding_model.encode(
                    [
                        str(chunk["text"])
                        for chunk in chunks
                    ]
                )

                # ==================================
                # Create document
                # ==================================

                document_id = database.add_document(
                    filename=upload.name,
                    file_hash=file_hash
                )

                # ==================================
                # Store chunks
                # ==================================

                stored = database.add_chunks(
                    chunks=chunks,
                    embeddings=vectors,
                    document_id=document_id,
                    document_name=upload.name
                )

                # ==================================
                # Add to active knowledge base
                # ==================================

                indexed_ids.append(
                    document_id
                )

                indexed_names.append(
                    upload.name
                )

                total_chunks += stored
                successful_files += 1

                st.success(
                    f"✅ {upload.name} — "
                    f"{stored} chunks"
                )

            except Exception as e:

                failed_files += 1

                st.error(
                    f"❌ Error processing "
                    f"{upload.name}: {e}"
                )

        # ==========================================
        # Make this batch ACTIVE
        # ==========================================

        st.session_state.active_document_ids = (
            indexed_ids
        )

        st.session_state.active_document_names = (
            indexed_names
        )

        overall_progress.progress(
            1.0,
            text="Indexing complete!"
        )

        # ==========================================
        # Summary
        # ==========================================

        st.divider()

        st.success(
            f"Finished processing {total_files} PDFs."
        )

        st.write(
            f"📄 Active PDFs: {len(indexed_ids)}"
        )

        st.write(
            f"🧩 New chunks: {total_chunks}"
        )

        if skipped_files:
            st.write(
                f"⏭️ Already indexed: {skipped_files}"
            )

        if failed_files:
            st.write(
                f"❌ Failed: {failed_files}"
            )

        st.info(
            "All successfully indexed PDFs in this "
            "batch are now part of the active knowledge base."
        )


    # ==========================================
    # ACTIVE DOCUMENTS
    # ==========================================

    st.divider()

    st.subheader("📄 Active PDFs")

    active_names = (
        st.session_state.active_document_names
    )

    if active_names:

        st.success(
            f"{len(active_names)} PDF"
            f"{'s' if len(active_names) != 1 else ''} active"
        )

        for name in active_names:

            st.caption(
                f"• {name}"
            )

        st.caption(
            "Questions will search only these PDFs."
        )

    else:

        st.info(
            "No PDFs indexed yet."
        )


# ==========================================
# MAIN CHAT AREA
# ==========================================

st.subheader("💬 Ask your documents")


active_ids = (
    st.session_state.active_document_ids
)

active_names = (
    st.session_state.active_document_names
)


if active_ids:

    st.caption(
        f"🔎 Searching across "
        f"**{len(active_ids)} active PDF"
        f"{'s' if len(active_ids) != 1 else ''}**"
    )

else:

    st.info(
        "Upload and index PDF files from the sidebar "
        "before asking questions."
    )


# ==========================================
# QUESTION
# ==========================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask something about your PDFs..."
)


# ==========================================
# ANSWER
# ==========================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):

        st.write(question)

    with st.chat_message("assistant"):

        matches = []

        if not active_ids:

            answer = (
                "Please upload and index at least "
                "one PDF first."
            )
            st.warning(answer)

        else:

            try:

                with st.spinner(
                    "Searching your PDFs..."
                ):

                    previous_questions = [
                        message["content"]
                        for message in st.session_state.messages[:-1]
                        if message["role"] == "user"
                    ][-2:]
                    retrieval_question = "\n".join(
                        previous_questions + [question]
                    )
                    matches = retriever.retrieve(
                        question=retrieval_question,
                        document_ids=active_ids,
                        n_results=5
                    )

                if matches:

                    answer = generate_answer(
                        question=question,
                        contexts=matches,
                        history=st.session_state.messages,
                    )

                    st.markdown(answer)

                else:

                    answer = (
                        "I couldn't find relevant information "
                        "in the active PDFs."
                    )
                    st.info(answer)

            except Exception as e:

                answer = "I could not answer that question because an internal error occurred."
                st.error(f"{answer} {e}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })


        # ======================================
        # SOURCES
        # ======================================

        if matches:

            st.subheader(
                "📌 Sources"
            )

            for match in matches:

                with st.expander(
                    f"{match['source']} · "
                    f"Page {match['page']}"
                ):

                    st.write(
                        match["text"]
                    )

                    st.caption(
                        f"Similarity: "
                        f"{match['similarity']:.4f}"
                    )