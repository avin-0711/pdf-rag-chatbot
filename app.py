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
    "Upload a PDF, index it, and ask questions about that PDF."
)


# ==========================================
# SESSION STATE
# ==========================================

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None

if "active_document_name" not in st.session_state:
    st.session_state.active_document_name = None


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def get_database():
    return VectorDatabase()


@st.cache_resource
def get_retriever():
    return Retriever()


embedding_model = get_embedding_model()
database = get_database()
retriever = get_retriever()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("📚 Documents")

    uploads = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False
    )

    index_document = st.button(
        "Index PDF",
        type="primary",
        disabled=not uploads
    )

    # ======================================
    # INDEX DOCUMENT
    # ======================================

    if index_document and uploads:

        upload = uploads

        progress = st.progress(
            0,
            text="Preparing PDF..."
        )

        try:

            # ==================================
            # Read PDF bytes
            # ==================================

            file_bytes = upload.getvalue()

            file_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            progress.progress(
                10,
                text="Checking document..."
            )

            # ==================================
            # Check whether PDF already exists
            # ==================================

            existing = (
                database.client
                .table("documents")
                .select("id, filename")
                .eq("file_hash", file_hash)
                .execute()
            )

            if existing.data:

                document_id = existing.data[0]["id"]

                document_name = existing.data[0]["filename"]

                # Make this document active
                st.session_state.active_document_id = document_id
                st.session_state.active_document_name = document_name

                progress.progress(
                    100,
                    text="PDF selected."
                )

                st.success(
                    f"📄 {document_name} is already indexed."
                )

                st.info(
                    "This PDF is now the active document. "
                    "Questions will only search this PDF."
                )

            else:

                # ==================================
                # Save temporary PDF
                # ==================================

                progress.progress(
                    20,
                    text="Reading PDF..."
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(file_bytes)

                    temp_path = temp_file.name

                # ==================================
                # Extract text
                # ==================================

                pages = extract_text_from_pdf(
                    temp_path
                )

                if not pages:

                    st.error(
                        f"Could not extract text from {upload.name}."
                    )

                    progress.empty()

                else:

                    progress.progress(
                        40,
                        text="Creating chunks..."
                    )

                    # ==================================
                    # Create chunks
                    # ==================================

                    chunks = chunk_pages(
                        pages
                    )

                    if not chunks:

                        st.error(
                            f"No text chunks found in {upload.name}."
                        )

                        progress.empty()

                    else:

                        progress.progress(
                            60,
                            text="Creating embeddings..."
                        )

                        # ==================================
                        # Generate embeddings
                        # ==================================

                        vectors = embedding_model.encode(
                            [
                                str(chunk["text"])
                                for chunk in chunks
                            ]
                        )

                        progress.progress(
                            75,
                            text="Saving document..."
                        )

                        # ==================================
                        # Add document
                        # ==================================

                        document_id = database.add_document(
                            filename=upload.name,
                            file_hash=file_hash
                        )

                        # ==================================
                        # Add chunks
                        # ==================================

                        stored = database.add_chunks(
                            chunks=chunks,
                            embeddings=vectors,
                            document_id=document_id,
                            document_name=upload.name
                        )

                        # ==================================
                        # Make newly indexed PDF ACTIVE
                        # ==================================

                        st.session_state.active_document_id = (
                            document_id
                        )

                        st.session_state.active_document_name = (
                            upload.name
                        )

                        progress.progress(
                            100,
                            text="Indexing complete!"
                        )

                        st.success(
                            f"✅ {upload.name}"
                        )

                        st.write(
                            f"Added {stored} chunks."
                        )

                        st.info(
                            "This PDF is now the active document. "
                            "Questions will only search this PDF."
                        )

                        progress.empty()

        except Exception as e:

            progress.empty()

            st.error(
                f"Error processing {upload.name}: {e}"
            )


    # ==========================================
    # ACTIVE DOCUMENT
    # ==========================================

    st.divider()

    st.subheader("📄 Active Document")

    if st.session_state.active_document_id:

        st.success(
            st.session_state.active_document_name
        )

        st.caption(
            "Only this PDF will be searched."
        )

    else:

        st.info(
            "No PDF selected."
        )


# ==========================================
# CHAT INTERFACE
# ==========================================

st.subheader("💬 Ask your document")

if not st.session_state.active_document_id:

    st.info(
        "Upload and index a PDF from the sidebar "
        "before asking a question."
    )

else:

    st.caption(
        f"Searching: "
        f"**{st.session_state.active_document_name}**"
    )


question = st.chat_input(
    "Ask something about your PDF..."
)


# ==========================================
# ANSWER QUESTION
# ==========================================

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        matches = []

        if not st.session_state.active_document_id:

            st.warning(
                "Please upload and index a PDF first."
            )

        else:

            try:

                with st.spinner(
                    "Searching the active PDF..."
                ):

                    matches = retriever.retrieve(
                        question=question,
                        document_id=(
                            st.session_state.active_document_id
                        ),
                        n_results=5
                    )

                if matches:

                    answer = generate_answer(
                        question,
                        matches
                    )

                    st.markdown(answer)

                else:

                    st.info(
                        "I couldn't find relevant information "
                        "in the active PDF."
                    )

            except Exception as e:

                st.error(
                    f"Error while answering: {e}"
                )


    # ==========================================
    # SOURCES
    # ==========================================

    if matches:

        st.subheader("📌 Sources")

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