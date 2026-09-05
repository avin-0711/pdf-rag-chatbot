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
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #20333a;
        --muted: #65757a;
        --paper: #fbfaf7;
        --line: #dfe7e4;
        --sage: #477568;
        --coral: #c56b52;
    }
    .stApp { background: var(--paper); }
    [data-testid="stSidebar"] { background: #f1f5f2; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    .app-kicker { color: var(--coral); font-size: .74rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .app-title { color: var(--ink); font-size: clamp(2.2rem, 4vw, 4.4rem); line-height: .98; letter-spacing: -.045em; font-weight: 750; margin: .45rem 0 .8rem; }
    .app-deck { color: var(--muted); font-size: 1rem; max-width: 42rem; line-height: 1.55; }
    .metric-strip { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: .8rem 0; margin: 1.5rem 0 2.1rem; }
    .source-label { color: var(--sage); font-weight: 650; }
    [data-testid="stChatMessage"] { border-bottom: 1px solid rgba(223, 231, 228, .75); padding-bottom: 1.25rem; }
    div[data-testid="stFileUploader"] { border: 1px dashed #a9beb7; border-radius: 10px; background: rgba(255,255,255,.52); padding: .35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-kicker">Private reading room</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Ask your documents<br><span style="color:#477568">with receipts.</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-deck">Bring your PDFs together, search their meaning, and get concise answers tied back to the exact document and page.</div>',
    unsafe_allow_html=True,
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

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

theme_mode = st.session_state.theme_mode


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

    st.header("Your library", icon=":material/folder_open:")
    st.caption("Up to 50 PDFs per indexing session")

    theme_mode = st.segmented_control(
        "Appearance",
        options=["Light", "Dark"],
        default=st.session_state.theme_mode,
        key="theme_mode",
        format_func=lambda value: f":material/{'light_mode' if value == 'Light' else 'dark_mode'}: {value}",
    )

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
        icon=":material/bolt:",
        disabled=not uploads,
        width="stretch",
    )

    if st.button(
        "Clear conversation",
        icon=":material/refresh:",
        width="stretch",
        disabled=not st.session_state.messages,
    ):
        st.session_state.messages = []
        st.rerun()

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

                document_id = None

                if existing.data:

                    existing_document_id = existing.data[0]["id"]
                    document_name = existing.data[0]["filename"]
                    existing_chunks = (
                        database.client
                        .table("chunks")
                        .select("id")
                        .eq("document_id", existing_document_id)
                        .limit(1)
                        .execute()
                    )

                    if existing_chunks.data:
                        document_id = existing_document_id
                        indexed_ids.append(document_id)
                        indexed_names.append(document_name)
                        skipped_files += 1
                        st.info(
                            f"⏭️ {document_name} "
                            f"is already indexed."
                        )
                        continue

                    st.warning(
                        f"Reprocessing {document_name}; its previous index had no chunks."
                    )

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

                if document_id is None:
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

    st.subheader("Active PDFs", icon=":material/library_books:")

    active_names = (
        st.session_state.active_document_names
    )

    if active_names:

        st.badge(f"{len(active_names)} active", icon=":material/check:", color="green")

        for name in active_names:

            st.markdown(f":material/description: **{name}**")

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

if st.session_state.theme_mode == "Dark":
    st.markdown(
        """
        <style>
        :root {
            --ink: #edf3f0;
            --muted: #a9bbb5;
            --paper: #162321;
            --line: #30443f;
            --sage: #8bc0ad;
            --coral: #e89a7e;
        }
        body, .stApp, [data-testid="stAppViewContainer"] { background: #162321 !important; }
        [data-testid="stHeader"] { background: #162321 !important; }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: #101a19 !important; border-right-color: #30443f !important; }
        [data-testid="stSidebar"] * { color: #edf3f0; }
        .app-title, .app-deck, .metric-strip, [data-testid="stChatMessage"] { color: var(--ink); }
        .app-deck, [data-testid="stCaptionContainer"] { color: var(--muted); }
        div[data-testid="stFileUploader"] { background: rgba(255,255,255,.04); border-color: #55766c; }
        [data-testid="stFileUploaderDropzone"] { background: #1c2c29 !important; border-color: #55766c !important; }
        [data-testid="stAlert"] { background: #20332f !important; border-color: #55766c !important; color: #edf3f0 !important; }
        [data-testid="stAlertContainer"], [data-testid="stAlertContainer"] * { background: #20332f !important; color: #edf3f0 !important; }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: #edf3f0 !important; }
        section[data-testid="stAppScrollToBottomContainer"] > div,
        [data-testid="stBottom"], [data-testid="stBottom"] > div,
        [data-testid="stBottom"] > div > div { background: #162321 !important; }
        [data-testid="stChatInput"], [data-testid="stChatInput"] > div { background: #20332f !important; border-color: #55766c !important; }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input { color: #edf3f0 !important; caret-color: #e89a7e !important; }
        [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stChatInput"] input::placeholder { color: #a9bbb5 !important; opacity: 1 !important; }
        [data-testid="stChatInput"] button { color: #edf3f0 !important; }
        [data-testid="stAppViewContainer"] button:not([kind="primary"]) { color: #edf3f0; background: #20332f; border-color: #55766c; }
        </style>
        """,
        unsafe_allow_html=True,
    )

active_ids = st.session_state.active_document_ids
active_names = st.session_state.active_document_names

st.markdown('<div class="metric-strip">', unsafe_allow_html=True)
metric_a, metric_b, metric_c = st.columns(3)
with metric_a:
    st.metric("Active PDFs", len(active_ids))
with metric_b:
    st.metric("Conversation turns", len(st.session_state.messages) // 2)
with metric_c:
    st.metric("Retrieval mode", "Top-K semantic")
st.markdown('</div>', unsafe_allow_html=True)

st.subheader("A clear answer starts with a clear question", icon=":material/chat:")


if active_ids:

    st.caption(
        f":material/search: Searching across "
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

if not st.session_state.messages:
    suggestions = {
        ":material/summarize: Give me a concise summary": "Give me a concise summary of the uploaded documents.",
        ":material/compare_arrows: Compare the documents": "Compare the main themes across the uploaded documents.",
        ":material/fact_check: Find a supporting detail": "What is the most important supporting detail in the documents?",
    }
    suggestion = st.pills(
        "Start with a prompt",
        list(suggestions),
        label_visibility="collapsed",
    )
    suggested_question = suggestions.get(suggestion)
else:
    suggested_question = None
# ==========================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask something about your PDFs...",
    submit_mode="disable",
)
question = question or suggested_question


# ==========================================
# ANSWER
# ==========================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user", avatar=":material/person:"):

        st.write(question)

    with st.chat_message("assistant", avatar=":material/menu_book:"):

        matches = []

        if not active_ids:

            answer = (
                "Please upload and index at least "
                "one PDF first."
            )
            st.warning(answer)

        else:

            try:

                with st.status(":shimmer[Searching your PDFs]", type="compact") as search_status:

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
                        document_ids=database.searchable_document_ids(
                            active_ids,
                            active_names,
                        ),
                        n_results=5
                    )
                    search_status.update(label="Sources found", state="complete")

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