from types import SimpleNamespace

from src.chunker import chunk_pages
from src.generator import generate_answer
from src.pdf_processor import extract_text_from_pdf
from src.retriever import Retriever


def test_chunking_preserves_metadata_and_overlap():
    pages = [{"filename": "report.pdf", "page_number": 2, "text": "one two three four five six"}]
    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    assert chunks[0]["filename"] == "report.pdf"
    assert chunks[0]["page_number"] == 2
    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"] == "four five six"


def test_pdf_fixture_extracts_text():
    pages = extract_text_from_pdf("pdfs/test.pdf")
    assert isinstance(pages, list)


def test_empty_context_returns_grounding_message():
    answer = generate_answer("What happened?", [])
    assert "couldn't find relevant" in answer.lower()


def test_retriever_uses_rpc_and_similarity_threshold():
    class FakeEmbeddingModel:
        def encode(self, texts):
            return [[0.1, 0.2]]

    class FakeDatabase:
        def __init__(self):
            self.call = None

        def search_chunks(self, query_embedding, document_ids, n_results):
            self.call = (query_embedding, document_ids, n_results)
            return [
                {"id": 1, "filename": "a.pdf", "page_number": 2, "content": "supported", "similarity": 0.8},
                {"id": 2, "filename": "b.pdf", "page_number": 3, "content": "weak", "similarity": 0.1},
            ]

    database = FakeDatabase()
    retriever = Retriever(database=database, embedding_model=FakeEmbeddingModel())
    results = retriever.retrieve("question", [10, 11], n_results=4)

    assert database.call == ([0.1, 0.2], [10, 11], 4)
    assert [result["source"] for result in results] == ["a.pdf"]


def test_database_recognizes_missing_rpc_error_text():
    from src.database import VectorDatabase

    class FakeRpc:
        def execute(self):
            raise RuntimeError(
                "Could not find the function public.match_chunks in the schema cache"
            )

    class FakeQuery:
        def select(self, columns):
            return self

        def in_(self, column, values):
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    class FakeClient:
        def rpc(self, name, params):
            return FakeRpc()

        def table(self, name):
            return FakeQuery()

    database = object.__new__(VectorDatabase)
    database.client = FakeClient()
    assert database.search_chunks([0.1, 0.2], [1], 5) == []


def test_invalid_citation_labels_are_removed(monkeypatch):
    class FakeClient:
        class Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(text="Supported [S1], unsupported [S99].")

        models = Models()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.generator.genai",
        SimpleNamespace(Client=lambda api_key: FakeClient()),
    )
    answer = generate_answer(
        "question",
        [{"source": "report.pdf", "page": 4, "text": "context"}],
    )

    assert "[S1]" in answer
    assert "[S99]" not in answer
