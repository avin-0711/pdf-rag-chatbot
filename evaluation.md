# Testing and Evaluation

## Automated checks

Run the offline test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover:

- PDF page extraction and filename/page metadata.
- Chunk size, overlap, and metadata preservation.
- Empty-context behavior.
- Removal of invalid citation labels.
- No-answer behavior when retrieval returns no chunks.

Database, embedding, and Gemini integration tests require configured services and are kept as manual smoke scripts in the repository root.

## Local results

| Check | Result |
| --- | --- |
| Python compilation for `app.py` and `src/` | Passed |
| Offline pytest suite | 5 passed |
| Test PDF extraction | Passed for text-based PDFs; supplied malformed PDF has compressed-stream errors |
| Chunker behavior | Passed |
| Dependency import check | Passed |
| Streamlit clean startup and home screen | Passed |
| Supabase RPC search | Requires applying `supabase/schema.sql` and valid credentials |
| Gemini answer generation | Requires `GEMINI_API_KEY` |
| Streamlit end-to-end workflow | Requires Supabase schema, credentials, and a running app |

## Evaluation dataset

For a formal RAG evaluation, record questions with expected supporting filename/page values, then measure:

- Retrieval hit rate@K: expected source appears in the Top-K results.
- Citation precision: displayed citations refer to retrieved source chunks.
- Answer faithfulness: every factual claim is supported by a retrieved chunk.
- No-answer accuracy: unsupported questions are declined instead of answered from outside knowledge.
- P50/P95 indexing and query latency.

The application should be evaluated with both single-document questions and cross-document comparison questions.
