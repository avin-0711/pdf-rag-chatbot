# PDF RAG Chatbot

A Streamlit application that lets you upload multiple PDFs, extract their text, split the text into overlapping chunks, create sentence embeddings, store the indexed content in Supabase, and ask grounded questions using Gemini.

## Features

- Upload and index up to 50 PDFs at a time.
- Extract text page by page with PyMuPDF.
- Split pages into overlapping word chunks.
- Generate embeddings with `all-MiniLM-L6-v2`.
- Store documents, chunks, and embeddings in Supabase.
- Avoid indexing the same PDF twice by checking its SHA-256 hash.
- Search only the PDFs in the active indexing batch.
- Generate answers from retrieved context with Gemini.
- Show source filename and page citations in responses.

## Project Structure

```text
pdf-rag-chatbot/
├── app.py
├── requirements.txt
├── .env.example
├── architecture.md
├── docs/
│   ├── demo-script.md
│   └── screenshots/
│       └── app-home.png
├── evaluation.md
├── supabase/
│   └── schema.sql
├── pdfs/
│   └── test.pdf
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── database.py
│   ├── retriever.py
│   └── generator.py
└── test*.py
└── tests/
	└── test_rag_components.py
```

## Requirements

- Python 3.10 or newer
- A Supabase project with the required tables and vector data
- A Gemini API key
- Internet access on first run so the sentence-transformers model can be downloaded

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root. Start from `.env.example`:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key
GEMINI_API_KEY=your-gemini-api-key
```

Never commit `.env` or paste live credentials into source code. For Streamlit Community Cloud, add the same values under **App settings -> Secrets** using TOML syntax:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-supabase-key"
GEMINI_API_KEY = "your-gemini-api-key"
```

## Supabase Schema

The application expects these tables:

### `documents`

| Column | Type | Purpose |
| --- | --- | --- |
| `id` | `bigint` or `int` | Primary key |
| `filename` | `text` | Uploaded PDF name |
| `file_hash` | `text` | SHA-256 hash used for deduplication |

### `chunks`

| Column | Type | Purpose |
| --- | --- | --- |
| `id` | `bigint` or `int` | Primary key |
| `document_id` | `bigint` or `int` | References `documents.id` |
| `filename` | `text` | PDF name |
| `page_number` | `int` | Source page |
| `chunk_index` | `int` | Chunk order within the PDF |
| `content` | `text` | Chunk text |
| `embedding` | `vector(384)` | `all-MiniLM-L6-v2` embedding |

The complete table, RPC, and HNSW index setup is in `supabase/schema.sql`. Run that file in the Supabase SQL editor before using the app.

The RPC performs Top-K cosine retrieval inside Postgres. The application also rejects results below a similarity threshold so unsupported questions can receive a no-answer response.

## Run Locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If the default Streamlit port is busy, choose another one:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8503
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## Use the App

1. Open the Streamlit app.
2. Upload one or more text-based PDFs in the sidebar.
3. Click **Index PDFs**.
4. Wait for extraction, chunking, embedding, and database storage to finish.
5. Ask a question in the chat input.
6. Review the answer and the cited source pages.

Scanned PDFs without a text layer may produce no chunks. OCR them first if the PDF contains only images.

## Test the PDF Reader

The supplied test document is at `pdfs/test.pdf`. Run:

```powershell
.\.venv\Scripts\python.exe -c "from src.pdf_processor import extract_text_from_pdf; pages = extract_text_from_pdf('pdfs/test.pdf'); print('Pages:', len(pages)); print(pages[:1])"
```

The command should print at least one extracted page for a text-based PDF.

## Smoke Tests

Run the offline automated tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The repository includes standalone scripts for individual components. Run them from the project root with the virtual-environment interpreter:

```powershell
python testchunker.py
python testdatabase.py
python testembeddings.py
python testgenerator.py
python testpipeline.py
python testretriever.py
python test_rpc.py
```

Some scripts require valid Supabase or Gemini credentials and may perform external API calls. The most reliable offline checks are the PDF reader and chunker tests.

## Deploy to Streamlit Community Cloud

1. Push the project to GitHub.
2. In Streamlit Community Cloud, select **Deploy an app**.
3. Choose repository `avin-0711/pdf-rag-chatbot`.
4. Select branch `main`.
5. Set the main file to `app.py`.
6. Add `SUPABASE_URL`, `SUPABASE_KEY`, and `GEMINI_API_KEY` in the app secrets.
7. Deploy or reboot the app after saving secrets.

The repository URL is https://github.com/avin-0711/pdf-rag-chatbot.

## Evidence

- [Architecture diagram](architecture.md)
- [Evaluation and testing results](evaluation.md)
- [Application screenshot](docs/screenshots/app-home.png)
- [3-5 minute demonstration script](docs/demo-script.md)

## Troubleshooting

### `ModuleNotFoundError`

Make sure the selected Python interpreter is the project virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Supabase configuration errors

Check that `SUPABASE_URL` and `SUPABASE_KEY` exist in `.env` locally or in Streamlit Cloud secrets.

### Gemini configuration errors

Check that `GEMINI_API_KEY` is configured. The generator uses the Gemini model configured in `src/generator.py`.

### Port already in use

Run Streamlit with another port, for example `--server.port 8503`.

### No text extracted

The PDF may be scanned, encrypted, or malformed. Try a text-based PDF or run OCR before uploading it.

## Security Notes

- Keep `.env` private and rotate any key that has been exposed.
- Use least-privilege Supabase credentials where possible.
- Do not commit generated databases, virtual environments, model caches, or API keys.