# 3-5 Minute Demonstration Script

## 0:00-0:30 - Introduce the app

Show the Streamlit home screen and explain that the app supports up to 50 PDFs, page-aware extraction, semantic retrieval, grounded Gemini answers, and source citations.

## 0:30-1:15 - Upload documents

Upload two text-based PDFs. Point out the active document list and the indexing progress indicator. Explain that SHA-256 hashes prevent the same PDF from being processed twice.

## 1:15-2:00 - Indexing pipeline

Click **Index PDFs**. Explain the sequence: PyMuPDF extraction, overlapping chunk creation, 384-dimensional embeddings, and Supabase storage in the `documents` and `chunks` tables.

## 2:00-3:00 - Cross-document questions

Ask a question whose answer appears in the first document and show the answer with filename/page sources. Ask a comparison question involving both PDFs and show that the active document set is searched together.

## 3:00-3:45 - Follow-up question

Ask a follow-up that uses context from the previous question. Show that the conversation remains visible and that recent history is included in retrieval and generation.

## 3:45-4:30 - Grounding and no-answer behavior

Ask a question unrelated to the uploaded documents. Show that the app reports that no relevant information is available instead of inventing an answer. Expand the Sources panels to show retrieved filename, page number, and similarity.

## 4:30-5:00 - Deployment and architecture

Briefly show the GitHub repository, `architecture.md`, `supabase/schema.sql`, and `evaluation.md`. Mention that Streamlit Community Cloud requires the three secrets configured in the app settings.
