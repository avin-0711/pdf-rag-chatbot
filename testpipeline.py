from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embeddings import EmbeddingModel
from src.database import VectorDatabase


PDF_PATH = "pdfs/test.pdf"


print("1. Extracting PDF...")
pages = extract_text_from_pdf(PDF_PATH)
print(f"   Pages extracted: {len(pages)}")


print("2. Creating chunks...")
chunks = chunk_pages(pages)
print(f"   Chunks created: {len(chunks)}")


print("3. Creating embeddings...")
embedding_model = EmbeddingModel()

texts = [chunk["text"] for chunk in chunks]
embeddings = embedding_model.encode(texts)

print(f"   Embeddings created: {len(embeddings)}")
print(f"   Dimensions: {len(embeddings[0])}")


print("4. Connecting to Supabase...")
db = VectorDatabase()

document_id = db.add_document(
    filename="test.pdf",
    file_hash="pipeline_test_123"
)

print(f"   Document ID: {document_id}")


print("5. Saving chunks to Supabase...")

stored = db.add_chunks(
    chunks=chunks,
    embeddings=embeddings,
    document_id=document_id,
    document_name="test.pdf"
)

print(f"   Chunks stored: {stored}")

print("\n✅ COMPLETE PIPELINE SUCCESSFUL!")