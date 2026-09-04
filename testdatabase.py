from src.database import VectorDatabase

db = VectorDatabase()

print("Supabase connection successful!")

document_id = db.add_document(
    filename="test.pdf",
    file_hash="test_hash_123"
)

print("Document inserted!")
print("Document ID:", document_id)