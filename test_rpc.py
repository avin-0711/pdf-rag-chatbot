import ast

from src.database import VectorDatabase

db = VectorDatabase()

# Get an existing embedding from Supabase
response = (
    db.client
    .table("chunks")
    .select("embedding, filename, page_number")
    .not_.is_("embedding", "null")
    .limit(1)
    .execute()
)

stored = response.data[0]

print("File:", stored["filename"])
print("Page:", stored["page_number"])

# Supabase returns pgvector as a string
embedding_list = ast.literal_eval(stored["embedding"])

print("Dimensions:", len(embedding_list))

# Convert the stored vector back into TEXT
query_embedding = "[" + ",".join(
    str(float(x)) for x in embedding_list
) + "]"

print("Vector string length:", len(query_embedding))

# Send that exact stored vector through the RPC
result = db.client.rpc(
    "match_chunks",
    {
        "query_embedding": query_embedding,
        "match_count": 5
    }
).execute()

print("RPC results:", len(result.data))
print(result.data)