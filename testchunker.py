from src.pdf_processor import extract_text_from_pdf
from src.chunker import chunk_pages

pages = extract_text_from_pdf("pdfs/test.pdf")

chunks = chunk_pages(pages)

print("Pages extracted:", len(pages))
print("Chunks created:", len(chunks))

for i, chunk in enumerate(chunks[:3], start=1):
    print("\n" + "=" * 50)
    print("CHUNK:", i)
    print("FILE:", chunk["filename"])
    print("PAGE:", chunk["page_number"])
    print("WORDS:", len(chunk["text"].split()))
    print("TEXT:", chunk["text"][:300])