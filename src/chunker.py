import re


def chunk_pages(
    pages: list[dict[str, str | int]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict[str, str | int]]:

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "chunk_size must be positive and overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:
        words = re.findall(r"\S+", str(page["text"]))

        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunks.append({
                "text": " ".join(words[start:end]),
                "filename": page["filename"],
                "page_number": int(page["page_number"]),
            })

            if end == len(words):
                break

            start = end - overlap

    return chunks