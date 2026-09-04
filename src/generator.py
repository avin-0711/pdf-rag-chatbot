import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_answer(
    question: str,
    contexts: list[dict],
    model: str = "gemini-3.6-flash",
) -> str:
    """
    Generate a grounded answer using only retrieved PDF content.
    """

    if not contexts:
        return (
            "I couldn't find relevant information in the uploaded "
            "documents to answer this question."
        )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "GEMINI_API_KEY is missing from the .env file."
        )

    # Build the context sent to Gemini
    context_parts = []

    for item in contexts:
        context_parts.append(
            f"[Source: {item['source']}, Page: {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document-grounded RAG assistant.

Answer the user's question using ONLY the information contained
in the provided document context.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the context does not contain enough information to answer,
   clearly say that the information is not available in the
   uploaded documents.
4. Keep the answer concise and useful.
5. At the end of relevant statements, include citations using:
   [filename, Page X]
6. Only cite pages that actually support the statement.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        answer = response.text

        if not answer:
            return "No answer was returned by Gemini."

        return answer.strip()

    except Exception as e:
        return f"Gemini error: {e}"