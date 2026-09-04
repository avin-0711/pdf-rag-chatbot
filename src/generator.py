import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()


def generate_answer(
    question: str,
    contexts: list[dict],
    history: list[dict] | None = None,
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

    for index, item in enumerate(contexts, start=1):
        context_parts.append(
            f"[S{index}] {item['source']}, Page {item['page']}\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)
    conversation = "\n".join(
        f"{item['role']}: {item['content']}"
        for item in (history or [])[-6:]
    ) or "No previous conversation."

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
5. Cite every factual statement with one or more valid source labels such as [S1].
6. Use only source labels that appear in the provided context.
7. If no source supports the answer, say that the information is not available.

DOCUMENT CONTEXT:
{context}

RECENT CONVERSATION:
{conversation}

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

        answer = answer.strip()
        allowed = {f"S{index}" for index in range(1, len(contexts) + 1)}
        cited = set(re.findall(r"\[(S\d+)\]", answer))
        answer = re.sub(
            r"\[(S\d+)\]",
            lambda match: match.group(0)
            if match.group(1) in allowed
            else "",
            answer,
        ).strip()
        if not cited & allowed:
            answer += "\n\nSources: " + ", ".join(
                f"[{f'S{index}'}] {item['source']}, page {item['page']}"
                for index, item in enumerate(contexts, start=1)
            )
        return answer

    except Exception as e:
        return f"Gemini error: {e}"