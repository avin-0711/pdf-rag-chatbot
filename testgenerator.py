from src.generator import generate_answer

contexts = [
    {
        "source": "test.pdf",
        "page": 3,
        "text": """
        The National Testing Agency (NTA) was established as an
        autonomous and self-sustained testing organization to conduct
        entrance examinations for higher educational institutions.
        """
    }
]

question = "What is the National Testing Agency?"

answer = generate_answer(
    question,
    contexts
)

print("\nANSWER:")
print(answer)