
from ollama import embed
import math


MODEL = "nomic-embed-text"


def get_embedding(text: str) -> list[float]:
    response = embed(
        model=MODEL,
        input=text,
    )
    return response["embeddings"][0]


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


texts = [
    "The application is written in Python.",
    "Python is the programming language used by the application.",
    "I enjoy eating mangoes in summer.",
]

embeddings = [get_embedding(text) for text in texts]

for i in range(1, len(texts)):
    score = cosine_similarity(embeddings[0], embeddings[i])

    print(f"\nText A: {texts[0]}")
    print(f"Text B: {texts[i]}")
    print(f"Cosine similarity: {score:.4f}")
