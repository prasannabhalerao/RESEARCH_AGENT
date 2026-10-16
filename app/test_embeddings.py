
from ollama import embed


MODEL = "nomic-embed-text"

text = "The application is written in Python."

response = embed(
    model=MODEL,
    input=text,
)

vector = response["embeddings"][0]

print("Original text:")
print(text)

print("\nNumber of values in embedding:")
print(len(vector))

print("\nFirst 10 embedding values:")
print(vector[:10])
