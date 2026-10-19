
from app.tools import chunk_text


text = (
    "Python is used for the application backend. "
    "Ollama runs the Qwen3 8B model locally. "
    "The calculator performs arithmetic operations. "
    "The notes tool saves information to a text file. "
    "Semantic search retrieves relevant document passages."
)

chunks = chunk_text(
    text,
    max_words=10,
    overlap_sentences=1,
)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)
    print("Word count:", len(chunk.split()))
