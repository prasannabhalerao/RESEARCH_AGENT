import math
from pathlib import Path
from ollama import embed

DATA_DIR = Path("data")
NOTES_FILE = DATA_DIR / "notes.txt"
DOCUMENTS_DIR = Path("data/documents")

def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic on two numbers."""
    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    raise ValueError("Unsupported operation.")


def save_note(note: str) -> str:
    """Save a text note to the local notes file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with NOTES_FILE.open("a", encoding="utf-8") as file:
        file.write(note.replace("\n", " ") + "\n")

    return "Note saved successfully."


def read_notes() -> str:
    """Read all previously saved notes."""
    if not NOTES_FILE.exists():
        return "No notes have been saved yet."

    return NOTES_FILE.read_text(encoding="utf-8")

def search_notes(query: str) -> str:
    """Search saved notes for matching keywords."""

    if not NOTES_FILE.exists():
        return "No notes have been saved yet."

    notes = NOTES_FILE.read_text(encoding="utf-8").splitlines()

    keywords = set(query.lower().split())

    # Ignore very short words that usually add little meaning.
    keywords = {word for word in keywords if len(word) > 2}

    matches = [
        note for note in notes
        if any(word in note.lower() for word in keywords)
    ]

    if not matches:
        return "No matching notes found."

    return "\n".join(matches)

"""def search_documents(query: str) -> str:
    Search project documents and return relevant text with filenames.

    if not DOCUMENTS_DIR.exists():
        return "No documents directory found."

    keywords = {
        word.lower().strip(".,?!:;()")
        for word in query.split()
        if len(word.strip(".,?!:;()")) > 2
    }

    if not keywords:
        return "Please provide a more specific search query."

    results = []

    for file_path in DOCUMENTS_DIR.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        for paragraph in paragraphs:
            paragraph_words = {
                word.lower().strip(".,?!:;()")
                for word in paragraph.split()
            }

            score = len(keywords & paragraph_words)

            if score > 0:
                results.append(
                    (score, file_path.name, paragraph)
                )

    if not results:
        return "No relevant information found in the documents."

    results.sort(key=lambda item: item[0], reverse=True)

    output = []

    for score, filename, paragraph in results[:5]:
        output.append(
            f"Source: {filename}\n"
            f"Relevance score: {score}\n"
            f"Content: {paragraph}"
        )

    return "\n\n".join(output)"""
    
    
    
def get_embedding(text: str) -> list[float]:
    """Convert text into a numerical vector using Ollama."""
    response = embed(
        model="nomic-embed-text",
        input=text,
    )
    return response["embeddings"][0]


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Measure the similarity between two vectors."""
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


"""def search_documents(query: str) -> str:
    Retrieve document paragraphs by semantic similarity.

    if not DOCUMENTS_DIR.exists():
        return "No documents directory found."

    paragraphs = []

    for file_path in DOCUMENTS_DIR.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        for paragraph in text.splitlines():
            paragraph = paragraph.strip()

            if paragraph:
                paragraphs.append(
                    (file_path.name, paragraph)
                )

    if not paragraphs:
        return "No document content found."

    query_embedding = get_embedding(query)
    scored_results = []

    for filename, paragraph in paragraphs:
        paragraph_embedding = get_embedding(paragraph)

        score = cosine_similarity(
            query_embedding,
            paragraph_embedding,
        )

        scored_results.append(
            (score, filename, paragraph)
        )

    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    output = []

    for score, filename, paragraph in scored_results[:3]:
        output.append(
            f"Source: {filename}\n"
            f"Similarity: {score:.4f}\n"
            f"Content: {paragraph}"
        )

    return "\n\n".join(output)"""
 
 

def chunk_text(text: str, max_words: int = 100, overlap: int = 20) -> list[str]:
    """Split text into word-based chunks with a small overlap."""

    if max_words <= 0:
        raise ValueError("max_words must be positive.")

    if overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be between 0 and max_words.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def search_documents(query: str) -> str:
    """Retrieve document paragraphs using cached semantic embeddings."""

    if not DOCUMENTS_DIR.exists():
        return "No documents directory found."

    # Initialize the cache.
    if not hasattr(search_documents, "_cache"):
        search_documents._cache = {}

    cache = search_documents._cache

    # Read document paragraphs.
    paragraphs = []

    for file_path in DOCUMENTS_DIR.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        for paragraph in text.splitlines():
            paragraph = paragraph.strip()

            if paragraph:
                paragraphs.append((str(file_path), paragraph))

    if not paragraphs:
        cache.clear()
        return "No document content found."

    # Rebuild embeddings only when document content changes.
    current_content = tuple(paragraphs)

    if cache.get("content") != current_content:
        cache.clear()
        cache["content"] = current_content
        cache["embeddings"] = [
            get_embedding(paragraph)
            for _, paragraph in paragraphs
        ]

    # Embed the user's query.
    query_embedding = get_embedding(query)
    scored_results = []

    for index, (filename, paragraph) in enumerate(paragraphs):
        paragraph_embedding = cache["embeddings"][index]

        score = cosine_similarity(
            query_embedding,
            paragraph_embedding,
        )

        scored_results.append((score, filename, paragraph))

    # Sort paragraphs by similarity, highest first.
    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # Filter out low-similarity results.
    SIMILARITY_THRESHOLD =  0.50

    relevant_results = [
        item for item in scored_results
        if item[0] >= SIMILARITY_THRESHOLD
    ]

    if not relevant_results:
        return (
            "No sufficiently relevant information was found "
            "in the project documents."
        )

    # Return up to three relevant paragraphs.
    output = []

    for score, filename, paragraph in relevant_results[:3]:
        output.append(
            f"Source: {Path(filename).name}\n"
            f"Similarity: {score:.4f}\n"
            f"Content: {paragraph}"
        )

    return "\n\n".join(output)
