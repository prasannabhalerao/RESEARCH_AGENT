from pathlib import Path


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

def search_documents(query: str) -> str:
    """Search project documents and return relevant text with filenames."""

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

    return "\n\n".join(output)