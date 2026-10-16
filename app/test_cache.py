
from app.tools import search_documents


question_one = "Which programming language is used by the application?"
question_two = "Which AI model runs locally?"

print("First search:")
print(search_documents(question_one))

print("\nSecond search:")
print(search_documents(question_two))

print("\nCache test:")
print("Cached embeddings:", len(search_documents._cache["embeddings"]))
print("Cache contains document content:", "content" in search_documents._cache)
