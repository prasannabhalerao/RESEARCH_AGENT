from ollama import chat

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an AI tutor for a beginner. "
                "Explain concepts in very simple language. "
                "Always give one practical example."
            )
        },
        {
            "role": "user",
            "content": "Explain what an API is."
        }
    ]
)

print(response["message"]["content"])