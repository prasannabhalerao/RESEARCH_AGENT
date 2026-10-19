
from app.tools import search_documents


test_cases = [
    {
        "question": "Which programming language is used?",
        "expected": "Python",
    },
    {
        "question": "Which model runs locally?",
        "expected": "Qwen3 8B",
    },
    {
        "question": "What is the calculator?",
        "expected": "calculator is a Python function",
    },
]


for test in test_cases:
    print("\n" + "=" * 60)
    print("Question:", test["question"])

    result = search_documents(test["question"])

    print("Retrieved passages:")
    print(result)

    if test["expected"].lower() in result.lower():
        print("Evaluation: PASS")
    else:
        print("Evaluation: CHECK")
        print("Expected term:", test["expected"])
