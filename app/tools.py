def calculator(a, b, operation):
    """Perform a basic arithmetic calculation."""

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")

        return a / b

    else:
        raise ValueError("Unsupported operation.")