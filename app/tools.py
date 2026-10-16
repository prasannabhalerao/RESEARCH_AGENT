def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic on two numbers.

    Args:
        a: The first number.
        b: The second number.
        operation: One of add, subtract, multiply, or divide.
    """
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

    raise ValueError("Unsupported operation.")