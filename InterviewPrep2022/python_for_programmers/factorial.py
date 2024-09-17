from smtplib import SMTPServerDisconnected


def factorial(n: int) -> int:
    if n < 0:
        return -1

    fact = 1
    for i in range(1, n + 1):
        fact *= i
    return fact


def check_balance(brackets: str) -> bool:
    stack = []
    for bracket in brackets:
        if bracket == "[":
            stack.append("[")
        elif bracket == "]":
            if len(stack) == 0:
                return False
            stack.pop()
    return len(stack) == 0


def check_sum(num_list: list) -> bool:
    """You must implement the check_sum() function which takes in a list and returns True if the sum of two numbers in the list is zero. If no such pair exists, return False."""
    for i in range(len(num_list)):
        for j in range(i + 1, len(num_list)):
            if num_list[i] + num_list[j] == 0:
                return True
    return False


num_list = [10, -14, 26, 5, -3, 13, -5]

assert check_sum(num_list) == True


def fibonacci(n: int) -> int:
    # The base cases
    if n <= 1:  # First number in the sequence
        return 0
    elif n == 2:  # Second number in the sequence
        return 1
    else:
        # Recursive call
        return fibonacci(n - 1) + fibonacci(n - 2)


def iterative_fibonacci(n: int) -> int:
    first = 0
    second = 1

    if n < 1:
        return -1

    elif n == 1:
        return first
    elif n == 2:
        return second

    count = 3
    while count <= n:
        fib_n = first + second
        first = second
        second = fib_n
        count += 1
    return fib_n


assert iterative_fibonacci(7) == 8
