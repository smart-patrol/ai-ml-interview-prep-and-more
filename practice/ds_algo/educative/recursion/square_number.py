def square_number(n: int) -> int:
    """Returns the square of number recursively"""
    # base case
    if n == 0:
        return 0
    # recursive case
    return square_number(n - 1) + (2 * n) - 1


def square_number_iterative(n: int) -> int:
    """Returns the square of number iteratively"""
    out: int = 0
    while n > 0:
        out += 2 * n - 1
        n -= 1
    return out


assert square_number(5) == 25
assert square_number(2) == 4
assert square_number(1) == 1


assert square_number_iterative(5) == 25
assert square_number_iterative(2) == 4
assert square_number_iterative(1) == 1
