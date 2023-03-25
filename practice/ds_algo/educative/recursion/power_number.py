def power(base: int, exponent: int) -> int:
    # base case
    if exponent == 0:
        return 1
    # recursive case
    else:
        return base * power(base, exponent - 1)


def power_iterative(base: int, exponent: int) -> int:
    """
    iterative power of a given exponent
    """
    out: int = 1
    for _ in range(exponent):
        out *= base
    return out


assert power(2, 3) == 8
assert power(3, 6) == 729

assert power_iterative(2, 3) == 8
assert power_iterative(3, 6) == 729
