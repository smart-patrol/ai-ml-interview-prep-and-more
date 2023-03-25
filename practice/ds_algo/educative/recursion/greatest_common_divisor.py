def gcd(a: int, b: int) -> int:
    if a == b:
        return a
    # recursive case
    if a > b:
        return gcd(a - b, b)
    else:
        return gcd(b - a, a)


def gcd_iterative(a: int, b: int) -> int:
    while a:
        temp: int = b % a
        b: int = a
        a: int = temp
    return b


assert gcd(56, 42) == 14
assert gcd(14, 30) == 2
assert gcd(23, 23) == 23


assert gcd_iterative(56, 42) == 14
assert gcd_iterative(14, 30) == 2
assert gcd_iterative(23, 23) == 23
