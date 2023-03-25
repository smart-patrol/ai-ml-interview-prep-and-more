from functools import lru_cache


# lru makes it top down dp
@lru_cache(maxsize=100)
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 2) + fibonacci(n - 1)


def fibonacci_iterative(n: int) -> int:
    """
    iterative fibonacci function
    """
    fn0: int = 0
    fn1: int = 1
    for i in range(n):
        temp: int = fn0 + fn1

        # setting for the next iteration
        fn0: int = fn1
        fn1: int = temp
    return fn0


def fibonacci_dp(n: int) -> int:
    if n <= 1:
        return n
    cache: list = [0] * (n + 1)
    cache[1] = 1
    for i in range(2, n + 1):
        cache[i] = cache[i - 1] + cache[i - 2]
    return cache[n]


assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(7) == 13

assert fibonacci_iterative(1) == 1
assert fibonacci_iterative(2) == 1
assert fibonacci_iterative(7) == 13

assert fibonacci_dp(1) == 1
assert fibonacci_dp(2) == 1
assert fibonacci_dp(7) == 13
