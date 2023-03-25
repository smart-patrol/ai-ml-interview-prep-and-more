def factorial_recursive(num: int) -> int:
    # Base case
    if num < 0:
        return 1
    else:
        return num * factorial_recursive(num - 1)


def factorial_iterative(targetNumber: int) -> int:
    """
    compute the factorial using iteration
    """
    index: int = targetNumber - 1  # set the index to target - 1

    while index >= 1:
        targetNumber: int = (
            targetNumber * index
        )  # multiply targetNumber with one less than itself, i.e, index here
        index -= 1  # reduce index in each iteration

    return targetNumber


assert factorial_iterative(5) == 120
assert factorial_recursive(5) == 120
