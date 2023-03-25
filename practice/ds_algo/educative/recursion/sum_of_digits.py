def sum_digits_iter(digits: str) -> int:
    ttl: int = 0
    for _, d in enumerate(digits):
        ttl += int(d)
    return ttl


def sum_digits_recursive(digits: str) -> int:
    # base case
    if digits == "":
        return 0
    else:
        # recursive case
        return int(digits[0]) + sum_digits_iter(digits[1:])


assert sum_digits_iter("345") == 12
assert sum_digits_iter("1") == 1
assert sum_digits_iter("050") == 5


assert sum_digits_recursive("345") == 12
assert sum_digits_recursive("1") == 1
assert sum_digits_recursive("050") == 5
