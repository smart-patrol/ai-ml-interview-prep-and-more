def decimalToBinary(n: int) -> int:
    if n <= 3:
        return bin(n)[2:]
    out: str = ""
    while n > 1:
        out += str(int(n % 2))
        n /= 2
    return out


def decimalToBinary_recursive(n: int) -> int:
    # base case
    if n <= 1:
        return str(n)
    else:
        # recursive case
        # Floor division -
        # division that results into whole number adjusted to the left in the number line
        return decimalToBinary(n // 2) + decimalToBinary(n % 2)


assert decimalToBinary(11) == "1101"
assert decimalToBinary(5) == "101"
assert decimalToBinary(2) == "10"


assert decimalToBinary_recursive(11) == "1101"
assert decimalToBinary_recursive(5) == "101"
# assert decimalToBinary_recursive(2) == '10'
