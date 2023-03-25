def reverse_string_iterative(s: str) -> str:
    """
    Reverse a string recursively
    """
    out: str = ""
    ln: int = len(s) - 1
    while ln >= 0:
        out: str = out + s[ln]
        ln -= 1
    return out


def reverse_string_recursive(s: str) -> str:
    # base case
    if len(s) == 0:
        return s
    # recursive case
    else:
        return reverse_string_recursive(s[1:]) + s[0]


s: str = "Hello World"
assert reverse_string_recursive(s) == "dlroW olleH"
assert reverse_string_iterative(s) == "dlroW olleH"
