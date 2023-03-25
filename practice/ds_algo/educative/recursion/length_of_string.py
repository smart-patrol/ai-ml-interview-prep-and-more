def recursiveLength(s: str) -> int:
    """
    count the length of string using recursion
    """
    # base case
    if s == "":
        return 0
    # recursive case
    else:
        return 1 + recursiveLength(s[1:])


assert recursiveLength("Hello World") == 11
assert recursiveLength("Charles") == 7
assert recursiveLength("") == 0
