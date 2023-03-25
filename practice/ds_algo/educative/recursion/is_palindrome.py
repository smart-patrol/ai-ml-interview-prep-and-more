def isPalindrome_recursive(s: str) -> bool:
    """
    Returns true if the given string is a palindrome, recursively
    """
    # base case
    if s == "":
        return True
    # recursive case
    ln: int = len(s)
    if s[0] == s[ln - 1]:
        return isPalindrome_recursive(s[1 : ln - 1])

    return False


def isPalindrome_iter(s: str) -> bool:
    if s == "":
        return True
    else:
        j: int = len(s)
        for i in range(j):
            if s[i] != s[j - 1]:
                return False
            j -= 1
    return True


assert isPalindrome_recursive("madam") == True
assert isPalindrome_recursive("0110") == True
assert isPalindrome_recursive("") == True
assert isPalindrome_recursive("charles") == False


assert isPalindrome_iter("madam") == True
assert isPalindrome_iter("0110") == True
assert isPalindrome_iter("") == True
assert isPalindrome_iter("charles") == False
