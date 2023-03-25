def isVowel(c: str) -> bool:
    c: str = c.lower()
    vowels: str = "aeiou"

    if c in vowels:
        return 1
    else:
        return 0


def countVowels(s: str) -> int:
    """
    Count the number of vowels in a string iteratively
    """
    count: int = 0
    for c in s:
        if isVowel(c):
            count += 1
    return count


def countVowelsRecursive(s: str, n: int) -> int:
    """
    Count the number of vowels recursively
    """
    # base case
    if n == 1:
        return isVowel(s[0])

    # recursive case
    return countVowelsRecursive(s, n - 1) + isVowel(s[n - 1])


assert countVowels("Charles") == 2
assert countVowelsRecursive("Charles", 7) == 2
