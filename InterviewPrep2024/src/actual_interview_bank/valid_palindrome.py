"""
Screen round interview question from company A
Check if string is a palindrome, considering only alphanumeric characters
"""


def valid_palindrome(s: str) -> bool:
    # Remove non-alphanumeric characters and convert to lowercase
    s = "".join(e.lower() for e in s if e.isalnum())

    # Compare characters from both ends of the string
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1

    return True


# Test cases
print(valid_palindrome("A man, a plan, a canal: Panama"))  # True
print(valid_palindrome("race a car"))  # False
print(valid_palindrome("Able was I ere I saw Elba"))  # True
print(valid_palindrome("No 'x' in Nixon"))  # False
print(valid_palindrome("Madam, in Eden, I'm Adam"))  # True
print(valid_palindrome("Was it a car or a cat I saw?"))  # False
