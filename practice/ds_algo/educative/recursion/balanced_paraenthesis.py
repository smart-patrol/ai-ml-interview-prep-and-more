from typing import List


def balanced_recursive(arr: List[int], start_idx: int = 0, curr_idx: int = 0) -> bool:
    """ """
    # base cases 1 and 2
    if start_idx == len(arr):
        return curr_idx == 0
    # base case 3
    # A closing bracket did not find its corresponding opening bracket
    if curr_idx < 0:
        return False
    # recursive case 1
    if arr[start_idx] == "(":
        return balanced_recursive(arr, start_idx + 1, curr_idx + 1)
    # recursive case 2
    if arr[start_idx] == ")":
        return balanced_recursive(arr, start_idx + 1, curr_idx - 1)


def balanced_iter(arr: List[int], start_idx: int = 0, curr_idx: int = 0) -> bool:
    """ """
    stack: List[int] = []
    mapping: dict = {")": "(", "}": "{", "]": "["}
    for char in arr:
        if char in mapping:
            top_element: int = stack.pop() if stack else "#"
        if mapping[char] != top_element:
            return False
        else:
            stack.append(char)
    return not stack


# assert balanced_iter(['(', '(', ')', ')', '(', ')']) == True
# assert balanced_iter(['(', ')', '(', ')']) == True
# assert balanced_iter(['(', '(', ')', '(', ')']) == False


assert balanced_recursive(["(", "(", ")", ")", "(", ")"]) == True
assert balanced_recursive(["(", ")", "(", ")"]) == True
assert balanced_recursive(["(", "(", ")", "(", ")"]) == False
