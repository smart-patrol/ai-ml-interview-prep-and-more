from typing import List


def average(arr: List[int], idx: int) -> int:
    """Compute the average of an array of integers"""
    # base case
    if idx == len(arr) - 1:
        return arr[idx]
    # recursive case 1
    if idx == 0:
        return (arr[idx] + average(arr, idx + 1)) / len(arr)
    # recursive case 2
    return arr[idx] + average(arr, idx + 1)


assert average([10, 2, 3, 4, 8, 0], 0) == 4.5
assert average([5, 0, 0, 0, 0], 0) == 1.0
assert average([1, 2], 0) == 1.5
