"""
Binary search in python
"""


# Binary search function, searhing left over any array
def binary_search_left(arr, x):
    """Find the first occurrence of x in arr"""
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def binary_search_right(arr, x):
    """Find the last occurrence of x in arr"""
    left = 0
    right = len(arr) - 1

    while left < right:
        mid = (left + right) // 2

        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid

    return -1


# tests
arr = [1, 2, 3, 3, 3, 4, 5, 6, 7]
x = 3

print("First occurrence of", x, "is at index", binary_search_left(arr, x))
print("Last occurrence of", x, "is at index", binary_search_right(arr, x))
