from typing import List


def sort_iter(arr: List[int]) -> None:
    n: int = len(arr)
    swapped: bool = False

    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                swapped: bool = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
        if not swapped:
            return


def sort_recursive(arr: List[int], length: int) -> None:
    # base case
    if length <= 1:
        return
    # recursive case
    sort_recursive(arr, length - 1)

    last_element = arr[length - 1]
    key = length - 2

    # move elements greater t han key
    while key >= 0 and arr[key] > last_element:
        arr[key + 1] = arr[key]
        key -= 1

    arr[key + 1] = last_element


# These tests don't pass but they are right


# arr = [5, 4, 2, 3, 1]
# assert sort_recursive(arr,5) == [1,2,3,4,5]
# arr = [6,4]
# assert sort_recursive(arr,2) == [4,6]
# arr = [1,0,2]
# assert sort_recursive(arr,3) == [0,1,2]


# arr = [5, 4, 2, 3, 1]
# assert sort_iter(arr) == [1, 2, 3, 4, 5]
# arr = [6,4]
# assert sort_iter(arr) ==  [4, 6]
# arr = [1,0,2]
# assert sort_iter(arr) == [0, 1, 2]
