"""
    Insertion Sort
    The insertion sort algorithm iterates through an input array and removes one element per iteration, finds the place the element belongs in the array, and then places it there. This process grows a sorted list from left to right. The algorithm is as follows:
    https://brilliant.org/wiki/insertion/
"""

from typing import List


def insertion_sort(arr: List[int]) -> List[int]:
    """Insertion sort algorithm"""
    for i in range(len(arr)):
        x = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > x:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = x
        print(arr)
    return arr


def insertion_sort_recursive(arr: List[int], n) -> List[int]:
    if n <= 1:
        return arr
    last = arr[n - 1]
    j = n - 2
    while j >= 0 and arr[j] > last:
        arr[j + 1] = arr[j]
        j -= 1
    arr[j + 1] = last
    insertion_sort_recursive(arr, n - 1)
    return arr


if __name__ == "__main__":
    arr = [6, 5, 3, 1, 8, 7, 2, 4]
    out = insertion_sort(arr)
    print(out)
    out = insertion_sort_recursive(arr, len(arr))
    print(out)
