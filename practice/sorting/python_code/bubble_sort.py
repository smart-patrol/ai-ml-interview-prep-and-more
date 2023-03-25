"""
Bubble Sort
The Bubble sort algorithm compares each pair of elements in an array and swaps them if they are out of order until the entire array is sorted. For each element in the list, the algorithm compares every pair of elements.

By convention, empty arrays and singleton arrays (arrays consisting of only one element) are always sorted. This is a key point for the base case of many sorting algorithms.

Source: https://brilliant.org/wiki/bubble-sort/
"""

from typing import List


def bubble_sort(arr: List[int]) -> List[int]:
    """Bubble Sort"""
    if len(arr) == 1 or arr is None:
        return arr
    index = len(arr) - 1
    while index >= 0:
        for j in range(index):
            if arr[j] > arr[j + 1]:
                arr[j + 1], arr[j] = arr[j], arr[j + 1]
        index -= 1
        print(arr, index)

    return arr


def bubble_sort_recursive(arr: List[int], n: int) -> List[int]:
    if n <= 1:
        return arr

    for i in range(n - 1):
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]

    bubble_sort_recursive(arr, n - 1)
    return arr


if __name__ == "__main__":
    arr = [12, 4, 8, 2, 15, 13, 1]
    # print(bubble_sort(arr))
    print(bubble_sort_recursive(arr, len(arr)))
