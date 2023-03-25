"""
Merge Sort
Here is the recursive mergesort algorithm:

1) If the list has only one element, return the list and terminate. (Base case)
2) Split the list into two halves that are as equal in length as possible. (Divide)
3) Using recursion, sort both lists using mergesort. (Conquer)
4) Merge the two sorted lists and return the result. (Combine)

"""
from typing import List


def merge(left: List[int], right: List[int]) -> List[int]:
    out = []
    l, r = 0, 0
    while l < len(left) and r < len(right):
        if left[l] <= right[r]:
            out.append(left[l])
            l += 1
        else:
            out.append(right[r])
            r += 1
    if left:
        out.extend(left[l:])
    if right:
        out.extend(right[r:])
    return out


def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    print(f"array split {arr}")
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return list(merge(left, right))


if __name__ == "__main__":
    arr = [2, 4, 9, 1, 7, 13, 15]
    out = merge_sort(arr)
    print(f"Sorted array:{out}")
