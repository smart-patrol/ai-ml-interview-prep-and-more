"""
Quicksort
"""

from typing import List


def quick_sort(arr: List[int]) -> List[int]:
    left, right, pivot_list = [], [], []
    if len(arr) <= 1:
        return arr
    else:
        print(arr)
        # pivot = arr[0]
        # using median of 3
        n = len(arr)
        if n > 2:
            pivot = sorted([arr[0]] + [arr[n // 2]] + [arr[-1]])[1]
        else:
            pivot = arr[0]
        for num in arr:
            if num < pivot:
                left.append(num)
            elif num > pivot:
                right.append(num)
            else:
                pivot_list.append(num)

        left = quick_sort(left)
        right = quick_sort(right)

        return left + pivot_list + right


if __name__ == "__main__":
    arr = [5, 6, 33, 129, 16, 22, 8, 31, 24]
    out = quick_sort(arr)
    print(out)
