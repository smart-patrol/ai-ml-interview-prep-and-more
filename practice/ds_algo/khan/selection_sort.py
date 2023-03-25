from typing import List


def selection_sort(arr:List[int]) -> List[int]:
    """
    1) Initialize minimum value(min_idx) to location 0.
    2) Traverse the array to find the minimum element in the array.
    3) While traversing if any element smaller than min_idx is found then 
swap both the values.
    4) Then, increment min_idx to point to the next element.
    5) Repeat until the array is sorted.
    """
    for i in range(len(arr)):
        min_idx: int = i

        # find the minimum element in the array
        for j in range(i+1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx: int = j
        # swap the min_idx to the next element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]


A: List[int] = [64, 25, 12, 22, 11]
selection_sort(A)
assert A == [11,12,22,25,64]
