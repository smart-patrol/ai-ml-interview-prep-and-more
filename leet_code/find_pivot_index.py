from typing import List
from collections import defaultdict

def pivotIndex(nums: List[int]) -> int:
    """
    Given an array of integers nums, calculate the pivot index of this array.
    The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal to the sum of all the numbers strictly to the index's right.
    
    If the index is on the left edge of the array, then the left sum is 0 because there are no elements to the left. This also applies to the right edge of the array.
    
    Return the leftmost pivot index. If no such index exists, return -1.
    """
    total = sum(nums)
    left_sum = 0
    for i, num in enumerate(nums):
        if left_sum == total - left_sum - num:
            return i
        left_sum += num
    return -1


assert pivotIndex([1,7,3,6,5,6]) == 3
assert pivotIndex([1,2,3]) == -1
assert pivotIndex([2,1,-1]) == 0