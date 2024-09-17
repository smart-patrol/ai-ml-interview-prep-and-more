from typing import List
import math
from random import randint, shuffle


def maxSubArray(nums: List[int]) -> int:
    """
    Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
    
    A subarray is a contiguous part of an array.
    """
    if not nums:
        return 0
    max_sum = nums[0]
    curr_sum = nums[0]
    for i in range(1, len(nums)):
        curr_sum = max(nums[i], curr_sum + nums[i])
        max_sum = max(max_sum, curr_sum)
    return max_sum

nums = [-2,1,-3,4,-1,2,1,-5,4]
assert maxSubArray(nums) == 6
nums = [1]
assert maxSubArray(nums) == 1
nums = [5,4,-1,7,8]
assert maxSubArray(nums) == 23


def maxSubArray2(nums: List[int]) -> int:
    """Divide and conquer approach for maxSubArray"""
    def findBestSubarray(nums:List[int], left:int,right:int) -> int:
        # base case - empty array
        if left > right:
            return -math.inf
        
        mid = (left + right) // 2
        curr = best_left_sum = best_right_sum = 0

        # Iterate from the middle to the beginning
        for i in range(mid - 1, left - 1, -1):
            curr += nums[i]
            best_left_sum = max(best_left_sum, curr)
        
        # Reset curr and iterate from the middle to the end
        curr = 0
        for i in range(mid+1, right+1):
            curr += nums[i]
            best_right_sum = max(best_right_sum, curr)
        
        # test best combind sum uses middle element
        best_combined_sum = best_left_sum + best_right_sum + nums[mid]

        # recursively find best subarray on either side of the middle
        left_half = findBestSubarray(nums, left, mid - 1)
        right_half = findBestSubarray(nums, mid + 1, right)

        # the largest of the 3 is the answer
        return max(best_combined_sum, left_half, right_half)

    return findBestSubarray(nums, 0, len(nums) - 1)

nums = [-2,1,-3,4,-1,2,1,-5,4]
assert maxSubArray2(nums) == 6
nums = [1]
assert maxSubArray2(nums) == 1
nums = [5,4,-1,7,8]
assert maxSubArray2(nums) == 23


# implement quickselect
# 1) Choose a pivot p  
# 2) Partition the array in two sub-arrays w.r.t. p (same partitioning as in quicksort)
# 3) LEFT –> elements smaller than or equal to p  
# 4) RIGHT–>  elements greater than p  
# 5) If index(pivot) == k:  
# 6)    Return pivot (or index of pivot)  
# 7) If k > index(pivot)  
# 8)    QuickSelect(LEFT)  
# 9) Else:  
# 10    QuickSelect(RIGHT)

#https://github.com/maximerihouey/QuickSelect-playground/blob/master/python/quickselect.py

def quickselect(arr:List[int], k:int) -> int:

    n = k-1

    def partition(left:int, right:int, pivot_index:int) -> int:
        pivot_value = arr[pivot_index]
        # move pivot to end
        arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
        store_index = left
        for i in range(left, right):
            if arr[i] < pivot_value:
                arr[store_index], arr[i] = arr[i], arr[store_index]
                store_index += 1
        # move pivot to its final place
        arr[right], arr[store_index] = arr[store_index], arr[right]

    def select(left:int, right:int):
        if left == right:
            return arr[left]
        pivot_index = randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        # the pivot is in its final sorted position
        if n == pivot_index:
            return arr[n]
        elif n < pivot_index:
            return select(left, pivot_index - 1)
        else:
            return select(pivot_index + 1, right)

example_array = list(range(1, 11+1))
shuffle(example_array)
print(quickselect(example_array, 5)

#https://www.askpython.com/python/examples/quicksort-algorithm

def pivot(arr:List[int], start:int, end:int) -> int:
    pivot_index =arr[start]
    low = start + 1
    high = end

    while True:
        # moving high towards left
        while low <= high and arr[high] >= pivot:
            high = high - 1
        # moving low towrds right
        while low <= high and arr[low] <= pivot:
            low = low + 1

        # check if low and high have not crossed
        if low <= high:
            arr[low], arr[high] = arr[high], arr[low]

        else:
            break

        # swapping pivot with high so that pivot is at it's right
        arr[start], arr[high] = arr[high], arr[start]

        return high

def quick_sort(arr:List[int], start:int, end:int):
    if start >= end:
        return
    # call pivot
    pivot_index = pivot(arr, start, end)
    # recursively sort left and right
    quick_sort(arr,start,pivot_index-1)
    quick_sort(arr,pivot_index+1, end)

def run_quick_sort(arr:List[int]):
    return quick_sort(arr, 0, len(arr) - 1)