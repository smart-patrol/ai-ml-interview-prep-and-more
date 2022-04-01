from typing import List

def shortest_window_sort(arr:List[int]) -> int:
    """
    Given an array, find the length of the smallest subarray in it which when sorted will sort the whole array.
    """
    low = 0
    high = len(arr) - 1
    # find the first number out of sorting order from the beginning
    while low < high and arr[low] <= arr[low+1]:
        low += 1
    
    if low == high: # if the array is already sorted
        return 0

    # find the first number out of sorting order from the end
    while high > 0 and arr[high] >= arr[high-1]:
        high -= 1
    
    # find the maximum and minimum number of the subarray
    subarray_max = float("-inf")
    subarray_min = float("inf")
    for k in range(low, high+1):
        subarray_max = max(subarray_max, arr[k])
        subarray_min = min(subarray_min, arr[k])

    # extend the subarray to include any number which is bigger than the minimum of the subarray
    while low < 0 and arr[low-1] > subarray_min:
        low -= 1
    while high < len(arr)-1 and arr[high+1] < subarray_max:
        high += 1
    
    return high - low +1


arr = [1, 2, 5, 3, 7, 10, 9, 12]
assert shortest_window_sort(arr) == 5
arr =  [1, 3, 2, 0, -1, 7, 10]
assert shortest_window_sort(arr) == 5
arr = [1, 2, 3]
assert shortest_window_sort(arr) == 0
arr = [3, 2, 1]
assert shortest_window_sort(arr) == 3


def backspace_compare(str1:str, str2:str) -> bool:
    """Given two strings containing backspaces (identified by the character #),
     check if the two strings are equal.
    """
    index1 = len(str1) - 1
    index2 = len(str2) - 1
    while index1 >= 0 or index2 >= 0:
        i1 = get_next_valid_char_index(str1, index1)
        i2 = get_next_valid_char_index(str2, index2)
        if i1 < 0 and i2 < 0:
            return True
        if i1 < 0 or i2 < 0:
            return False
        if str1[i1] != str2[i2]:
            return False
        index1 = i1 - 1
        index2 = i2 - 1
    return True

def get_next_valid_char_index(s, index):
    """Given a string and an index, return the index of the next valid character in the string.
    """
    backspace_count = 0
    while index >= 0:
        if s[index] != '#':
            backspace_count+=1
        elif backspace_count > 0:
            backspace_count-=1
        else:
            break
        index -= 1
    return index


str1="xy#z"; str2="xzz#"
assert backspace_compare(str1, str2) == True
str1="xy#z"; str2="xyz#"
assert backspace_compare(str1, str2) == False
str1="xp#"; str2="xyz##"
assert backspace_compare(str1, str2) == True
str1="xywrrmp"; str2="xywrrmu#p"
assert backspace_compare(str1, str2) == True

def search_quadruplets(arr:List[int], target:int) -> List[List[int]]:
    """
    Given an array of unsorted numbers and a target number, find all unique quadruplets in it, whose sum is equal to the target number.
    """
    arr.sort()
    quadruplets = []
    for i in range(len(arr) - 3):
        if i > 0 and arr[i] == arr[i-1]:
            continue
        for j in range(i+1, len(arr) - 2):
            if j > i + 1 and arr[j] == arr[j-1]:
                continue
            search_quadruplets_helper(arr, target, i,j, quadruplets)
    return quadruplets

def search_quadruplets_helper(arr:List[int], target_sum:int, first:int, second:int, quadruplets:List[List[int]]):
    left = second + 1
    right = len(arr) - 1
    while left < right:
        quad_sum = arr[first] + arr[second] + arr[left] + arr[right]
        if quad_sum == target_sum:
            quadruplets.append(
                [arr[first] + arr[second] + arr[left] + arr[right]])
            left += 1
            right -= 1
            while left < right and arr[left] == arr[left-1]:
                left += 1
            while left < right and arr[right] == arr[right+1]:
                right -=1
        elif quad_sum < target_sum:
            left += 1
        else:
            right -= 1


arr = [4, 1, 2, -1, 1, -3]; target=1
assert search_quadruplets(arr,target) == [[-3, -1, 1, 4], [-3, 1, 1, 2]]
arr = [2, 0, -1, 1, -2, 2];  target=2
assert search_quadruplets(arr,target) == [[-2, 0, 2, 2], [-1, 0, 1, 2]]


def dutch_flag_sort(arr:List[int]) -> List[int]:
    """
    Given an array containing 0s, 1s and 2s, sort the array in-place. 
    You should treat numbers of the array as objects, hence, we can't count 0s, 1s, and 2s to recreate the array.
    """
    low = 0
    high = len(arr)-1
    i = 0
    while i <= high:
        if arr[i] == 0:
            arr[i], arr[low] = arr[low], arr[i]
            i += 1
            low += 1
        elif arr[i] == 1:
            i += 1
        else:
            arr[i], arr[high] = arr[high], arr[i]
            high -= 1
        
    return arr

arr = [1, 0, 2, 1, 0]
assert dutch_flag_sort(arr) == [0, 0, 1, 1, 2, 2]
arr = [2, 2, 0, 1, 2, 0]
assert dutch_flag_sort(arr) == [0, 0, 1, 1, 2, 2]

def find_subarray(arr: List[int], target: int):
    """
    Given an array with positive numbers and a positive target number, find all of its contiguous subarrays whose product is less than the target number.
    """
    result = []
    for i in range(len(arr)-1):
        if arr[i] == arr[i+1]:
            continue
        elif target > arr[i] * arr[i+1]: 
            result.append([arr[i], arr[i+1]])
        else:
            continue
    print(result) 
    return result + arr


arr = [2, 5, 3, 10];  target=30 
assert find_subarray(arr,target) == [[2], [5], [2, 5], [3], [5, 3], [10]]
arr = [8, 2, 6, 5] ; target=50 
assert find_subarray(arr,target) == [[8], [2], [8, 2], [6], [2, 6], [5], [6, 5]]

def triplet_with_smaller_sum_list(arr:List[int], target:int) -> List[List[int]]:
    """Sames triplet_with_smaller_sum but rerurns a list instead of the count"""
    arr.sort()
    triplets = []
    for i in range(len(arr) - 2):
        triplet_helper(arr, target-arr[i], i, triplets)
    
    return triplets

def triplet_helper(arr:List[int], target_sum:int, first:int, triplets:List[List[int]]):
    right = len(arr) - 1
    left = first + 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum < target_sum:
            for i in range(right, left, -1):
                triplets.append([arr[first], arr[left], arr[i]])
            left += 1
        else:
            right-=1

assert triplet_with_smaller_sum_list([-1, 0, 2, 3], 3) == [[-1, 0, 3], [-1, 0, 2]]
assert triplet_with_smaller_sum_list([-1, 4, 2, 1, 3], 5) == [[-1, 1, 4], [-1, 1, 3], [-1, 1, 2], [-1, 2, 3]]




def triplet_with_smaller_sum(arr:List[int], target:int) -> int:
    """
    Given an array arr of unsorted numbers and a target sum, count all triplets in it such that arr[i] + arr[j] + arr[k] < target 
    where i, j, and k are three different indices. Write a function to return the count of such triplets.
    """
    arr.sort()
    count = 0
    for i in range(len(arr) - 2):
        count += helper(arr, target-arr[i], i)
    return count

def helper(arr, target_sum, first):
    count = 0 
    left = first + 1
    right = len(arr) - 1
    while left < right:
        if arr[left] + arr[right] < target_sum:
            count += right - left
            left += 1
        else:
            right -= 1

    return count




arr = [-1, 4, 2, 1, 3]
target=5 
assert triplet_with_smaller_sum(arr, target) == 2
arr = [-1, 0, 2, 3]
target = 3
assert  triplet_with_smaller_sum(arr, target) == 2



def triplet_sum_close_to_target(arr: List[int], target_sum:int) -> int:
    """
    Given an array of unsorted numbers and a target number, find a triplet in the array whose sum is as close to the target number as possible, return the sum of the triplet. If there are more than one such triplet, return the sum of the triplet with the smallest sum.  
     """
    arr.sort()
    min_diff = float('inf')
    for i in range(len(arr) - 2):
        left = i + 1
        right = len(arr) - 1
        while left < right:
            curr_sum = arr[i] + arr[left] + arr[right]
            curr_diff = abs(curr_sum - target_sum)
            if curr_diff < min_diff:
                min_diff = curr_diff
                min_sum = curr_sum
            if curr_sum < target_sum:
                left += 1
            elif curr_sum > target_sum:
                right -= 1
            else:
                return curr_sum

    return min_sum

arr =  [-2, 0, 1, 2]
assert triplet_sum_close_to_target(arr,2)  == 1
arr = [1, 0, 1, 1]
assert triplet_sum_close_to_target(arr,100)  == 3

def triplet_sum_zero(arr: List[int]) -> List[List[int]]:
    "Given an array of unsorted numbers, find all unique triplets in it that add up to zero."
    arr.sort()
    triplets = []
    for i in range(len(arr)):
        if i > 0 and arr[i] == arr[i-1]:
            continue
        search_helper(arr, -arr[i], i+1, triplets)

    return triplets

def search_helper(arr:List[int], target_sum:int, left:int, triplets:List[List[int]]) -> None:
    right = len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target_sum:
            triplets.append([-target_sum, arr[left], arr[right]])
            left += 1
            right -= 1
            # if same number is encountered again, skip it
            while left < right and arr[left] == arr[left-1]:
                left+=1
            while left < right and arr[right] == arr[right+1]:
                right-=1
        elif current_sum < target_sum:
            left += 1
        else:
            right-=1



arr = [-3, 0, 1, 2, -1, 1, -2]
assert triplet_sum_zero(arr) == [[-3, 1, 2], [-2, 0, 2], [-2, 1, 1], [-1, 0, 1]]
arr = [-3, 0, 1, 2, -1, 1, -2]
assert triplet_sum_zero(arr) == [[-5, 2, 3], [-2, -1, 3]]





def make_squares(arr:List[int]) -> List[int]:
    """Given a sorted array, create a new array containing squares of all the numbers of the input 
    array in the sorted order."""
    n = len(arr)
    squares = [0] * n
    highestSquareIdx = n -1
    left = 0
    right = n - 1

    while left <= right:
        left_square = arr[left] * arr[left]
        right_square = arr[right] * arr[right]
        if left_square > right_square:
            squares[highestSquareIdx] = left_square
            left += 1
        else:
            squares[highestSquareIdx] = right_square
            right -= 1
        highestSquareIdx -= 1
    
    return squares

arr = [-2, -1, 0, 2, 3]
assert make_squares(arr) == [0, 1, 4, 4, 9]
arr = [-3, -1, 0, 1, 2]
assert make_squares(arr) == [0, 1, 1, 4, 9]

# def remove_duplicates(arr:List[int]) -> int:
#     return len(set(arr))

def remove_duplicates(arr:List[int]) -> int:
    """
    Given an array of sorted numbers, remove all duplicates from it. You should not use any extra space; after removing the duplicates in-place return the length of the subarray that has no duplicate in it.
    """
    next_non_duplicate = 0

    i = 1
    while i < len(arr):
        if arr[next_non_duplicate] != arr[i]:
            next_non_duplicate += 1
            arr[next_non_duplicate] = arr[i]
        i += 1
    return next_non_duplicate

arr = [2, 3, 3, 3, 6, 9, 9]
assert remove_duplicates(arr) == 4
arr = [2, 2, 2, 11]
assert remove_duplicates(arr) == 2

# def pair_with_targetsum(arr: List[int], target:int) -> int:
#     """
#     Given an array of sorted numbers and a target sum, find a pair in the array whose sum is equal to the given target.
#     """
#     hash_map = {}
#     for i,num in enumerate(arr):
#         if target - num in hash_map:
#             return [hash_map[target], i]
#         hash_map[num] = i

def pair_with_targetsum(arr: List[int], target:int) -> List[int]:
    left = 0
    right = len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left+=1
        else:
            right-=1
    
    return [-1, -1]


arr = [1, 2, 3, 4, 6]; target=6
assert pair_with_targetsum(arr, target) == [1, 3]
arr = [2, 5, 9, 11] ;  target=11
assert pair_with_targetsum(arr, target) == [0,2]