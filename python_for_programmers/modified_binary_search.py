from typing import List
import math

def count_rotations(arr:List[int]) -> int:
    """
    Given an array of numbers which is sorted in ascending order and is 
    rotated k times around a pivot, find k.
    You can assume that the array does not have any duplicates.
    """
    left = 0
    right = len(arr) - 1

    while left < right:
        mid = left +  (right-left) // 2

        # if mid is greater than the next element
        if mid < right and arr[mid] > arr[mid+1]:
            return mid+1

        # if mid is less than the previous element
        if mid > left and arr[mid] < arr[mid-1]:
            return mid

        if arr[left] < arr[mid]:  # left side is sorted so pivot on the right side
            left = mid + 1
        else:  # right side is sorted so pivot on the left side
            right  = mid - 1
        
    return 0

assert count_rotations([10, 15, 1, 3, 8]) == 2
assert count_rotations([4, 5, 7, 9, 10, -1, 2] ) == 5
assert count_rotations([1, 3, 8, 10]) == 0

def search_rotated_array(arr:List[int], key:int) -> int:
    """
    Given an array of numbers which is sorted in ascending order and also rotated by some arbitrary number, find if a given key is present in it.
    
    Write a function to return the index of the key in the rotated array. 
    If the key is not present, return -1.
    You can assume duplicates are possible.
    """
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == key:
            return mid
        
        # if there are duplicates
        if arr[left] == arr[mid] and arr[right] == arr[mid]:
            left += 1
            right -= 1
            continue
        
        if arr[left] <= arr[mid]: # left side is sorted in ascending order
            if key >= arr[left] and key < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else: # right side is sorted in ascending order
            if key > arr[mid] and key <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
        
    return -1
    

assert search_rotated_array([10, 15, 1, 3, 8], 15) == 1 
assert search_rotated_array([4, 5, 7, 9, 10, -1, 2], 10) == 4


def search_bitonic_array(arr: List[int], key:int) -> int:
    """
    Given a Bitonic array, find if a given key is present in it. 
    An array is considered bitonic if it is monotonically increasing and then monotonically decreasing. 
    Monotonically increasing or decreasing means that for any index i in the array arr[i] != arr[i+1].
    Write a function to return the index of the key. If the key is not present, return -1.
    """

    # first find the max element in the array return it's index
    def find_max(arr:List[int]) -> int:
        start, end = 0, len(arr) - 1
        while start < end:
            mid = start + (end - start) // 2
            if arr[mid] > arr[mid + 1]:
                end = mid
            else:
                start = mid + 1
        # at the end of the while loop, 'start == end'
        return start

    
    def binary_search_helper(arr:List[int], key:int, start:int, end:int) -> int:
        while start <= end:
            mid = int(start + (end - start) / 2)

            if key == arr[mid]:
                return mid

            if arr[start] < arr[end]:  # ascending order
                if key < arr[mid]:
                        end = mid - 1
                else:  # key > arr[mid]
                    start = mid + 1
            else:  # descending order
                if key > arr[mid]:
                    end = mid - 1
                else:  # key < arr[mid]
                    start = mid + 1

        return -1  # element is not found

    max_index = find_max(arr) # max value index
    # search in ascending order for key
    key_index = binary_search_helper(arr, key, 0, max_index)
    if key_index != -1:
        return key_index
    # search in descending order for key
    return binary_search_helper(arr, key, max_index + 1, len(arr) - 1)

assert search_bitonic_array([1, 3, 8, 4, 3], 4) == 3
assert search_bitonic_array([3, 8, 3, 1], 8) == 1
assert search_bitonic_array([1, 3, 8, 12], 12) == 3
assert search_bitonic_array([10, 9, 8], 10) == 0

def find_max_in_bitonic_array(arr:List[int]) -> int:
    left = 0
    right = len(arr) - 1

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1

    return arr[left]

assert find_max_in_bitonic_array([1, 3, 8, 12, 4, 2]) == 12
assert find_max_in_bitonic_array([3, 8, 3, 1]) == 8
assert find_max_in_bitonic_array([1, 3, 8, 12]) == 12
assert find_max_in_bitonic_array([10, 9, 8])  == 10





def search_min_diff_element(arr:List[int], key:int) -> int:
    """
    Given an array of numbers sorted in ascending order, find the element in 
    the array that has the minimum difference with the given key.
    """
    if key < arr[0]:
        return arr[0]
    n = len(arr)
    if key > arr[n - 1]:
        return arr[n - 1]

    start, end = 0, n - 1
    while start <= end:
        mid = start + (end - start) // 2
        if key < arr[mid]:
            end = mid - 1
        elif key > arr[mid]:
            start = mid + 1
        else:
            return arr[mid]

    # at the end of the while loop, 'start == end+1'
    # we are not able to find the element in the given array
    # return the element which is closest to the 'key'
    if (arr[start] - key) < (key - arr[end]):
        return arr[start]
    return arr[end]

assert search_min_diff_element([4, 6, 10], 7) == 6
assert search_min_diff_element([4, 6, 10], 4) == 4
assert search_min_diff_element([1, 3, 8, 10, 15], 12) == 10
assert search_min_diff_element([4, 6, 10], 17) == 10


class ArrayReader:

  def __init__(self, arr):
    self.arr = arr

  def get(self, index):
    if index >= len(self.arr):
      return math.inf
    return self.arr[index]


def search_in_infinite_array(reader:'ArrayReader', key: int) -> int:
    """
    Given an array of sorted numbers, find a given number 'key'.
    The array has infinite size, i.e. it wraps around.
    You should assume that the array has infinite size.
    You can assume that there will be at-most one duplicate number in the array.
    """
    start = 0
    end = 1

    while reader.get(end) < key:
        new_start = end + 1
        end += (end - start+ 1) * 2
        start = new_start
    
    return binary_search_reader(reader, key, start, end)


def binary_search_reader(reader, key, start, end):
    
    while start <= end:
        mid = start + (end - start) // 2
        if key < reader.get(mid):
            end = mid - 1
        elif key > reader.get(mid):
            start = mid + 1
        else:
            return mid

    return -1


reader = ArrayReader([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
assert search_in_infinite_array(reader, 16) == 6
assert search_in_infinite_array(reader, 11) == -1
reader = ArrayReader([1, 3, 8, 10, 15])
assert search_in_infinite_array(reader, 15) == 4
assert search_in_infinite_array(reader, 200) == -1



def find_range(arr: List[int], key: int) -> List[int]:
    """
    Given an array of numbers sorted in ascending order, find the range of
    a given number key.
    The range of the key will be the first and last position of the key
     in the array.
    """

    def binary_search(arr, key, findMaxIndex):
        left = 0
        right = len(arr) - 1
        key_index = -1

        while right >= left:
            mid = left + (right - left) // 2

            if key < arr[mid]:
                right = mid - 1
            elif key > arr[mid]:
                left = mid + 1
            else:
                # key == arr[mid]
                key_index = mid
                if findMaxIndex:
                    left = mid + 1
                else:
                    right = mid - 1
        return key_index

    result = [-1, -1]
    result[0] = binary_search(arr, key, False)
    if result[0] != -1:
        result[1] = binary_search(arr, key, True)
    return result


assert find_range([4, 6, 6, 6, 9], 6) == [1, 3]
assert find_range([1, 3, 8, 10, 15], 10) == [3, 3]
assert find_range([1, 3, 8, 10, 15], 12) == [-1, -1]


def search_next_letter(letters: List[str], key: int) -> int:
    """
    Given an array of lowercase letters sorted in ascending order,
    find the smallest letter in the given array greater than a given key.

    Assume the given array is a circular list, which means that the last letter is assumed to be connected with the first letter. This also means that the smallest letter in the given array is greater than the last letter of the array and is also the first letter of the array."""
    left = 0
    right = len(letters) - 1

    while right >= left:

        mid = left + (right - left) // 2

        if key < letters[mid]:
            right = mid - 1
        else:  # key > letters[mid]
            left = mid + 1
    # since the loop is running until right < left,
    # then end will be left == right+1
    return letters[left % len(letters)]


assert search_next_letter(["a", "c", "f", "h"], "f") == "h"
assert search_next_letter(["a", "c", "f", "h"], "b") == "c"
assert search_next_letter(["a", "c", "f", "h"], "m") == "a"


def search_floor_of_a_number(arr: List[int], key: int) -> int:
    """
    Given an array of numbers sorted in ascending order, find the floor of a
    given number 'key'.
    The floor of the 'key' will be the biggest element in the given array
    smaller than or equal to the 'key'.
    """
    n = len(arr)
    # if key is greater than biggest element
    if key < arr[0]:
        return -1

    left = 0
    right = n - 1
    while left <= right:
        mid = left + (right - left) // 2
        if key > arr[mid]:
            left = mid + 1
        elif key < arr[mid]:
            right = mid - 1
        else:
            return mid

    return right


def search_ceiling_of_a_number(arr: List[int], key: int) -> int:
    """
    Given an array of numbers sorted in an ascending order,
    find the ceiling of a given number 'key'.
    The ceiling of the 'key' will be the smallest element in the given
    array greater than or equal to the 'key'.
    """
    n = len(arr)
    # if key is smaller than smallest element
    if key > arr[n - 1]:
        return -1

    left = 0
    right = n - 1
    while left <= right:
        mid = left + (right - left) // 2
        if key < arr[mid]:
            right = mid - 1
        elif key > arr[mid]:
            left = mid + 1
        else:
            return mid

    return left


assert search_ceiling_of_a_number([4, 6, 10], 6) == 1
assert search_ceiling_of_a_number([1, 3, 8, 10, 15], 12) == 4
assert search_ceiling_of_a_number([4, 6, 10], 17) == -1
assert search_ceiling_of_a_number([4, 6, 10], -1) == 0


def binary_search(arr: List[int], key: int) -> int:
    """
    Given a sorted array of numbers, find if a given number 'key' is present in
     the array. Though we know that the array is sorted, we don't know if
     it's sorted in ascending or descending order.
     You should assume that the array can have duplicates.
    """
    arr.sort()

    start = 0
    end = len(arr) - 1

    while start <= end:
        mid = start + (end - start) // 2

        if key == arr[mid]:
            return mid

        if key < arr[mid]:
            end = mid - 1
        else:
            start = mid + 1

    return -1


assert binary_search([4, 6, 10], 10) == 2
assert binary_search([1, 2, 3, 4, 5, 6, 7], 5) == 4
#assert binary_search([10, 6, 4], 10) == 0
#assert binary_search([10, 6, 4], 4) == 