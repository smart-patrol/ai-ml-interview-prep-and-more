from typing import List
import bisect
from functools import reduce
from collections import deque



def find_max_sum_sublist(arr:List[int]) -> List[int]:
    """
    Given an unsorted array A, return the maximum sum of the sub array(contiguous elements) from array A,
    for which the sum of the elements is maximum.
    Kadane's algorithm is used to find the maximum sub array.
    """
    # TC: O(n)
    # SC: O(1)
    if len(arr) < 1:
        return None
    
    curr_max = arr[0]
    global_max = arr[0]
    length = len(arr)
    for i in range(1, length):
        if curr_max < 0:
            curr_max = arr[i]
        else:
            curr_max += arr[i]
        if global_max < curr_max:
            global_max = curr_max
    
    return global_max

lst = [-4, 2, -5, 1, 2, 3, 6, -5, 1]
assert find_max_sum_sublist(lst) == 12


    

            


def max_min(lst:List[int]) -> List[int]:
    """
    Implement a function called max_min(lst) which will re-arrange the elements of a sorted list such that the 0th index will have the largest number, the 1st index will have the smallest, and the 2nd index will have second-largest, and so on. In other words, all the even-numbered indices will have the largest numbers in the list in descending order and the odd-numbered indices will have the smallest numbers in ascending order.
    """
    # TC: O(n)
    # SC: O(1)
    # Return empty list for empty list
    if (len(lst) is 0):
        return []

    maxIdx = len(lst) - 1  # max index
    minIdx = 0  # first index
    maxElem = lst[-1] + 1  # Max element
    # traverse the list
    for i in range(len(lst)):
        # even number means max element to append
        if i % 2 == 0:
            lst[i] += (lst[maxIdx] % maxElem) * maxElem
            maxIdx -= 1
        # odd number means min number
        else:
            lst[i] += (lst[minIdx] % maxElem) * maxElem
            minIdx += 1

    for i in range(len(lst)):
        lst[i] = lst[i] // maxElem
    return lst


assert max_min([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == [9, 0, 8, 1, 7, 2, 6, 3, 5, 4]


def max_min2(arr:List[int]) -> List[int]:
    # TC: O(n)
    # SC: O(n)
    result = []
    for i in range(len(arr)//2):
        # append the last element
        result.append(arr[-(i+1)])
        # append the current element
        result.append(arr[i])
    if len(arr) %2 == 1:
        # if middle value append
        result.append(arr[len(arr)//2])
    return result

assert max_min([1, 2, 3, 4, 5, 6]) == [6, 5, 4, 3, 2, 1]

def max_min3(lst:List[int]) -> List[int]:
    q = deque(lst)

    out = []
    i=1
    j=len(lst)

    while j >= i and len(q) > 0:
        if i % 2 == 0:
            x = q.popleft()
            out.append(x)
        else:
            y = q.pop()
            out.append(y)
        i += 1
    
    return out


assert max_min2([-10, -1, 1, 1, 1, 1]) == [-10, -1, 1, 1, 1, 1]


def rearrange(arr:List[int]) -> List[int]:
    # pythonic way
    return [i for i in arr if i < 0] + [i for i in arr if i >= 0]

def rearrange2(arr:List[int]) -> List[int]:
    # TC: O(n)
    # SC: O(n)
    left_pos = 0

    for i in range(len(arr)):
        if arr[i] < 0:
            if i != left_pos:
                # swap
                arr[i], arr[left_pos] = arr[left_pos], arr[i]
            # update left_pos
            left_pos += 1
    return arr

assert rearrange2([10, -1, 20, 4, 5, -9, -6]) == [-1, -9, -6, 4, 5, 10, 20]
assert rearrange([10, -1, 20, 4, 5, -9, -6]) == [-1, -9, -6, 10, 20, 4, 5]

def rearrange_list2(lst:List[int]) -> List[int]:
    """
    Implement a function rearrange(lst) which rearranges the elements such that all the negative elements appear on the left and positive elements appear at the right of the list. Note that it is not necessary to maintain the sorted order of the input list.
    """
    # TC: O(n)
    # SC: O(n)
    # get all negative numbers
    negative = []
    positive = []
    for n in lst:
        if n < 0:
            negative.append(n)
        else:
            positive.append(n)
    # merge negative and positive
    return negative + positive


def right_rotate(arr:List[int], k:int) -> List[int]:
    # TC: O(k)
    # SC: O(N)
    if len(arr) == 0:
        k = 0
    else:
        k = k % len(arr)
    return arr[-k:] + arr[:-k]

assert right_rotate([10,20,30,40,50], abs(3)) == [30,40,50,10,20]



def right_rotate2(arr:List[int], k:int)->List[int]:
    """
    Implement a function right_rotate(lst, k) which will rotate the given list by k. This means that the right-most elements will appear at the left-most position in the list and so on. You only have to rotate the list by one element at a time.
    """
    # TC: O(n)
    # SC: O(N)
    n = len(arr)
    return arr[n-k:] + arr[:n-k]

def find_second_maximum(arr: List[int]) -> int:
    """find the 2nd maximum element in a given array."""
    # TC: O(n)
    # SC: O(1)
    if len(arr) <= 1:
        return None
    
    max_no = second_max_no = float('inf')
    for n in arr:
        # if n is greater than max_no, then update max_no
        if n > max_no:
            second_max_no = max_no
            max_no = n
        elif n > second_max_no and arr != max_no:
            second_max_no = n
    if second_max_no == float('inf'):
        return None
    else:
        return second_max_no

assert find_second_maximum([9,2,3,6]) == 6

def find_minimum(arr:List[int]) -> int:
    """Find the minimum element in a given array."""
    # TC: O(n)
    # SC: O(1)
    if len(arr) <= 0:
        return None
    mininum = arr[0]
    for n in arr:
        if n < mininum:
            mininum = n
    return mininum

assert find_minimum([9,2,3,6]) == 2

def find_product(arr:List[int]) -> List[int]:
    # TC: O(n)
    # SC: O(1)
    # get product start from the left
    left = 1
    product = []
    for ele in arr:
        product.append(left)
        left = left * ele
        print(left)
    # get product from right
    right = 1
    for i in range(len(arr)-1, -1, -1):
        print(right)
        product[i] = product[i] * right
        right = right * arr[i]

    return product

assert find_product([0,1,2,3]) == [6, 0, 0, 0]


def find_product2(lst:List[int]) -> List[int]:
    """Implement a function, find_product(lst), which modifies a list so that each index has a product of all the numbers present in the list except the number stored at that index."""
    output = []
    for i in range(len(lst)):
        output.append(reduce(lambda x,y: x*y, lst[:i] + lst[i+1:]))
    
    return output


assert find_product2([1,2,3,4]) == [24, 12, 8, 6]
assert find_product2([2, 5, 9, 3, 6]) == [810, 324, 180, 540, 270]

def find_sum(arr: List[int], k: int) -> List[int]:
    # TC: O(n)
    # SC: O(1)
    lookup = {}
    for n in arr:
        if n in lookup:
            return [lookup[n], n]
        complement = k - n
        lookup[complement] = n

def find_sum2(arr: List[int], k: int) -> List[int]:
    """binary search version"""
    # TC: O(nlog(n))
    # SC: O(1)
    arr.sort()
    for j in range(len(arr)):
        index = bisect.bisect(arr, k - arr[j])
        if index is not -1 and index != j:
            return [arr[j], k - arr[j]]

assert find_sum([1, 5, 3], 2) == None
assert find_sum([1, 2, 3, 4], 5) == [1,4]


def merge_lists2(lst1:List[int], lst2:List[int]) -> List[int]:
    # TC: O(n + m)
    # SC: O(m)
    ind1=0
    ind2=0

    while ind1 < len(lst1) and ind2 < len(lst2):

        #If the current element of the first list is greater than the current element of the second list
        # , insert the current element of the second list in place of the current element of the 
        # first list and increment both index variables by
        if lst1[ind1] > lst2[ind2]:
            lst1.insert(ind1, lst2[ind2])
            ind1 += 1
            ind2 += 1
        else:
            ind1 += 1

    # append what is left from lst2
    if ind2 < len(lst2):
        lst1.extend(lst2[ind2:])

    return lst1

assert merge_lists2([4, 5, 6], [-2, -1, 0, 7]) ==[-2, -1, 0, 4, 5, 6, 7]




# Merge list1 and list2 and return resulted list
def merge_lists(lst1, lst2):
    # TC: O(n + m)
    # SC: O(n + m)
    index_arr1 = 0
    index_arr2 = 0
    index_result = 0
    result = []

    for i in range(len(lst1)+len(lst2)):
        result.append(i)
    # Traverse Both lists and insert smaller value from arr1 or arr2
    # into result list and then increment that lists index.
    # If a list is completely traversed, while other one is left then just
    # copy all the remaining elements into result list
    while (index_arr1 < len(lst1)) and (index_arr2 < len(lst2)):
        if (lst1[index_arr1] < lst2[index_arr2]):
            result[index_result] = lst1[index_arr1]
            index_result += 1
            index_arr1 += 1
        else:
            result[index_result] = lst2[index_arr2]
            index_result += 1
            index_arr2 += 1
    while (index_arr1 < len(lst1)):
        result[index_result] = lst1[index_arr1]
        index_result += 1
        index_arr1 += 1
    while (index_arr2 < len(lst2)):
        result[index_result] = lst2[index_arr2]
        index_result += 1
        index_arr2 += 1
    return result


assert merge_lists([4, 5, 6], [-2, -1, 0, 7]) == [-2, -1, 0, 4, 5, 6, 7]
