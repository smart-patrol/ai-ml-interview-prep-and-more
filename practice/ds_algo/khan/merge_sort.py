from typing import List

# left right, indexes and array
# base case is  left index is greater than equal to right, ie length 1 or 
0
# recursive cases
# take the floor or left and right divided by two to get mid
# call mergesort on left index and mid index
# call mergesort on mid index +1 and right index
# merge the array by left, mid, and right


def merge(arr:List[int]) -> List[int]:
    """
    merge arrays into single array
    """
    mid:int = len(arr)//2

    left_arr: List[int] = arr[:mid]
    right_arr: List[int] = arr[mid+1:]

    i:int  = 0
    j:int  = 0
    k:int = 0

    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            arr[k] = left_arr[i]
            i+=1
        else:
            arr[k] = right_arr[j]
            j+=1
        k+=1

    while i < len(left_arr):
        arr[k] = left_arr[i]
        i+=1
        k+=1
    while j < len(right_arr):
        arr[k] = right_arr[i]
        j+=1
        k+=1


def merge_sort(arr:List[int], left:int, right:int) -> List[int]:
    """_summary_

    Args:
        arr (List[int]): _description_
        left (int): _description_
        right (int): _description_

    Returns:
        List[int]: _description_
    """
    if left >= right:
        return
    else:
        mid: int = (left+right)//2

        merge_sort(arr, left, mid)
        merge_sort(arr,mid+1, right)
        merge(arr)


array: List[int] = [3, 7, 12, 14, 2, 6, 9, 11]
merge_sort(array)
assert array == [2, 3, 6, 7, 9, 11, 12, 14]

