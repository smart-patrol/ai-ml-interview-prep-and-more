from typing import List


def partition(arr:List[int], l:int, r:int) -> int:
    """_summary_

    Args:
        arr (List[int]): _description_
        l (int): _description_
        r (int): _description_

    Returns:
        int: _description_
    """
    left,right =l,r
    if l!=r and l < r:
        pivot = arr[l]
        left = left+1

        while left <= right:
            if arr[right] < pivot and arr[left] > pivot:
                arr[left], arr[right] = arr[right], arr[left]
            if not arr[left] > pivot:
                left+=1
            if not arr[right] < pivot:
                right-=1
    arr[l],arr[right] = arr[right], arr[l]
    return right

def quick_sort(arr:List[int], left:int ,right:int) -> None:
    if left < right:
        pi:int = partition(arr, left, right)
        quick_sort(arr, left, pi-1)
        quick_sort(arr,pi+1, right)


array: List[int] = [ 1, 7, 8, 9, 1, 2]
quick_sort(array, 0, len(array) - 1)
assert array[0] == 1
assert array[-1] == 9
#2.12 µs ± 4.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
