from typing import List

# python implementation of intsertion sort


def insertion_sort(arr:List[int]) -> List[int]:
    """_summary_

    Args:
        arr (List[int]): _description_

    Returns:
        List[int]: _description_
    """
    for i,key in enumerate(arr):

        # move elements that are greater than key to the
        j: int = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j-=1
        arr[j+1] = key

# testing the implementation
arr: List[int]=[12,11,13,5,6]
# done in place
insertion_sort(arr)
assert arr == [5,6,11,12,13]
#501 ns ± 1.88 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
