def firstIndex(arr: list, target: int, idx: int) -> int:
    # base case 1
    if len(arr) == idx:
        return -1
    # base case 2
    if arr[idx] == target:
        return idx
    # recursive case
    return firstIndex(arr, target, idx + 1)


def firstIndex_iterative(arr: list, target: int, idx: int) -> int:
    for i in range(idx, len(arr)):
        if arr[i] == target:
            return i
    return -1


assert firstIndex([9, 8, 1, 8, 1, 7], 1, 0) == 2
assert firstIndex([9, 8, 1, 8, 1, 7], 8, 0) == 1
assert firstIndex([9, 8, 1, 8, 1, 7], 7, 0) == 5
