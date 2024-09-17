# Implementation of QuickSort
def quickSort(arr: list[int], s: int, e: int) -> list[int]:
    if e - s + 1 <= 1:
        return arr

    pivot = arr[e]
    left = s  # pointer for left side

    # Partition: elements smaller than pivot on left side
    for i in range(s, e):
        if arr[i] < pivot:
            # Swap elements
            arr[left], arr[i] = arr[i], arr[left]
            # tmp = arr[left]
            # arr[left] = arr[i]
            # arr[i] = tmp
            left += 1

    # Move pivot in-between left & right sides
    arr[e] = arr[left]
    arr[left] = pivot

    # Quick sort left side
    quickSort(arr, s, left - 1)

    # Quick sort right side
    quickSort(arr, left + 1, e)

    return arr


# Test the function
arr = [12, 11, 13, 5, 6]
print("Sorted array is:", quickSort(arr, 0, len(arr) - 1))

# Test with duplicate elements
arr = [12, 11, 13, 11, 5, 6, 11]
print("Sorted array with duplicate elements is:", quickSort(arr, 0, len(arr) - 1))
