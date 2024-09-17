"""
Python bucket sort implementation
"""


def bucketSort(arr: list[int]) -> None:
    # Find the maximum element in the array
    max_val = max(arr)

    # Create a bucket array of size max_val + 1
    buckets = [[] for _ in range(max_val + 1)]

    # Distribute elements into respective buckets
    for num in arr:
        buckets[num].append(num)

    # Concatenate elements from all buckets into a sorted array
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    # Copy sorted array back to original array
    arr[:] = sorted_arr


# Test the function using assertions
assert bucketSort([4, 2, 2, 8, 3, 3, 1]) == [1, 2, 2, 3, 3, 4, 8]
assert bucketSort([10, 7, 8, 9, 1, 5]) == [1, 5, 7, 8, 9, 10]
print("All tests passed!")
