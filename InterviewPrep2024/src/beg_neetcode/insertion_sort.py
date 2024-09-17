"""
Intsertion sort algorithm
"""


def insertionSort(arr):
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        # set index to current position minus 1
        j = i - 1
        # iterate backwards from current element to beginning of array
        # compare each element with the key and shift larger elements to the right
        # this mean thats jth element is smaller than the previous one
        while j >= 0 and arr[j + 1] < arr[j]:
            # swap elements
            # repeat this process until j is less than or equal to 0 or the current element is smaller than the previous one
            tmp = arr[j + 1]
            arr[j + 1] = arr[j]
            arr[j] = tmp
            j -= 1
    return arr


# test the function
arr = [12, 11, 13, 5, 6]
insertionSort(arr)
print("Sorted array is:")
for i in range(len(arr)):
    print("%d" % arr[i]),
