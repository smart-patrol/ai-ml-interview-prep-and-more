def binary_search(arr: list, target: int) -> int:
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            high = mid - 1
        else:
            low = mid + 1


def binary_search_recursive(arr: list, target: int, low: int, high: int) -> int:
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, low, mid - 1)
    else:
        return binary_search_recursive(arr, target, mid + 1, high)


if __name__ == "__main__":
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target = 7
    print(binary_search(arr, target))
    print(binary_search_recursive(arr, target, 0, len(arr) - 1))

    arr = [2, 5, 6, 7, 8, 8, 9]
    target = 8
    print(binary_search(arr, target))
    print(binary_search_recursive(arr, target, 0, len(arr) - 1))
