"""
Heap Sort

https://brilliant.org/wiki/heap-sort/
"""

from typing import List
import heapq


def max_heapify(arr: List[int], heap_size: int, i: int):
    l = 2 * i + 1
    r = 2 * i + 2
    largest = i
    if l < heap_size and arr[l] > arr[largest]:
        largest = l
    if r < heap_size and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        max_heapify(arr, heap_size, largest)

    return arr


def build_heap(arr: List[int]):
    for i in range(len(arr) // 2, -1, -1):
        max_heapify(arr, len(arr), i)
    return arr


def heapsort(arr: List[int]) -> List[int]:
    heap_size = len(arr)
    build_heap(arr)
    print(arr)
    for i in range(heap_size - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heap_size -= 1
        max_heapify(arr, heap_size, 0)

    return arr


def heapq_sort(arr: List[int]) -> List[int]:
    out = [n for n in arr]
    heapq.heapify(out)
    return [heapq.heappop(out) for _ in range(len(out))]


if __name__ == "__main__":
    arr = [100, 19, 36, 6, 3, 25, 1, 2, 7]
    out = heapsort(arr)
    print(out)
    out = heapq_sort(arr)
    print(out)
