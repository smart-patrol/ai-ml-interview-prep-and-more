from typing import List
import heapq
from MinHeap import MinHeap
from MaxHeap import MaxHeap


def find_maximum_capital(
    capital: List[int], profits: List[int], numberOfProjects: int, initialCapital: int
) -> int:
    """
    Given a set of investment projects with their respective profits, we need to find the most profitable projects.
    We are given an initial capital and are allowed to invest only in a fixed number of projects.
    Our goal is to choose projects that give us the maximum profit.
    Write a function that returns the maximum total capital after selecting the most profitable projects.

    We can start an investment project only when we have the required capital.
    Once a project is selected, we can assume that its profit has become our capital.
    """
    minCapitalHeap = []
    maxProfitHeap = []

    # insert all project capitals to a min-heap
    for i in range(0, len(profits)):
        heapq.heappush(minCapitalHeap, (capital[i], i))

    # let's try to find a total of 'numberOfProjects' best projects
    availableCapital = initialCapital
    for _ in range(numberOfProjects):
        # find all projects that can be selected within the available capital and insert them in a max-heap
        while minCapitalHeap and minCapitalHeap[0][0] <= availableCapital:
            capital, i = heapq.heappop(minCapitalHeap)
            heapq.heappush(maxProfitHeap, (-profits[i], i))
        # terminate if we are not able to find any project that can be completed within the available capital
        if not maxProfitHeap:
            break

        # select the project with the maximum profit
        availableCapital += -heapq.heappop(maxProfitHeap)[0]

    return availableCapital


assert find_maximum_capital([0, 1, 2], [1, 2, 3], 2, 1) == 6
assert find_maximum_capital([0, 1, 2, 3], [1, 2, 3, 5], 3, 0) == 8


class SlidingWindowMedian:
    def __init__(self):
        self.maxHeap, self.minHeap = [], []

    def find_sliding_window_median(self, nums, k):
        result = [0.0 for x in range(len(nums) - k + 1)]
        for i in range(0, len(nums)):
            if not self.maxHeap or nums[i] <= -self.maxHeap[0]:
                heappush(self.maxHeap, -nums[i])
            else:
                heappush(self.minHeap, nums[i])

            self.rebalance_heaps()

            if i - k + 1 >= 0:  # if we have at least 'k' elements in the sliding window
                # add the median to the the result array
                if len(self.maxHeap) == len(self.minHeap):
                    # we have even number of elements, take the average of middle two elements
                    result[i - k + 1] = -self.maxHeap[0] / 2.0 + self.minHeap[0] / 2.0
                else:  # because max-heap will have one more element than the min-heap
                    result[i - k + 1] = -self.maxHeap[0] / 1.0

                # remove the element going out of the sliding window
                elementToBeRemoved = nums[i - k + 1]
                if elementToBeRemoved <= -self.maxHeap[0]:
                    self.remove(self.maxHeap, -elementToBeRemoved)
                else:
                    self.remove(self.minHeap, elementToBeRemoved)

                self.rebalance_heaps()

        return result

    # removes an element from the heap keeping the heap property
    def remove(self, heap, element):
        ind = heap.index(element)  # find the element
        # move the element to the end and delete it
        heap[ind] = heap[-1]
        del heap[-1]
        # we can use heapify to readjust the elements but that would be O(N),
        # instead, we will adjust only one element which will O(logN)
        if ind < len(heap):
            heapq._siftup(heap, ind)
            heapq._siftdown(heap, 0, ind)

    def rebalance_heaps(self):
        # either both the heaps will have equal number of elements or max-heap will have
        # one more element than the min-heap
        if len(self.maxHeap) > len(self.minHeap) + 1:
            heappush(self.minHeap, -heappop(self.maxHeap))
        elif len(self.maxHeap) < len(self.minHeap):
            heappush(self.maxHeap, -heappop(self.minHeap))


def main():

    slidingWindowMedian = SlidingWindowMedian()
    result = slidingWindowMedian.find_sliding_window_median([1, 2, -1, 3, 5], 2)
    print("Sliding window medians are: " + str(result))

    slidingWindowMedian = SlidingWindowMedian()
    result = slidingWindowMedian.find_sliding_window_median([1, 2, -1, 3, 5], 3)
    print("Sliding window medians are: " + str(result))


def find_sliding_window_median(nums: List[int], k: int) -> List[float]:
    """
    Given an array of numbers and a number k, find the median of all the k sized sub-arrays (or windows) of the array.
    """
    result = []
    for i in range(0, len(nums)):
        if len(nums + 1) > i + k:
            result.append(find_median(nums[i : i + k]))

    return result


def find_median(nums: List[int]) -> float:
    if len(nums) % 2 == 0:
        return (nums[len(nums) // 2] + nums[len(nums) // 2 - 1]) / 2
    else:
        return nums[len(nums) // 2]


# The time complexity of creating a heap is O(n) and removing max is O(klogn)
# So the total time complexity is O(n + klogn) or O(klogn)


def findKLargest(lst: List[int], k: int) -> List[int]:
    """
    Implement a function findKLargest(lst,k) that takes an unsorted integer list as input and returns the k
    largest elements in the list using a Max Heap. The maxHeap class that was written in a previous lesson is prepended in this exercise so feel free to use it! Have a look at the illustration given for a clearer picture of the problem. Implement a function findKlargest() which takes a list as input and finds the k largest elements in the list using a Max-Heap. An illustration is also provided for your understanding.
    """
    heap = MaxHeap()
    heap.buildHeap(lst)
    out = [heap.removeMax() for _ in range(k)]
    return out


lst = [9, 4, 7, 1, -2, 6, 5]
k = 3
assert findKLargest(lst, k) == [9, 7, 6]


def findKLargest(lst, k):
    heapq.heapify(lst)
    return heapq.nlargest(k, lst)


# TC creating heap O(n) and removing is O(k log n)
# so TC O(n+k log n) or O(k log n)
def findKSmallest(lst: List[int], k: int) -> List[int]:
    heap = MinHeap()
    heap.buildHeap(lst)
    out = [heap.removeMin() for _ in range(k)]
    return out


lst = [9, 4, 7, 1, -2, 6, 5]
k = 3
assert findKSmallest(lst, k) == [-2, 1, 5]


def findKSmallest2(lst: List[int], k: int) -> List[int]:
    """
    Implement a function findKSmallest(lst,k) that takes an unsorted integer list as input and returns the "k" smallest elements in the list using a Heap. The minHeap class that was written in a previous lesson is prepended to this exercise so feel free to use it! Have a look at the illustration given for a clearer picture of the problem.
    """
    heapq.heapify(lst)
    return heapq.nsmallest(k, lst)


# TC O(n*log(n))
def minHeapify(heap, index):
    left = index * 2 + 1
    right = (index * 2) + 2
    smallest = index
    # check if left child exists and is less than smallest
    if len(heap) > left and heap[smallest] > heap[left]:
        smallest = left
    # check if right child exists and is less than smallest
    if len(heap) > right and heap[smallest] > heap[right]:
        smallest = right
    # check if current index is not the smallest
    if smallest != index:
        # swap current index value with smallest
        tmp = heap[smallest]
        heap[smallest] = heap[index]
        heap[index] = tmp
        # minHeapify the new node
        minHeapify(heap, smallest)
    return heap


def convertMax(maxHeap):
    # iterate from middle to first element
    # middle to first indices contain all parent nodes
    for i in range((len(maxHeap)) // 2, -1, -1):
        # call minHeapify on all parent nodes
        maxHeap = minHeapify(maxHeap, i)
    return maxHeap


def convertMax2(maxHeap):
    heapq.heapify(maxHeap)
    return maxHeap


maxHeap = [9, 4, 7, 1, -2, 6, 5]
convertMax(maxHeap) == [-2, 1, 4, 5, 6, 7, 9]
