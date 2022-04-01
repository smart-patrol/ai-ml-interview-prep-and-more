from typing import List
from heapq import *
import heapq


"""
    Design a class to calculate the median of a number stream. The class should have the following two methods:

    insertNum(int num): stores the number in the class
    findMedian(): returns the median of all numbers inserted in the class
    If the count of numbers inserted in the class is even, the median will be the average of the middle two numbers.
"""

class MedianOfAStream:

    max_heap = []
    min_heap = []
    
    def insert_num(self, num):
        if not self.max_heap or -self.max_heap[0] >= num:
            heappush(self.max_heap, -num)
        else:
            heappush(self.min_heap, num)

        # either both the heaps will have equal number of elements or max-heap will have one
        # more element than the min-heap
        if len(self.max_heap) >  len(self.min_heap) + 1:
            heappush(self.min_heap, -heappop(self.max_heap))
        elif len(self.max_heap) < len(self.min_heap):
            heappush(self.max_heap, -heappop(self.min_heap))
        
    def find_median(self):
        if len(self.max_heap) == len(self.min_heap):
            # we have even number of elements, take the average of middle
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            # we have odd number of elements, return the middle element
            return -self.max_heap[0]


def main():
  medianOfAStream = MedianOfAStream()
  medianOfAStream.insert_num(3)
  medianOfAStream.insert_num(1)
  print("The median is: " + str(medianOfAStream.find_median()))
  medianOfAStream.insert_num(5)
  print("The median is: " + str(medianOfAStream.find_median()))
  medianOfAStream.insert_num(4)
  print("The median is: " + str(medianOfAStream.find_median()))


#main()



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
                result[i - k + 1] = -self.maxHeap[0] / \
                                2.0 + self.minHeap[0] / 2.0
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


# slidingWindowMedian = SlidingWindowMedian()
# result = slidingWindowMedian.find_sliding_window_median(nums=[1, 2, -1, 3, 5],k= 2)
# assert result ==  [1.5, 0.5, 1.0, 4.0]
# slidingWindowMedian = SlidingWindowMedian()
# result = slidingWindowMedian.find_sliding_window_median(nums=[1, 2, -1, 3, 5], k=3)
# assert result ==  [1.0, 2.0, 3.0]
        

def find_maximum_capital(capital:List[int], profits:List[int],numberOfProjects:int, intialCapital:int):
    """
    Given a set of investment projects with their respective profits, we need to find the most profitable projects. We are given an initial capital and are allowed to invest only in a fixed number of projects. Our goal is to choose projects that give us the maximum profit. Write a function that returns the maximum total capital after selecting the most profitable projects.

    We can start an investment project only when we have the required capital. Once a project is selected, we can assume that its profit has become our capital.
    """
    min_heap = []
    max_heap = []

    # insert all the project capital to a min heap
    for i in range(0,len(profits)):
        heappush(min_heap, (capital[i], i))
    
    # find total number of best projects
    available_captial = intialCapital
    for _ in range(0,numberOfProjects):
        # finad all project that can be selected within captial available
        while min_heap and min_heap[0][0] <= available_captial:
            capital, i = heappop(min_heap)
            heappush(max_heap, (-profits[i], i))

        # terminate if we are not able to find any project
        if not max_heap:
            break  
    
    # select the project with the maximum profit
    available_captial += -heappop(max_heap)[0]

    return available_captial


#assert find_maximum_capital([0, 1, 2], [1, 2, 3], 2, 1) == 6
#assert find_maximum_capital([0, 1, 2, 3], [1, 2, 3, 5], 3, 0) == 8


class Interval:
  def __init__(self, start, end):
    self.start = start
    self.end = end


def find_next_interval(intervals):
    """
    Given an array of intervals, find the next interval of each interval. 
    In a list of intervals, for an interval i its next interval j will have
    the smallest start greater than or equal to the end of i.

    Write a function to return an array containing indices of the next interval of each input interval.
    If there is no next interval of a given interval, return -1. 
    It is given that none of the intervals have the same start point.
    """
    result =[-1 for x in range(len(intervals))]
    # sort the intervals by start time
    intervals.sort(key=lambda x : x.start)
    # find the next interval for each interval
    for i in range(0, len(intervals)):
        # find the next interval for the current interval
        for j in range(i+1, len(intervals)):
            if intervals[j].start >= intervals[i].end:
                result[i] = j
                break
    return result


assert find_next_interval([Interval(2, 3), Interval(3, 4), Interval(5, 6)]) == [1, 2, -1]
assert find_next_interval([Interval(3, 4), Interval(1, 5), Interval(4, 6)]) == [-1, 2, -1]