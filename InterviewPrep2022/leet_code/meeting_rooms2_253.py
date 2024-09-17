from typing import List
from heapq import *


def minMeetingRooms(intervals: List[List[int]]) -> int:
    """
    Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.
    """
    intervals.sort(key=lambda x: x[0])

    heap = []
    for interval in intervals:
        if heap and interval[0] >= heap[0]:
            # means the current meeting room is not available
            heappop(heap)
        else:
            # means the current meeting room is available
            heappush(heap, interval[1])
    return len(heap)


intervals = [[0, 30], [5, 10], [15, 20]]
assert minMeetingRooms(intervals) == 2
intervals = [[7, 10], [2, 4]]
assert minMeetingRooms(intervals) == 1
