from typing import List
from heapq import *

class Interval:
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def print_interval(self):
    print("[" + str(self.start) + ", " + str(self.end) + "]", end='')


def merge(intervals:'Interval') -> 'Interval':
    """Given a list of intervals, merge all the overlapping intervals to
     produce a list that has only mutually exclusive intervals."""

    if len(intervals) < 2:
        return intervals
    merged = []

    # sort the intervals by start time
    intervals.sort(key=lambda x: x.start)

    start = intervals[0].start
    end = intervals[0].end
    for i in range(1, len(intervals)):
        interval = intervals[i]
        if interval.start <= end: #overalping intervals, adjust end
            end = max(interval.end, end)
        else:  #non-overlapping intervals, add to merged and reset start and end
            merged.append(Interval(start, end))
            start = interval.start
            end = interval.end
    
    merged.append(Interval(start,end))
    return merged

# assert merge([Interval(1, 4), Interval(2, 5), Interval(7, 9)]) == [Interval(1, 5), Interval(7, 9)]
# assert merge([Interval(6, 7), Interval(2, 4), Interval(5, 9)]) == [Interval(2, 4), Interval(5, 9)] 
# assert merge([Interval(1, 4), Interval(2, 6), Interval(3, 5)]) == [Interval(1, 6)]


def merge_2(intervals:'Interval') -> 'Interval':
    """Given a set of intervals, find out if any two intervals overlap.
       Return the intervals that overlap.
    """
    if len(intervals) < 2:
        return intervals
    overlap = []
    intervals.sort(key=lambda x : x.start)

    for i in range(len(intervals)):
        interval = intervals[i]
        if interval.start <= interval.end:
            overlap.append(interval)
    
    return overlap

# assert merge2([Interval(1, 4), Interval(2, 5), Interval(7, 9)]) == [Interval(1, 4), Interval(2, 5)]

def insert_intervals(intervals:'Interval', new_interval:'Interval') -> 'Interval':
    """
    Given a list of non-overlapping intervals sorted by their start time,
    insert a given interval at the correct position and merge all necessary 
    intervals to produce a list that has only mutually exclusive intervals.
    """
    merged = []
    i, start, end  = 0, 0 , 1
    # skip (and add to output) all intervals that come before the 'new_interval'
    while i < len(intervals) and intervals[i][end] < new_interval[start]:
        merged.append(intervals[i])
        i += 1
    # merge all the intervals that overlap with 'new_interval'
    while i < len(intervals) and intervals[i][start] <= new_interval[end]:
        new_interval[start] = min(intervals[i][start], new_interval[start])
        new_interval[end] = max(intervals[i][end], new_interval[end])
        i += 1
    # insert the new interval
    merged.append(new_interval)

    # add all the remaining intervals to the output
    while i < len(intervals):
        merged.append(intervals[i])
        i+=1
    return merged

# assert insert([[1, 3], [5, 7], [8, 12]], [4, 6]) == [[1, 3], [4, 7], [8, 12]]
# assert insert([[1, 3], [5, 7], [8, 12]], [4, 10]) ==  [[1, 3], [4, 12]]
# assert insert([2, 3], [5, 7]], [1, 4])) == [[1, 4], [5, 7]]


def interval_intersection(intervals_a:List[List[int]], intervals_b: List[List[int]]) -> List[List[int]]:
    """
    Given two lists of intervals, find the intersection of these two lists. 
    Each list consists of disjoint intervals sorted on their start time.
    Return the intersections.
    """
    result = []
    i, j, start, end = 0, 0, 0, 1
    
    while i < len(intervals_a) and j < len(intervals_b):
        # check if intervals overlap and intervals_a[i]'s start time lies within the other intervals_b[j]
        a_overlaps_b = intervals_a[i][start] >= intervals_b[j][start] and \
                   intervals_a[i][start] <= intervals_b[j][end]
                   
        # check if intervals overlap and intervals_a[j]'s start time lies within the other intervals_b[i]
        b_overlaps_a = intervals_b[j][start] >= intervals_a[i][start] and \
            intervals_b[j][start] <= intervals_a[i][end]

        # store the the intersection part
        if (a_overlaps_b or b_overlaps_a):
            result.append([max(intervals_a[i][start], intervals_b[j][start]), min(
                intervals_a[i][end], intervals_b[j][end])])

        # move next from the interval which is finishing first
        if intervals_a[i][end] < intervals_b[j][end]:
            i += 1
        else:
            j += 1
            
    return result

arr1 = [[1, 3], [5, 6], [7, 9]]  
arr2 = [[2, 3], [5, 7]]
print(interval_intersection(arr1,arr2))

arr1=[[1, 3], [5, 7], [9, 12]]
arr2=[[5, 10]]
print(interval_intersection(arr1,arr2))


def can_attend_all_appointments(intervals:List[int]) -> bool:
    """
    Given an array of intervals representing 'N' appointments, find out if a person can attend all the appointments.
    """
    intervals.sort(key=lambda x: x[0])

    end_time = intervals[0][1]
    for interval in intervals:
        if interval[0] > end_time:
            end_time = interval[1]
        else:
            return False
    return True

intervals = [[1,4], [2,5], [7,9]]
assert can_attend_all_appointments(intervals) == False
intervals = [[6,7], [2,4], [8,12]]
#assert can_attend_all_appointments(intervals) == True
intervals = [[4,5], [2,3], [3,6]]
assert can_attend_all_appointments(intervals) == False


class Meeting:
  def __init__(self, start, end):
    self.start = start
    self.end = end

def __lt__(self, other):
    return self.end < other.end

def min_meeting_rooms(meetings:'Meeting') -> int:
    """
    Given a list of intervals representing the start and 
    end time of 'N' meetings, find the minimum number of rooms required to
    hold all the meetings.
    """
    # use a min heap to store the end time of the meetings
    # pop the meeting with the earliest end time
    # if the end time of the meeting is greater than the start time of the next meeting,
    # then the next meeting can be held in the same room
    # else, the next meeting requires a new room
    # repeat the above process until all the meetings are done
    # return the size of the heap
    # sort the meetings by start time
    meetings.sort(key=lambda x: x.start)

    minRooms = 0
    minHeap = []
    heapify(minHeap)
    for meeting in meetings:
        if minHeap and minHeap[0] <= meeting.start:
            heappop(minHeap)
        heappush(minHeap, meeting.end)
        minRooms = max(minRooms, len(minHeap))
    return minRooms

    
assert min_meeting_rooms([Meeting(1, 4), Meeting(2, 5), Meeting(7, 9)]) == 2
assert min_meeting_rooms([Meeting(6, 7), Meeting(2, 4), Meeting(8, 12)]) == 1   
assert min_meeting_rooms([Meeting(1, 4), Meeting(2, 3), Meeting(3, 6)]) == 2
assert min_meeting_rooms([Meeting(4, 5), Meeting(2, 3), Meeting(2, 4), Meeting(3, 5)]) == 2

class job:
  def __init__(self, start, end, cpu_load):
    self.start = start
    self.end = end
    self.cpu_load = cpu_load

def find_max_cpu_load(jobs:'jobs') -> int:
    """
    We are given a list of Jobs. Each job has a Start time, an End time, and a CPU 
    load when it is running. Our goal is to find the maximum CPU load at any time
    if all the jobs are running on the same machine.
    """
    jobs.sort(key = lambda x : x.start)
    
    start = jobs[0].start
    end = jobs[0].end
    load = jobs[0].cpu_load
    max_output = 0
    
    for i in range(1, len(jobs)):
        job = jobs[i]
        if job.start <= end:
            end = max(job.end, end)
            load += job.cpu_load
            max_output = max(max_output, load)
        else:
            start = job.start
            end = job.end
            load = job.cpu_load
    
    max_output = max(max_output, load)
    return max_output

assert find_max_cpu_load([job(1, 4, 3), job(2, 5, 4), job(7, 9, 6)]) == 7
assert find_max_cpu_load([job(6, 7, 10), job(2, 4, 11), job(8, 12, 15)]) == 15
assert find_max_cpu_load([job(1, 4, 2), job(2, 4, 1), job(3, 6, 5)]) == 8


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class EmployeeInterval:

    def __init__(self, interval, employeeIndex, intervalIndex):
        self.interval = interval  # interval representing employee's working hours
        # index of the list containing working hours of this employee
        self.employeeIndex = employeeIndex
        self.intervalIndex = intervalIndex  # index of the interval in the employee list

    def __lt__(self, other):
        # min heap based on meeting.end
        return self.interval.start < other.interval.start

def find_employee_free_time(schedule):
    """
    For 'K' employees, we are given a list of intervals representing the working 
    hours of each employee. Our goal is to find out if there is a free interval that
    is common to all employees. You can assume that each list of employee working
    hours is sorted on the start time.
    """
    if schedule is None:
        return []

    n = len(schedule)
    result, minHeap = [], []

    heapify(minHeap)

    for i in range(n):
        heappush(minHeap, EmployeeInterval(schedule[i][0], i, 0))

    previous_interval = minHeap[0].interval

    for i in range(1, n):
        queue_top = heappop(minHeap)
        # if previous interval is not overlapping with the current interval, insert a free interval
        if previous_interval.end < queue_top.interval.start:
            result.append(Interval(previous_interval.end, queue_top.interval.start))
            previous_interval = queue_top.interval
        else: #overlapping intervals, update the previous interval
            if previous_interval.end < queue_top.interval.end:
                previous_interval = queue_top.interval
        
        # if there are more intervals available in the current employee's list, push it to the heap
        employee_schedule = schedule[queue_top.employeeIndex]
        if len(employee_schedule) > queue_top.intervalIndex + 1:
            heappush(minHeap, EmployeeInterval(employee_schedule[queue_top.intervalIndex + 1], queue_top.employeeIndex, queue_top.intervalIndex + 1))

        return result

        
intervals = [[Interval(1, 3), Interval(5, 6)], [Interval(2, 3), Interval(6, 8)]]
output = find_employee_free_time(intervals) 
assert output[0].start == 3 and output[0].end == 5
assert len(output) == 1