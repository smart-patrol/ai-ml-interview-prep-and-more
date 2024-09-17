from typing import List
from collections import defaultdict, Counter
from collections import deque


def knightDialer(N: int) -> int:
    """
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    if N == 1:
        return 10
    chessmap = {1: [4, 2], 2: [1, 3], 3: [4, 2], 4: [3, 1, 0], 0: [4, 4]}
    table = [1] * 5
    for i in range(2, N + 1):
        tmp = [0] * 5
        for j in range(5):
            for k in chessmap[j]:
                tmp[j] += table[k]
        table = tmp
    return (sum(table) * 2 - table[0]) % (10 ** 9 + 7)


def subarraysDivByK(nums: List[int], k: int) -> int:
    """
    Given an integer array nums and an integer k, return the number of non-empty subarrays that have a sum divisible by k.
    A subarray is a contiguous part of an array.
    """
    res = 0
    prefix = 0
    count = [1] + [0] * k
    for a in nums:
        prefix = (prefix + a) % k
        res += count[prefix]
        count[prefix] += 1
    return res


def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Time Complexity: O(N^2)
    Space Complexity: O(N+M)
    """

    def justify(line, width, maxWidth):
        if len(line) == 1:
            return line[0] + " " * (maxWidth - width)
        else:
            spaces = maxWidth - width
            locations = len(line) - 1
            assign = locations * [spaces // locations]
            for i in range(spaces % locations):
                assign[i] += 1
            s = ""
            for i in range(locations):
                s += line[i] + assign[i] * " "
            s += line[-1]
            return s

    answer = []
    line, width = [], 0
    for w in words:
        if width + len(w) + len(line) <= maxWidth:
            line.append(w)
            width += len(w)
        else:
            answer.append(justify(line, width, maxWidth))
            line, width = [w], len(w)
    answer.append(" ".join(line) + (maxWidth - width - len(line) + 1) * " ")
    return answer


def frequencySort(nums: List[int]) -> List[int]:
    """
    Time O(NlogN)
    Space O(N)
    """
    count = Counter(nums)
    return sorted(nums, key=lambda x: (count[x], -x))


from collections import OrderedDict

# TC: O(1)
# SC: O(capcity)
class LRUCache(OrderedDict):
    def __init__(self, capacity):
        self.capacity = capacity

    def get(self, key):
        if key not in self:
            return -1

        self.move_to_end(key)
        return self[key]

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)


def findItinerary(self, tickets: List[str]) -> List[str]:
    """
    Greedy graph traversal with backtracking
    TC: O(E^d) with E number of flights and d is the max number flights from airport
    SC: O(V + E) with V number of airports and E number of flights
    """

    def backtracking(self, origin, route):
        if len(route) == self.flights + 1:
            self.result = route
            return True

        for i, nextDest in enumerate(self.flightMap[origin]):
            if not self.visitBitmap[origin][i]:
                # mark the visit before the next recursion
                self.visitBitmap[origin][i] = True
                ret = self.backtracking(nextDest, route + [nextDest])
                self.visitBitmap[origin][i] = False
                if ret:
                    return True

        return False

    self.flightMap = defaultdict(list)
    for ticket in tickets:
        origin, dest = ticket[0], ticket[1]
        self.flightMap[origin].append(dest)

    self.visitBitmap = {}

    # sort the itinerary based on the lexical order
    for origin, itinerary in self.flightMap.items():
        # Note that we could have multiple identical flights, i.e. same origin and destination.
        itinerary.sort()
        self.visitBitmap[origin] = [False] * len(itinerary)

    self.flights = len(tickets)
    self.result = []
    route = ["JFK"]
    self.backtracking("JFK", route)

    return self.result


def maxSlidingWindow(nums: "List[int]", k: "int") -> "List[int]":
    """
    You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

    Return the max sliding window.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    # base cases
    n = len(nums)
    if n * k == 0:
        return []
    if k == 1:
        return nums

    def clean_deque(i):
        # remove indexes of elements not from sliding window
        if deq and deq[0] == i - k:
            deq.popleft()

        # remove from deq indexes of all elements
        # which are smaller than current element nums[i]
        while deq and nums[i] > nums[deq[-1]]:
            deq.pop()

    # init deque and output
    deq = deque()
    max_idx = 0
    for i in range(k):
        clean_deque(i)
        deq.append(i)
        # compute max in nums[:k]
        if nums[i] > nums[max_idx]:
            max_idx = i
    output = [nums[max_idx]]

    # build output
    for i in range(k, n):
        clean_deque(i)
        deq.append(i)
        output.append(nums[deq[0]])
    return output


def numPairsDivisibleBy60(time: List[int]) -> int:
    # O(N) and O(1)
    remainders = defaultdict(int)
    ret = 0
    for t in time:
        if t % 60 == 0:  # check if a%60==0 && b%60==0
            ret += remainders[0]
        else:  # check if a%60+b%60==60
            ret += remainders[60 - t % 60]
        remainders[t % 60] += 1  # remember to update the remainders
    return ret


# DFS search
# TC: O(V+E)
# SC: O(V+E)
from collections import defaultdict


class Solution:

    WHITE = 1
    GRAY = 2
    BLACK = 3

    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """

        # Create the adjacency list representation of the graph
        adj_list = defaultdict(list)

        # A pair [a, b] in the input represents edge from b --> a
        for dest, src in prerequisites:
            adj_list[src].append(dest)

        topological_sorted_order = []
        is_possible = True

        # By default all vertces are WHITE
        color = {k: Solution.WHITE for k in range(numCourses)}

        def dfs(node):
            nonlocal is_possible

            # Don't recurse further if we found a cycle already
            if not is_possible:
                return

            # Start the recursion
            color[node] = Solution.GRAY

            # Traverse on neighboring vertices
            if node in adj_list:
                for neighbor in adj_list[node]:
                    if color[neighbor] == Solution.WHITE:
                        dfs(neighbor)
                    elif color[neighbor] == Solution.GRAY:
                        # An edge to a GRAY vertex represents a cycle
                        is_possible = False

            # Recursion ends. We mark it as black
            color[node] = Solution.BLACK
            topological_sorted_order.append(node)

        for vertex in range(numCourses):
            # If the node is unprocessed, then call dfs on it.
            if color[vertex] == Solution.WHITE:
                dfs(vertex)

        return topological_sorted_order[::-1] if is_possible else []
