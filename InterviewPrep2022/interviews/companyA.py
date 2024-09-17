from collections import defaultdict
from typing import List

# https://leetcode.com/problems/time-based-key-value-store/
# Time Complexity: O(1) for set() and O(Log K) for get() where K = maximum length of values[] array among entries in the TimeMap
# Space Complexity: O(N), N = # of entries in the TimeMap
class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key, value, timestamp):
        self.store[key].append([timestamp, value])

    def get(self, key, timestamp):
        arr = self.store[key]
        n = len(arr)

        left = 0
        right = n

        while left < right:
            mid = (left + right) // 2
            if arr[mid][0] <= timestamp:
                left = mid + 1
            elif arr[mid][0] > timestamp:
                right = mid

        return "" if right == 0 else arr[right - 1][1]


# https://leetcode.com/problems/random-pick-with-weight/solution/

from random import random

# Time: O(n) + O(log N) - construction of prefix sums and pick index binar search
# Space: O(n) + O(1) - construction of prefix sums and pick index which is constant
class Picker:
    def __init__(self, w: List[int]):
        self.prefix_sums = []
        prefix_sum = 0
        for weight in w:
            prefix_sum += weight
            self.prefix_sums.append(prefix_sum)
        self.total_sum = prefix_sum

    def pickIndex(self) -> int:
        target = self.total_sum * random()
        # run a binary search to find the target zone
        low, high = 0, len(self.prefix_sums)
        while low < high:
            mid = low + (high - low) // 2
            if target > self.prefix_sums[mid]:
                low = mid + 1
            else:
                high = mid
        return low


def groupAnagrams(strs):
    ans = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        ans[tuple(count)].append(s)
    return ans.values()
