from typing import List
from collections import Counter


def minWindow(s: str, t: str) -> str:
    """Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string ''.

    The testcases will be generated such that the answer is unique.
    A substring is a contiguous sequence of characters within the string.
    """

    s_count = Counter()
    t_count = Counter(t)
    start = 0
    end = 0
    min_len = float("inf")
    min_start = 0
    min_end = 0
    count = 0

    while end < len(s):
        # Add one letter on the right
        s_count[s[end]] += 1
        # Remove one letter on the left
        if s_count[s[start]] == 1:
            del s_count[s[start]]
        else:
            s_count[s[start]] -= 1
        # If sliding window counter == reference counter, then we found an anagram
        if s_count == t_count:
            count += 1
            # Update the min_len and min_start
            while s_count[s[start]] > 0:
                s_count[s[start]] -= 1
                start += 1
            if end - start + 1 < min_len:
                min_len = end - start + 1
                min_start = start
                min_end = end
        end += 1
