"""
Phone Screen Interview Question
from Company T

Prior hide SQL and this one was given to find the unique minutes watched in overlapping
tuple pairs.
"""

from typing import List, Tuple


def unique_minutes_watched(watch_times: List[Tuple[int, int]]) -> int:
    if not watch_times:
        return 0

    # Sort intervals based on start times
    watch_times.sort(key=lambda x: x[0])

    total_time = 0
    current_start, current_end = watch_times[0]

    for start, end in watch_times[1:]:
        # no gap
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            # gap between current inteval and previous interval so need to add to total time
            total_time += current_end - current_start
            current_start, current_end = start, end

    # need to add difference here because the last interval is not included in the loop
    total_time += current_end - current_start

    return total_time


# Test cases
print(unique_minutes_watched([(0, 15), (10, 25)]))  # Expected: 26
print(unique_minutes_watched([(1, 5), (3, 7), (8, 10)]))  # Expected: 10
print(unique_minutes_watched([(1, 4), (2, 3), (5, 7), (6, 8)]))  # Expected: 8
print(unique_minutes_watched([(1, 10), (2, 6), (3, 5), (7, 9)]))  # Expected: 10
