from curses import window
from typing import List
from collections import Counter


def find_substring(s: str, pattern: str) -> List[int]:
    """
    Given a string and a pattern, find the smallest substring in the given string which has all the character occurrences of the given pattern.
    """
    window_start, matched, substr_start = 0, 0, 0
    char_frequency = {}
    min_length = float("inf")

    for chr in pattern:
        if chr not in char_frequency:
            char_frequency[chr] = 0
        char_frequency[chr] += 1

    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char in char_frequency:
            # Decrement the frequency of matched character
            char_frequency[right_char] -= 1
        if char_frequency[right_char] >= 0:
            matched += 1

        # shrink the window if we can
        while matched == len(pattern):
            if min_length > window_end - window_start + 1:
                min_length = window_end - window_start + 1
                substr_start = window_start

            left_char = s[window_start]
            window_start += 1
            if left_char in char_frequency:
                if char_frequency[left_char] == 0:
                    matched -= 1
                char_frequency[left_char] += 1

        if min_length > len(s):
            return ""
        return s[substr_start : substr_start + min_length]


assert find_substring("aabdec", "abc") == "abdec"
assert find_substring("aabdec", "abac") == "aabdec"
assert find_substring("adcad", "abc") == "adc"


def find_string_anagrams(s: str, pattern: str) -> List[int]:
    window_start, matched = 0, 0
    char_frequency = {}

    for chr in pattern:
        if chr not in char_frequency:
            char_frequency[chr] = 0
            char_frequency[chr] += 1

    result_indices = []
    # Our goal is to match all the characters from the 'char_frequency' with the current window
    # try to extend the range [window_start, window_end]
    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char in char_frequency:
            # Decrement the frequency of matched character
            char_frequency[right_char] -= 1
        if char_frequency[right_char] == 0:
            matched += 1

        if matched == len(char_frequency):  # Have we found an anagram?
            result_indices.append(window_start)

        # Shrink the sliding window
        if window_end >= len(pattern) - 1:
            left_char = s[window_start]
        window_start += 1
        if left_char in char_frequency:
            if char_frequency[left_char] == 0:
                matched -= (
                    1  # Before putting the character back, decrement the matched count
                )
                char_frequency[left_char] += 1  # Put the character back

    return result_indices


assert find_string_anagrams("ppqp", "pq")
assert find_string_anagrams("abbcabc", "abc")


def find_string_anagrams2(s: str, pattern: str) -> List[int]:
    """
    Given a string and a pattern, find all anagrams of the pattern in the given string.

    Every anagram is a permutation of a string. As we know, when we are not allowed to repeat characters while finding permutations of a string, we get N! permutations (or anagrams) of a string having NN characters.
    """
    result_indexes = []
    pattern_counter = Counter(pattern)
    window_start = 0

    for window_end in range(len(s) + 1):
        s_count = Counter(s[window_start:window_end])

        if not (s_count - pattern_counter):
            result_indexes.append(window_start)
        window_start += 1

    return result_indexes


def find_permutation(s: str, pattern: str) -> bool:
    window_start, matched = 0, 0
    char_frequency = {}

    for chr in pattern:
        if chr not in char_frequency:
            char_frequency[chr] = 0
        char_frequency[chr] += 1

    # our goal is to match all the characters from the 'char_frequency' with the current window
    # # try to extend the range [window_start, window_end]
    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char in char_frequency:
            # decrement the frequency of the character
            char_frequency[right_char] -= 1
            if char_frequency[right_char] == 0:
                matched += 1
        if matched == len(char_frequency):
            return True
        # shrink the window
        if window_end >= len(pattern) - 1:
            left_char = s[window_start]
            window_start += 1
            if left_char in char_frequency:
                if char_frequency[left_char] == 0:
                    matched -= 1
                char_frequency[left_char] += 1
    return False


assert find_permutation("oidbcaf", "abc")
assert find_permutation("odicf", "dc")
assert find_permutation("aaacb", "abc")


def find_permutation2(s: str, pattern: str) -> bool:
    """
    Given a string and a pattern, find out if the string contains any permutation of the pattern.

    Permutation is defined as the re-arranging of the characters of the string. For example, "abc" has the following six permutations
    """
    return not Counter(pattern) - Counter(s)


def length_of_longest_substring(arr: List[int], k: int) -> int:
    window_start, max_length, max_ones_count = 0, 0, 0

    for window_end in range(len(arr)):
        if arr[window_end] == 1:
            max_ones_count += 1
    # Current window size is from window_start to window_end, overall we have a maximum of 1s
    # repeating 'max_ones_count' times, this means we can have a window with 'max_ones_count' 1s
    # and the remaining are 0s which should replace with 1s.
    # now, if the remaining 0s are more than 'k', it is the time to shrink the window as we
    # are not allowed to replace more than 'k' 0s
    if (window_end - window_start + 1 - max_ones_count) > k:
        if arr[window_start] == 1:
            max_ones_count -= 1
        window_start += 1
    max_length = max(max_length, window_end - window_start + 1)

    return max_length


assert length_of_longest_substring([0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1], 2) == 6
assert length_of_longest_substring([0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1], 3) == 9


def length_of_longest_substring2(arr: List[int], k: int) -> int:
    """
    Given an array containing 0s and 1s, if you are allowed to replace no more than k 0s with 1s, find the length of the longest contiguous subarray having all 1s.
    """
    window_start = 0
    res = 0

    # look over the array
    for window_end in range(len(arr)):
        cnt = Counter(arr[window_start:window_end])
        # if we have more than 0s than k, shrink the window
        if cnt[0] > k:
            window_start += 1
        else:
            # we add plus one to include the last element
            res = max(res, window_end - window_start + 1)
    return res


def length_of_longest_substring(s: str, k: int) -> int:
    """
    Given a string with lowercase letters only, if you are allowed to replace no more than k letters with any letter, find the length of the longest substring having the same letters after replacement.
    """
    window_start, max_length, max_repeat_letter_count = 0, 0, 0
    frequency_map = {}

    for window_end in range(len(s)):
        right_char = s[window_end]
        if right_char not in frequency_map:
            frequency_map[right_char] = 0
        frequency_map[right_char] += 1
        max_repeat_letter_count = max(
            max_repeat_letter_count, frequency_map[right_char]
        )

        if window_end - window_start + 1 - max_repeat_letter_count > k:
            left_char = s[window_start]
            frequency_map[left_char] -= 1
            window_start += 1

        max_length = max(max_length, window_end - window_start + 1)
    return max_length


assert length_of_longest_substring("aabccbb", 2) == 5
assert length_of_longest_substring("abbcb", 1) == 4
assert length_of_longest_substring("abccde", 1) == 3


def length_of_longest_substring2(s: str, k: int) -> int:
    res = 0
    cnt = Counter()
    left = 0

    for right in range(len(s)):
        cnt[s[right]] += 1

        while (right - left + 1) - max(cnt.values()) > k:
            cnt[s[left]] -= 1
            left += 1

        res = max(res, right - left + 1)

    return res


def non_repeat_substring(s: str) -> int:
    """
    Given a string, find the length of the longest substring, which has all distinct characters.
    """
    window_start = 0
    max_length = 0
    char_index_map = {}

    for window_end in range(len(s)):
        right_char = s[window_end]
        # if the map already contains teh right_char, shrink the window
        # from the beginning until we only have one instance of the char
        if right_char in char_index_map:
            window_start = max(window_start, char_index_map[right_char] + 1)
        # insert right_char into map
        char_index_map[right_char] = window_end

        max_length = max(max_length, window_end - window_start + 1)

    return max_length


assert non_repeat_substring("aabccbb") == 3
assert non_repeat_substring("abbbb") == 2
assert non_repeat_substring("abccde") == 3


def non_repeat_substring2(s: str) -> int:
    window_start = 0
    max_len = 0

    for i in range(1, len(s)):
        sub = s[window_start:i]
        cnt = Counter(sub)

        if max(cnt.values()) == 1:
            max_len = max(max_len, len(sub))
        else:
            window_start += 1

    return max_len


def fruits_into_baskets(fruits: List[str]) -> int:
    """
    You are visiting a farm to collect fruits. The farm has a single row of fruit trees. You will be given two baskets, and your goal is to pick as many fruits as possible to be placed in the given baskets.

    You will be given an array of characters where each character represents a fruit tree. The farm has following restrictions:

    Each basket can have only one type of fruit. There is no limit to how many fruit a basket can hold.
    You can start with any tree, but you can't skip a tree once you have started.
    You will pick exactly one fruit from every tree until you cannot, i.e., you will stop when you have to pick from a third fruit type.
    Write a function to return the maximum number of fruits in both baskets.
    """
    window_start = 0
    max_length = 0
    fruit_frequency = {}

    for window_end in range(len(fruits)):
        right_fruit = fruits[window_end]
        if right_fruit not in fruit_frequency:
            fruit_frequency[right_fruit] = 0
        fruit_frequency[right_fruit] += 1

    # shirk the wilinding window intil we are left with 2 fruits
    while len(fruit_frequency) > 2:
        left_fruit = fruits[window_start]
        fruit_frequency[left_fruit] -= 1
        if fruit_frequency[left_fruit] == 0:
            del fruit_frequency[left_fruit]
        window_start += 1  # slide the window ahead
    max_length = max(max_length, window_end - window_start + 1)
    return max_length


def fruits_into_baskets2(fruits: List[str]) -> int:
    window_start = 0
    window_end = 1
    max_len = 0

    while window_end <= len(fruits):

        sub = fruits[window_start:window_end]
        cnt = Counter(sub)

        if len(cnt.keys()) <= 2:
            max_len = max(max_len, len(sub))
            window_end += 1
        else:
            window_start += 1

    return max_len


assert fruits_into_baskets(["A", "B", "C", "A", "C"]) == 3
assert fruits_into_baskets(["A", "B", "C", "B", "B", "C"]) == 5


def longest_substring_with_k_distinct_2(str1: str, k: int) -> int:
    window_start = 0
    max_len = 0

    for i in range(0, len(str1)):
        cnt = Counter(str1[window_start : i + 1])
        if len(cnt.keys()) > k:
            window_start += 1
        else:
            max_len = max(max_len, i - window_start + 1)

    return max_len


def longest_substring_with_k_distinct(str1: str, k: int) -> int:
    """
    Given a string, find the length of the longest substring in it with no more than K distinct characters.
    """
    window_start = 0
    max_length = 0
    char_frequency = {}

    for window_end in range(len(str1)):
        right_char = str1[window_end]
        if right_char not in char_frequency:
            char_frequency[right_char] = 0
        char_frequency[right_char] += 1

        while len(char_frequency) > k:
            left_char = str1[window_start]
            char_frequency[left_char] -= 1
            if char_frequency[left_char] == 0:
                del char_frequency[left_char]
            window_start += 1
        max_length = max(max_length, window_end - window_start + 1)

    return max_length


assert longest_substring_with_k_distinct("araaci", 2) == 4
assert longest_substring_with_k_distinct("araaci", 1) == 2
assert longest_substring_with_k_distinct("cbbebi", 3) == 5


def smallest_subarry_sum(s: int, arr: List[int]) -> int:
    """
    Given an array of positive numbers and a positive number 's' find the length of the smallest contiguous subarray whose sum is greater than or equal to 's'.
    Return 0 if no such subarray exists.
    """
    window_start, window_end = 0, 0
    window_sum = 0
    min_len = float("inf")

    for window_end in range(len(arr)):
        window_sum += arr[window_end]
        while window_sum >= s:
            min_len = min(min_len, window_end - window_start + 1)
            window_sum -= arr[window_start]
            window_start += 1
    return min_len if min_len != float("inf") else 0


assert smallest_subarry_sum(7, [2, 1, 5, 2, 3, 2]) == 2  # [5,2]
assert smallest_subarry_sum(7, [2, 1, 5, 2, 8]) == 1  # [8]
assert smallest_subarry_sum(8, [3, 4, 1, 1, 6]) == 3  #  [1,1,5]


def max_sub_array_of_size_k(k: int, arr: List[int]) -> int:
    """
    Given an array of positive numbers and a positive number k,
    find the maximum sum of any contiguous subarray of size k.
    """
    max_sum, window_sum = 0, 0
    window_start = 0

    for window_end in range(len(arr)):
        window_sum += arr[window_end]

        if window_end >= k - 1:
            max_sum = max(max_sum, window_sum)
            window_sum -= arr[window_start]
            window_start += 1  # slide the window ahead
    return max_sum


assert max_sub_array_of_size_k(3, [2, 1, 5, 1, 3, 2]) == 9
assert max_sub_array_of_size_k(4, [2, 3, 4, 1, 5]) == 13


def max_sub_array_of_size_k_2(k: int, arr: List[int]) -> int:
    """
    same as above but using sliding window
    """
    max_sum = 0
    for i in range(len(arr) - k + 1):
        max_sum = max(max_sum, sum(arr[i : i + k]))
    return max_sum


assert max_sub_array_of_size_k_2(3, [2, 1, 5, 1, 3, 2]) == 9
assert max_sub_array_of_size_k_2(4, [2, 3, 4, 1, 5]) == 13


def find_averages_of_subarrays(K: int, arr: List[int]) -> List[float]:
    """
    Given an array, find the average of all subarrays of K contiguous elements in it.
    """
    result = []
    window_sum, window_start = 0.0, 0

    for window_end in range(len(arr)):
        window_sum += arr[window_end]

        if window_end >= K - 1:
            result.append(window_sum / K)
            window_sum -= arr[window_start]
            window_start += 1
    return result


arr = [1, 3, 2, 6, -1, 4, 1, 8, 2]
assert find_averages_of_subarrays(5, arr) == [2.2, 2.8, 2.4, 3.6, 2.8]
