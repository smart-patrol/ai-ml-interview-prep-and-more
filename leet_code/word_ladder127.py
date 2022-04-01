from collections import deque, defaultdict
from typing import List


def ladderLength(beginWord: str, endWord: str, wordList: List[str]):
    """A transformation sequence from word beginWord to word endWord using a dictionar wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

    Every adjacent pair of words differs by a single letter.
    Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
    sk == endWord
    Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.
    """
    if endWord not in wordList or not endWord or not beginWord or not wordList:
        return 0

    # Start from beginWord and search the endWord using BFS.
    # Algorithm
    # Do the pre-processing on the given wordList and find all the possible generic/intermediate states. Save these intermediate states in a dictionary with key as the intermediate word and value as the list of words which have the same intermediate word.
    # Push a tuple containing the beginWord and 1 in a queue. The 1 represents the level number of a node. We have to return the level of the endNode as that would represent the shortest sequence/distance from the beginWord.
    # To prevent cycles, use a visited dictionary.
    # While the queue has elements, get the front element of the queue. Let's call this word as current_word.
    # Find all the generic transformations of the current_word and find out if any of these transformations is also a transformation of other words in the word list. This is achieved by checking the all_combo_dict.
    # The list of words we get from all_combo_dict are all the words which have a common intermediate state with the current_word. These new set of words will be the adjacent nodes/words to current_word and hence added to the queue.
    # Hence, for each word in this list of intermediate words, append (word, level + 1) into the queue where level is the level for the current_word.
    # Eventually if you reach the desired word, its level would represent the shortest transformation sequence length.
    # Termination condition for standard BFS is finding the end word.

    L = len(beginWord)

    all_combo_dict = defaultdict(list)
    for word in wordList:
        for i in range(L):
            all_combo_dict[word[:i] + "*" + word[i + 1 :]].append(word)

    queue = deque()
    queue.append((beginWord, 1))
    visited = set()
    visited.add(beginWord)

    while queue:
        current_word, level = queue.popleft()
        for i in range(L):
            intermeidate_word = current_word[:i] + "*" + current_word[i + 1 :]

            for word in all_combo_dict[intermeidate_word]:
                if word == endWord:
                    return level + 1
                if word not in visited:
                    visited.add(word)
                    queue.append((word, level + 1))
                all_combo_dict[intermeidate_word] = []

    return 0


assert ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
assert ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
