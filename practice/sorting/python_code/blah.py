import heapq
from collections import defaultdict


class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        n = len(hand)
        if n % groupSize != 0:
            return False
        counter = {}
        for num in hand:
            counter[num] = 1 + counter.get(num, 0)
        heap = list(counter.keys())
        heapq.heapify(heap)

        while heap:
            top = heap[0]
            if counter[top] > 0:
                i = top
                while i < top + groupSize:
                    if i not in counter:
                        return False
                        i += 1
                    counter[i] -= 1
                    if counter[i] == 0:
                        if i != heap[0]:
                            return False
                        heapq.heappop(heap)
        return True
