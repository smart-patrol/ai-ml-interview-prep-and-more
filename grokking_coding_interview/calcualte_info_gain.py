import collections
import math
from typing import List


class Solution:
    def calculateMaxInfoGain(
        self, petal_length: List[float], species: List[str]
    ) -> float:

        # Edge case
        if not petal_length or not species:
            return 0

        # Sort petal_length first to find a place to split
        # [(petal_length, species), (petal_length, species), ...]
        length_species_list = sorted(zip(petal_length, species))
        # Unpack to two lists
        petal_length, species = zip(*length_species_list)

        # To maximize information gain, minimize subgroup weighted entropy sum
        subgroup_weighted_entropy_sum = float("inf")
        n = len(petal_length)

        # Try all the splitting position
        for i in range(1, n):
            h_1 = self.calculateEntropy(species[:i])
            h_2 = self.calculateEntropy(species[i:])
            subgroup_weighted_entropy_sum = min(
                subgroup_weighted_entropy_sum, h_1 * (i / n) + h_2 * ((n - i) / n)
            )

        original_group_entropy = self.calculateEntropy(species)

        return original_group_entropy - subgroup_weighted_entropy_sum

    def calculateEntropy(self, input: List[str]) -> float:
        counter = collections.Counter(input)
        ans = 0
        for count in counter.values():
            p = count / len(input)
            ans += -1 * p * math.log2(p)
        return ans


if __name__ == "__main__":
    # Test
    petal_length = [0.5, 2.3, 1.0, 1.5]
    species = ["setosa", "versicolor", "setosa", "versicolor"]
    print(Solution().calculateMaxInfoGain(petal_length, species))
