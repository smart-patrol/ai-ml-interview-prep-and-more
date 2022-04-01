import sys
import math
import os
import random
import re
import sys


if __name__ == "__main__":
    s = "I came from the moon. He went to the other room. She went to the drawing room. "
    n = 3

    def make_trigrams(sentence: str) -> list:
        """make word trigrams from sentence"""
        words = sentence.split()
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append(
                words[i].lower()
                + " "
                + words[i + 1].lower()
                + " "
                + words[i + 2].lower()
            )
        return trigrams

    print(make_trigrams(s))
