from typing import Set, NamedTuple
import re
from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict


def tokenize(text: str) -> Set[str]:
    text = text.lower()  # Convert to lowercase,
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)  # remove duplicates.


assert tokenize("Data Science is science") == {"data", "science", "is"}


class Data(NamedTuple):
    text: str
    is_target: bool


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_target_counts: Dict[str, int] = defaultdict(int)
        self.token_other_counts: Dict[str, int] = defaultdict(int)
        self.target_datas = self.other_datas = 0

    def train(self, datas: Iterable[Data]) -> None:
        for data in datas:
            # Increment data counts
            if data.is_target:
                self.target_datas += 1
            else:
                self.other_datas += 1

            # Increment word counts
            for token in tokenize(data.text):
                self.tokens.add(token)
                if data.is_target:
                    self.token_target_counts[token] += 1
                else:
                    self.token_other_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | target) and P(token | not target)"""
        target = self.token_target_counts[token]
        other = self.token_other_counts[token]

        p_token_target = (target + self.k) / (self.target_datas + 2 * self.k)
        p_token_other = (other + self.k) / (self.other_datas + 2 * self.k)

        return p_token_target, p_token_other

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_target = log_prob_if_other = 0.0

        # Iterate through each word in our vocabulary.
        for token in self.tokens:
            prob_if_target, prob_if_other = self._probabilities(token)

            # If *token* appears in the data,
            # add the log probability of seeing it;
            if token in text_tokens:
                log_prob_if_target += math.log(prob_if_target)
                log_prob_if_other += math.log(prob_if_other)

            # otherwise add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_target += math.log(1.0 - prob_if_target)
                log_prob_if_other += math.log(1.0 - prob_if_other)

        prob_if_target = math.exp(log_prob_if_target)
        prob_if_other = math.exp(log_prob_if_other)
        return prob_if_target / (prob_if_target + prob_if_other)


datas = [
    Data("target rules", is_target=True),
    Data("other rules", is_target=False),
    Data("hello other", is_target=False),
]

model = NaiveBayesClassifier(k=0.5)
model.train(datas)

assert model.tokens == {"target", "other", "rules", "hello"}
assert model.target_datas == 1
assert model.other_datas == 2
assert model.token_target_counts == {"target": 1, "rules": 1}
assert model.token_other_counts == {"other": 2, "rules": 1, "hello": 1}

text = "hello target"

probs_if_target = [
    (1 + 0.5) / (1 + 2 * 0.5),  # "target"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "other"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5),  # "hello" (present)
]

probs_if_other = [
    (0 + 0.5) / (2 + 2 * 0.5),  # "target"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "other"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),  # "hello" (present)
]

p_if_target = math.exp(sum(math.log(p) for p in probs_if_target))
p_if_other = math.exp(sum(math.log(p) for p in probs_if_other))

# Should be about 0.83
assert model.predict(text) == p_if_target / (p_if_target + p_if_other)


def drop_final_s(word):
    return re.sub("s$", "", word)


# def main():
#     import glob, re

#     # modify the path to wherever you've put the files
#     path = 'target_data/*/*'

#     data: List[data] = []

#     # glob.glob returns every filename that matches the wildcarded path
#     for filename in glob.glob(path):
#         is_target = "other" not in filename

#         # There are some garbage characters in the emails, the errors='ignore'
#         # skips them instead of raising an exception.
#         with open(filename, errors='ignore') as email_file:
#             for line in email_file:
#                 if line.startswith("Subject:"):
#                     subject = line.lstrip("Subject: ")
#                     data.append(data(subject, is_target))
#                     break  # done with this file

#     import random
#     from scratch.machine_learning import split_data

#     random.seed(0)      # just so you get the same answers as me
#     train_datas, test_datas = split_data(data, 0.75)

#     model = NaiveBayesClassifier()
#     model.train(train_datas)

#     from collections import Counter

#     predictions = [(data, model.predict(data.text))
#                    for data in test_datas]

#     # Assume that target_probability > 0.5 corresponds to target prediction
#     # and count the combinations of (actual is_target, predicted is_target)
#     confusion_matrix = Counter((data.is_target, target_probability > 0.5)
#                                for data, target_probability in predictions)

#     print(confusion_matrix)

#     def p_target_given_token(token: str, model: NaiveBayesClassifier) -> float:
#         # We probably shouldn't call private methods, but it's for a good cause.
#         prob_if_target, prob_if_other = model._probabilities(token)

#         return prob_if_target / (prob_if_target + prob_if_other)

#     words = sorted(model.tokens, key=lambda t: p_target_given_token(t, model))

#     print("targetmiest_words", words[-10:])
#     print("othermiest_words", words[:10])

# if __name__ == "__main__": main()
