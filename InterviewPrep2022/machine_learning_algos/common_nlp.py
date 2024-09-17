import numpy as np
from typing import List, Dict, Set
from collections import Counter, defaultdict
import re


def bigrams(text: str) -> List[str]:
    """
    Create bigrams from a sentence
    """
    bigrams = [
        (ele, tex.split()[i + 1])
        for tex in text
        for i, ele in enumerate(tex.split())
        if i < len(tex.split()) - 1
    ]
    return bigrams


def trigrams(text: str) -> List[str]:
    """
    Create trigrams from a sentence
    """
    trigrams = [
        (ele, tex.split()[i + 1], tex.split()[i + 2])
        for tex in text
        for i, ele in enumerate(tex.split())
        if i < len(tex.split()) - 2
    ]
    return trigrams


def clean_text(sentence: str) -> str:
    """
    Remove punctuation, stopwords, and make everything lowercase
    Return list of words for sentence
    """
    # reomve white space
    string = re.sub(r"\s+", " ", sentence)
    # remove punctuation
    string = re.sub(r"[^a-zA-Z0-9]", " ", string)
    # make lower case
    string = string.lower()
    # toeknize
    words = string.split()
    # remove stopwords
    stopwords = set("stop")
    words = [word for word in words if word not in stopwords]
    # lemmatize
    # words = [lemmatizer.lemmatize(word) for word in words]
    return words


# -------------------------------------------------------
# calcualte tf-idf
# assuumes bag of words=doc.split()


def word_count_pairs(docA: List[str], docB: List[str]) -> Dict:
    """
    Return a dict of word count pairs
    """
    unique_words = set(docA).union(set(docB))

    word_countA = dict.fromkeys(unique_words, 0)
    word_countB = dict.fromkeys(unique_words, 0)

    for word in docA:
        word_countA[word] += 1
    for word in docB:
        word_countB[word] += 1

    return word_countA, word_countB


#  word_countsA, word_countsB = word_count_pairs(docA, docB)


def tf(word_counts: Counter, words: List[int]) -> Dict:
    """
    Term frequency
    Frequency of a word in a document divided by the total number of words in the document
    """
    tf_dict = {}
    bow_count = len(words)
    for word, count in word_counts.items():
        tf_dict[word] = count / bow_count
    return tf_dict


# tf_bow_A = tf(word_countsA, wordsA)
# tf_bow_B =  tf(word_countsB, wordsB)


def idf(docs: List[str]) -> Dict:
    """
    Inverse document frequency
    Total number of documents / number of documents containing the word
    """
    import math

    idf_dict = {}
    N = len(docs)

    idf_dict = dict.fromkeys(docs[0].keys(), 0)
    for doc in docs:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1

    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(N / float(val))

    return idf_dict


# idfs = idf([word_countsA, word_countsB])


def tf_idf(tf_bow: Dict, idfs: Dict) -> Dict:
    """
    tf-idf
    measures how important a word is to a document in a corpus
    """
    tf_idf_dict = {}
    for word, val in tf_bow.items():
        tf_idf_dict[word] = val * idfs[word]
    return tf_idf_dict


# tfidf_A = tf_idf(tf_bow_A, idfs)
# tfidf_B = tf_idf(tf_bow_B, idfs)


# -------------------------------------------------------
# calculate cosine similarity in numpy
def cosine_similarity(v1: np.array, v2: np.array) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# to use with docs
def tf_idf_cosine_similarity(tfidf_A: Dict, tfidf_B: Dict) -> float:
    # convert dict to numpy array
    tfidf_A = np.array(list(tfidf_A.values()))
    tfidf_B = np.array(list(tfidf_B.values()))

    return cosine_similarity(tfidf_A, tfidf_B)


def cosine_similarity2(v, w):
    return np.dot(v, w) / np.sqrt(np.dot(v, v) * np.dot(w, w))


def preplexity(A: np.array, Y: np.array) -> float:
    """
    Calcualte preplexity as entropy **2
    """
    m = len(A)
    cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return cost ** 2


docA = "The cat sat on my face"
docB = "The dog sat on my bed"
wordsA = docA.split()
wordsB = docB.split()
word_countsA, word_countsB = word_count_pairs(docA, docB)
tf_bow_A = tf(word_countsA, wordsA)
tf_bow_B = tf(word_countsB, wordsB)
idfs = idf([word_countsA, word_countsB])
tfidf_A = tf_idf(tf_bow_A, idfs)
tfidf_B = tf_idf(tf_bow_B, idfs)
print(tfidf_A)
print(tfidf_B)
print(tf_idf_cosine_similarity(tfidf_A, tfidf_B))
