from typing import List, Tuple
from ..src.ml_algos import  DecisionTree,Node
import numpy as np

# Test cases to validate the implementation
def test_node_initialization():
    node = Node(predicted_class=1)
    assert node.predicted_class == 1
    assert node.is_leaf_node() == True

def test_decision_tree_initialization():
    dt = DecisionTree(max_depth=5)
    assert dt.max_depth == 5
    assert dt.tree_ is None

def test_gini_impurity():
    dt = DecisionTree()
    y = np.array([0, 0, 1, 1])
    assert np.isclose(dt._gini(y), 0.5)
    y = np.array([0, 0, 0, 0])
    assert np.isclose(dt._gini(y), 0.0)
    y = np.array([0, 1, 2, 3])
    assert np.isclose(dt._gini(y), 0.75)

def test_best_split():
    dt = DecisionTree()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    dt.n_classes_ = 2
    dt.n_features_ = 2
    best_idx, best_thr = dt._best_split(X, y)
    assert best_idx == 0
    assert best_thr == 4.0

def test_grow_tree():
    dt = DecisionTree(max_depth=2)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    dt.n_classes_ = 2
    dt.n_features_ = 2
    tree = dt._grow_tree(X, y)
    assert isinstance(tree, Node)
    assert tree.feature_index == 0
    assert tree.threshold == 4.0
    assert tree.left.predicted_class == 0
    assert tree.right.predicted_class == 1

def test_fit_and_predict():
    dt = DecisionTree(max_depth=2)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    dt.fit(X, y)
    predictions = dt.predict(X)
    assert np.array_equal(predictions, y)

def test_predict_single():
    dt = DecisionTree(max_depth=2)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    dt.fit(X, y)
    assert dt._predict_single(np.array([2, 3])) == 0
    assert dt._predict_single(np.array([6, 7])) == 1

def run_all_tests():
    test_node_initialization()
    test_decision_tree_initialization()
    test_gini_impurity()
    test_best_split()
    test_grow_tree()
    test_fit_and_predict()
    test_predict_single()
    print("All tests passed!")