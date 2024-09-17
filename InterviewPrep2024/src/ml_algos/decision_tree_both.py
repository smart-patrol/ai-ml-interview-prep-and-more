from typing import Optional, Tuple, List
import numpy as np


class Node:
    """
    A class representing a node in the decision tree.

    Attributes:
        predicted_class (int): The class predicted by this node.
        feature_index (Optional[int]): The index of the feature used for splitting.
        threshold (Optional[float]): The threshold value for the feature used for splitting.
        left (Optional[Node]): The left child node.
        right (Optional[Node]): The right child node.
    """

    def __init__(self, predicted_class: int):
        self.predicted_class = predicted_class
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.split_method: Optional[str] = "entropy"

    def is_leaf_node(self) -> bool:
        """
        Check if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf node, False otherwise.
        """
        return self.left is None and self.right is None


class DecisionTree:
    """
    A simple implementation of a Decision Tree classifier.

    Attributes:
        max_depth (Optional[int]): The maximum depth of the tree.
        n_classes_ (int): The number of unique classes in the target variable.
        n_features_ (int): The number of features in the dataset.
        tree_ (Optional[Node]): The root node of the decision tree.
    """

    def __init__(self, max_depth: Optional[int] = None):
        self.max_depth = max_depth
        self.n_classes_: int = 0
        self.n_features_: int = 0
        self.tree_: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the decision tree to the training data.

        Args:
            X (np.ndarray): The feature matrix of the training data.
            y (np.ndarray): The target labels of the training data.
        """
        self.n_classes_ = len(np.unique(y))  # Determine the number of unique classes
        self.n_features_ = X.shape[1]  # Determine the number of features
        self.tree_ = self._grow_tree(X, y)  # Start growing the tree

    def predict(self, X: np.ndarray) -> List[int]:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): The feature matrix of the samples to predict.

        Returns:
            List[int]: The predicted class labels.
        """
        return [self._predict_single(inputs) for inputs in X]

    def _information_gain(self, y: np.ndarray) -> float:
        """
        Calculate the Gini impurity of a node or Entropy of a node.

        Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled
        if it was randomly labeled according to the distribution of labels in the subset.

        Args:
            y (np.ndarray): The target labels of the dataset.

        Returns:
            float: The Gini impurity.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        if self.method == "gini":
            return 1 - np.sum(probabilities**2)
        else:
            return -np.sum(probabilities * np.log2(probabilities))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best feature and threshold to split the dataset.

        This function iterates through each feature and threshold to find the split that results in the lowest Gini impurity or Entropy.

        Args:
            X (np.ndarray): The feature matrix of the dataset.
            y (np.ndarray): The target labels of the dataset.

        Returns:
            Tuple[Optional[int], Optional[float]]: The index of the best feature and the best threshold.
            If no split is found, returns (None, None).
        """
        m = y.size
        if m <= 1:
            return None, None

        parent_information_gain = self._information_gain(y)
        best_information_gain = parent_information_gain

        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            # Sort the data along the feature `idx`
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = [np.sum(y == c) for c in range(self.n_classes_)]

            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                # Skip if the threshold is the same as the previous one
                if thresholds[i] == thresholds[i - 1]:
                    continue

                # Calculate the information gain for the split
                information_gain = self._calculate_split_gini(num_left, num_right, i, m)

                # Update the best split if the current information gain is lower
                if information_gain < best_information_gain:
                    best_information_gain = information_gain
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _calculate_split(self, num_left: List[int], num_right: List[int], i: int, m: int) -> float:
        """
        Calculate the Gini impurity or Entropy for a split.

        Args:
            num_left (List[int]): The count of each class in the left subset.
            num_right (List[int]): The count of each class in the right subset.
            i (int): The number of samples in the left subset.
            m (int): The total number of samples in the dataset.

        Returns:
            float: The Gini impurity for the split.
        """
        if self.method == "gini":
            # calcualte as 1- sum of squared probabilities
            gini_left = 1.0 - sum((nl / i) ** 2 for nl in num_left)
            gini_right = 1.0 - sum((nr / (m - i)) ** 2 for nr in num_right)
            return (i * gini_left + (m - i) * gini_right) / m
        else:
            # calculate as - sum of probabilities * log2(probabilities)
            entropy_left = -sum((nl / i) * np.log2(nl / i) if nl != 0 else 0 for nl in num_left)
            entropy_right = -sum(
                (nr / (m - i)) * np.log2(nr / (m - i)) if nr != 0 else 0 for nr in num_right
            )
            return i * entropy_left + (m - i) * entropy_right

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively grow the decision tree.

        This function builds the decision tree by recursively splitting the dataset based on the best feature and threshold.
        The recursion stops when the maximum depth is reached or when all samples in a node are of the same class.

        Args:
            X (np.ndarray): The feature matrix of the dataset.
            y (np.ndarray): The target labels of the dataset.
            depth (int, optional): The current depth of the tree. Defaults to 0.

        Returns:
            Node: The root node of the decision tree.
        """
        # Count the number of samples per class
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(
            num_samples_per_class
        )  # Predict the class with the most samples
        node = Node(predicted_class=predicted_class)

        # Stop if the maximum depth is reached
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                # Split the dataset into left and right subsets
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict_single(self, inputs: np.ndarray) -> int:
        """
        Predict the class for a single sample.

        This function traverses the decision tree from the root to a leaf node to make a prediction.

        Args:
            inputs (np.ndarray): The feature values of the sample.

        Returns:
            int: The predicted class label.
        """
        node = self.tree_
        while not node.is_leaf_node():
            # Traverse the tree based on the feature value and threshold
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
