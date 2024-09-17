"""
K NearestNeighbors Classifier in Numpy
"""

import numpy as np
from scipy.stats import mode


class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_train = X.shape[0]

        return self

    def predict(self, X_test):
        self.X_test = X_test
        self.n_test = X_test.shape[0]
        y_pred = np.zeros(self.n_test)

        for i in range(self.n_test):
            p = self.X_test[i]
            neighbors = np.zeros(self.n_neighbors)
            neighbors = self.find_neighbors(p)
            y_pred[i] = mode(neighbors)[0][0]

        return y_pred

    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def find_neighbors(self, p):
        distances = np.zeros(self.n_train)

        for i in range(self.n_train):
            distance = self.euclidean_distance(p, self.X[i])
            distances[i] = distance
        _, y_train_sorted = zip(*sorted(zip(distances, self.y)))

        return y_train_sorted[: self.n_neighbors]


# Tests the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print(model.predict(np.array([[4, 5]])))  # Expected output: [0]
