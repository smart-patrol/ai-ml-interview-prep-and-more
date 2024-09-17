"""
Logistic Regression Classifier
"""

import numpy as np


class LogisitcRegression:
    """
    Implementation of logistic regression from:
    https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLC/notebooks/logistic_regression.ipynb
    """

    def __init__(self, lr=0.01, n_iters=1000):
        """
        Initialize logistic regression parameters.

        Args:
            lr (float): Learning rate. Defaults to 0.01.
            n_iters (int): Number of iterations. Defaults to 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the logistic regression model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target variable.
        """
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if _ % 100 == 0:
                cost = (-1 / n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
                print(f"Cost at iteration {_}: {cost}")

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def _sigmoid(self, z):
        """
        Compute sigmoid function.

        Args:
            z (np.ndarray): Input to the sigmoid function.

        Returns:
            np.ndarray: Sigmoid of z.
        """
        return 1 / (1 + np.exp(-z))

    def predict_binary(self, X):
        """
        Convert probabilities to binary class labels.

        Args:
            X (np.ndarray): Feature matrix for prediction.

        Returns:
            np.ndarray: Binary class labels.
        """
        return (self.predict(X) > 0.5).astype(int)


# Tests the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

model = LogisitcRegression(lr=0.01, n_iters=1000)
model.fit(X, y)
print(model.predict(np.array([[4, 5]])))  # Expected output: 0


class LogisticRegressionv2:
    """implementation from:
    https://www.yuan-meng.com/posts/md_coding/
    """

    def __init__(self, lr=0.01, epoch=10):
        self.lr = lr  # Learning rate
        self.epoch = epoch  # Number of epochs

    def fit(self, X, y):
        self.n_obs, self.n_features = X.shape

        self.w = np.zeros(self.n_features)
        self.b = 0

        self.X = X
        self.y = y

        for _ in range(self.epoch):
            self.update_weights()

        return self

    def update_weights(self):
        y_pred = self.predict(self.X)

        grad_w = -np.dot(self.X.T, (self.y - y_pred)) / self.n_obs
        grad_b = -np.sum(self.y - y_pred) / self.n_obs

        self.w = self.w - self.lr * grad_w
        self.b = self.b - self.lr * grad_b

        return self

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):

        z = X.dot(self.w) + self.b
        # Sigmoid -> transform [0, 1]
        p = self.sigmoid(z)

        return np.where(p < 0.5, 0, 1)


# test the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

model = LogisticRegressionv2(lr=0.01, epoch=100)
model.fit(X, y)
print(model.predict(np.array([[4, 5]])))  # Expected output: [0]


class LogisticRegressionv3:
    """
    This is based off of the logistic regression from the book:
    "Machine Learning with PyTorch and Scikit-Learn" by Sebastian Raschka et al.
    """
    def __init__(self, learning_rate:float=0.01, num_iterations:int=100, random_state:int=42
                 , regularization:float = 1e-4) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.regularization = regularization
        self._weights = None
        self.bias = None

    def sigmoid(self, z) -> float:
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._weights = np.random.randn(n_samples, n_features, seed=self.random_state)
        self._bias = 0
        self._losses = []

        for _ in range(self.num_iterations):
            z = np.dot(X, self._weights) + self.bias
            output = self.sigmoid(z)
            errors = y - output
            # w/ l2 regularization
            self._weights += self.learning_rate * (np.dot(X.T, errors) / n_samples) + self.regularization * self._weights / n_samples
            # w/o regularization
            #self._weights += self.learning_rate * np.dot(X.T, errors) / n_samples
            self._bias += self.learning_rate * np.sum(errors) / n_samples
            nll = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1-output)))) / n_samples
            self._losses.append(nll)
    
        return self
    
    def predict(self, X):
        z = np.dot(X, self._weights) + self.bias
        return np.where(self.sigmoid(z) >= 0.5, 1, 0) # cut off set at 0.5
    
### test the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])

model = LogisticRegressionv3(learning_rate=0.01, num_iterations=100)
model.fit(X, y)
print(model.predict(np.array([[4, 5]])))  # Expected output: [0]