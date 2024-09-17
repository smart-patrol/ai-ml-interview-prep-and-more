"""
Linear regression model with gradient descent
"""

import numpy as np


class LinearRegression:
    """
    Implentation from:
    https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLC/notebooks/linear_regression.ipynb
    """

    def __init__(self, alpha=0) -> None:
        self.alpha = alpha  # regurlarization parameter
        self.W = None

    def fit(self, X, y, lr=0.01, num_iter=1000):
        # check input data validity
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X and y must have the same length and cannot be empty")

        # Add bias term to X -> [1 X] , track with the other variables
        X = np.hstack([np.ones((len(X), 1)), X])

        # Initialize W to random values
        self.W = np.random.randn(X.shape[1])

        # Use gradient descent to minimize cost function
        for i in range(num_iter):
            # get predicted values
            y_pred = np.dot(X, self.W)
            # calculate derivatives with respect to weights and bias term
            gradients = 2 * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * 2 * self.W
            # update the weights
            self.W = self.W - lr * gradients

            if i % 1000 == 0:
                # print cost function to track convergence
                cost = np.sum((y_pred - y) ** 2) + self.alpha * np.sum(self.W**2)
                print(cost)

    def predict(self, X):
        # Add bias term to X
        X = np.hstack([np.ones((len(X), 1)), X])
        # Calculate predicted values with the bias term
        y_pred = np.dot(X, self.W)
        return y_pred


# Tests the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([2, 4, 6])

model = LinearRegression(alpha=0.1)
model.fit(X, y)
print(model.predict(np.array([[4, 5]])))  # Expected output: [5.00000000e+00]


class LinearRegressionv2:
    """
    Implentation from:
    https://www.yuan-meng.com/posts/md_coding/
    """

    def __init__(self, lr=0.01, epoch=10) -> None:
        self.lr = lr
        self.epoch = epoch

    def fit(self, X, y):
        self.n_obs, self.n_features = X.shape
        # intialize weights and biaess
        self.w = np.zeros(self.n_features)
        self.b = 0

        self.X = X
        self.y = y

        for _ in range(self.epoch):
            self.update_weights()

        return self

    def update_weights(self):
        y_pred = self.predict(self.X)
        # compute gradients with respect to weights and bias
        grad_w = -2 * np.dot(self.X.T, (self.y - y_pred)) / self.n_obs
        grad_b = -2 * np.sum(self.y - y_pred) / self.n_obs
        # update parameters by subtracting gradients multiplied by learning rate
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# Tests the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([2, 4, 6])

model = LinearRegressionv2(lr=0.01, epoch=100)
model.fit(X, y)
# assert model.score(X, y) < 1e-6
print(f"Weights: {model.get_params()[0]}, Bias: {model.get_params()[1]}")
print(f"Score: {model.score(X, y)}")

class LinearRegressionv3:
    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 42) -> None:
        """
        Initialize a Linear Regression model with Gradient Descent.
        From: "Machine Learning with PyTorch and Scikit-Learn" by Sebastian Raschka et al.

        Parameters:
        eta (float): The learning rate for the Gradient Descent algorithm. Default is 0.01.
        n_iter (int): The number of iterations for the Gradient Descent algorithm. Default is 50.
        random_state (int): The seed for the random number generator. Default is 42.

        Returns:
        None
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self._w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self._b = np.array([0])
        self.losses = []

        for i in range(self.n_iter):
            output = np.dot(X, self._w) + self._b
            errors = output - y
            self._w += self.eta * 2.0 * (X.T.dot(errors)) / X.shape[0]
            self._b += self.eta * 2.0 * errors.mean()
            loss = np.mean(errors ** 2)
            self.losses.append(loss)
        return self
    
    def predict(self, X):
        return np.dot(X, self._w) + self._b
    
# Tests the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([2, 4, 6])

model = LinearRegressionv3(eta=0.01, n_iter=100)
model.fit(X, y)
# assert model.score(X, y) < 1e-6


class LinearRegressionv4:
    """
    This is the closed from solution to the linear regression problem using numpy.
    From: "Machine Learning with PyTorch and Scikit-Learn" by Sebastian Raschka et al.
    """
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add a column of ones to the input data for the bias term
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Calculate the weights using the closed form solution
        self.weights = np.zeros(X.shape[1])
        z = np.linalg.inv(X.T, X)
        self.weights = np.dot(z, np.dot(X.T, y))
        # Extract the bias term from the weights
        self.bias = self.weights[-1]

        return self

    def predict(self, X):
        # Add a column of ones to the input data for the bias term
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        # Calculate the predicted values
        return X.dot(self.weights) + self.bias
    
## test the model on data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([2, 4, 6])

model = LinearRegressionv4()
model.fit(X, y)
# assert model.score(X, y) < 1e-6