"""
2nd round interview question from company A

You are given historical data of flight ticket prices.

Your task is to build a predictive model to estimate the prices of future flight tickets based on this data. The historical data includes the following features:

Distance of flight (in miles)
Number of stops (0 for non-stop, more for flights with stops)
Type of airline (encoded as integers; 1 for budget airlines, 2 for regular airlines, 3 for luxury airlines)
Day of the week (1 for Monday, 7 for Sunday)
You need to predict the Price of the ticket (in USD).

Input:
Your function will receive two inputs:

training_data: A list of tuples, where each tuple represents (distance, stops, airline_type, day_of_week, price).

test_data: A list of tuples, excluding the price (distance, stops, airline_type, day_of_week) for which you need to predict the ticket prices.

Output:
Your function should return a list of predicted prices for the test_data.

Example Answer:
array([378.81587407, 703.3430946 , …])

For a flight of 1500 miles, 0 stops, regular airline, and on a Wednesday: $378.82
"""

import numpy as np

# from data import training_data, test_data

# training date in list of tuples, number value with label
training_data = [
    (2000, 0, 3, 2),
    (1200, 1, 2, 6),
    (1800, 2, 1, 1),
    (1600, 0, 3, 5),
    (1400, 1, 2, 4),
    (1100, 2, 1, 7),
]

# test date in list of tuples, number value without label
test_data = [
    (2000, 0, 3, 2),
    (1200, 1, 2, 6),
    (1800, 2, 1, 1),
    (1600, 0, 3, 5),
    (1400, 1, 2, 4),
    (1100, 2, 1, 7),
]

num_iterations = 100
learning_rate = 0.001


def preprocess_data(data):
    X = np.array([d[:4] for d in data])  # Use all 4 features
    y = np.array([d[-1] for d in data])
    return X, y


def train_model(X, y):
    n_obs, n_features = X.shape
    w = np.random.randn(n_features)
    b = 0.0

    for _ in range(num_iterations):
        w, b = update_weights(X, y, n_obs, w, b)

    return w, b


def update_weights(X, y, n_obs, w, b):
    y_pred = predict_prices(X, w, b)
    grad_w = -2 * np.dot(X.T, (y - y_pred).reshape(-1, 1)) / n_obs
    grad_b = -2 * np.sum(y - y_pred) / n_obs
    w -= learning_rate * grad_w.reshape(w.shape)  # Reshape grad_w to match the shape of w
    b -= learning_rate * grad_b
    return w, b


def predict_prices(X, w, b):
    return np.dot(X, w) + b


X_train, y_train = preprocess_data(training_data)
weights, bias = train_model(X_train, y_train)
X_test = np.array([d[:4] for d in test_data])  # Use all 4 features
predicted_prices = predict_prices(X_test, weights, bias)
print("Predicted prices:", predicted_prices)
