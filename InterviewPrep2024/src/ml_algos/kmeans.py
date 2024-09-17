"""
My implementation of KMeans Clustering algorithm in Python.
"""

import numpy as np
from typing import Tuple


def kmeans(
    data: np.array,
    k: int,
    init: str = "random",
    max_iterations: int = 100,
    random_seed: int = 42,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering algorithm.

    Parameters:
    data (ndarray): The dataset with shape (n_samples, n_features).
    k (int): The number of clusters.
    init (str): Initialization method. 'random' or 'k-means++'.
    max_iterations (int): The maximum number of iterations for convergence.
    random_seed (int): Seed for reproducibility.
    tol (float): The tolerance for convergence.

    Returns:
    Tuple[ndarray, ndarray]: The cluster assignments for each data point and the cluster centers.
    """
    np.random.seed(random_seed)

    # 1) Initialize centroids
    if init == "random":
        centroids = data[np.random.permutation(data.shape[0])[:k]]
    elif init == "k-means++":
        centroids = initialize_kmeans_plus_plus(data, k)
    else:
        raise ValueError(f"Invalid initialization method: {init}")

    old_centroids = np.zeros_like(centroids)
    old_assignments = np.zeros(data.shape[0])
    dist = np.inf

    for _ in range(max_iterations):
        # Assign each example to the nearest centroid
        # assignment is done by taking the argmin of the squared distances (euclidean distance)
        assignments = np.argmin(
            np.linalg.norm(data[:, np.newaxis] - centroids[np.newaxis, :], axis=2),
            axis=1,
        )

        # Calcuate the new centroids as the mean of the assigned examples
        centroids = np.array([data[assignments == i].mean(axis=0) for i in range(k)])

        # Calculate the distance between the old and new centroids
        dist = np.linalg.norm(centroids - old_centroids)

        # Check for convergence for tolerance criterion
        if dist < tol:
            break

        # Check for convergence for no change in assignments
        if np.array_equal(assignments, old_assignments):
            break
        # Update old centroids and assignments
        old_centroids = centroids
        old_assignments = assignments

    return assignments, centroids


def initialize_kmeans_plus_plus(data: np.ndarray, k: int) -> np.ndarray:
    """
    Initialize centroids using the K-Means++ algorithm.

    Parameters:
    data (ndarray): The dataset with shape (n_samples, n_features).
    k (int): The number of clusters.

    Returns:
    ndarray: The initialized centroids.
    """
    centroids = [data[np.random.choice(range(data.shape[0]))]]
    for _ in range(1, k):
        distances = np.min(
            np.linalg.norm(data[:, np.newaxis] - np.array(centroids)[np.newaxis, :], axis=2),
            axis=1,
        )
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[j])
                break
    return np.array(centroids)
