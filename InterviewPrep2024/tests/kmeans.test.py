import numpy as np
import pytest
from ..src.ml_algos.kmeans import kmeans

# Test case: Should return correct assignments when data is uniformly distributed
def test_kmeans_uniform_data():
    np.random.seed(42)
    data = np.random.rand(100, 2)  # Uniformly distributed data
    k = 3
    expected_assignments = np.array([0] * 33 + [1] * 33 + [2] * 34)  # Expected assignments

    assignments, _ = kmeans(data, k)

    assert np.array_equal(assignments, expected_assignments), "Test case failed: Incorrect assignments for uniformly distributed data"


def test_kmeans_raises_error_with_less_than_one_cluster():

    data = np.array([[1, 2], [3, 4], [5, 6]])
    k = 0

    with pytest.raises(ValueError) as e:
        kmeans(data, k)

    assert str(e.value) == "Number of clusters must be greater than 0"


def test_kmeans_raises_error_with_invalid_initialization():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    k = 3
    init = 'invalid'

    with pytest.raises(NotImplementedError) as e:
        kmeans(data, k, init)

    assert str(e.value) == "Invalid initialization invalid"


def test_kmeans_dataset():
    np.random.seed(42)
    data = np.random.rand(1000, 10)  # Large dataset
    k = 5
    expected_assignments = None  # Expected assignments are not provided in this test case

    assignments, _ = kmeans(data, k)

    # Check if the function runs without errors
    assert assignments is not None, "Test case failed: Function returned None for large dataset"

    # Check if the number of assignments matches the number of data points
    assert len(assignments) == len(data), "Test case failed: Number of assignments does not match the number of data points"
