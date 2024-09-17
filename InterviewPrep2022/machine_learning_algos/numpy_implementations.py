import numpy as np

def k_nearest_neigbors(x:np.array, y:np.array, query:np.array, k:int=5) -> np.array:
    """
    K-Nearest Neighbors algorithm
    """
    # 1. Calculate the distance between each point
    distances = np.linalg.norm(x - query, axis=1)
    # 2. Sort the distances
    sorted_distances_indices = np.argsort(distances)
    # 3. Return the k nearest neighbors
    return y[sorted_distances_indices[:k]]

def k_means(x:np.array, k:int=5) -> np.array:
    """
    K-Means algorithm
    """
    # 1. Randomly choose k points as the initial centroids
    centroids = x[np.random.choice(x.shape[0], k, replace=False)]
    # 2. Assign each point to the closest centroid
    while True:
        # a. Assign each point to the closest centroid
        closest_centroids = np.argmin(np.linalg.norm(x - centroids[:, np.newaxis], axis=2), axis=1)
        # b. Update the centroids
        for i in range(k):
            centroids[i] = np.mean(x[closest_centroids == i], axis=0)
        # c. Stop if the centroids don't change
        if np.all(centroids == centroids_old):
            break
        centroids_old = centroids
    return closest_centroids

def k_mean_plusplus(x:np.array, k:int=5) -> np.array:
    """
    K-Means ++ algorthim implemented in numpy
    """
    # 1. Choose one point as the first centroid
    centroids = x[np.random.choice(x.shape[0], 1, replace=False)]
    # 2. Assign each point to the closest centroid
    while len(centroids) < k:
        # a. Calculate the distance between each point and the closest centroid
        distances = np.linalg.norm(x - centroids[:, np.newaxis], axis=2)
        # b. Choose the point that is farthest from the closest centroid
        farthest_point = np.argmax(np.min(distances, axis=1))
        # c. Add the farthest point to the centroids
        centroids = np.vstack((centroids, x[farthest_point]))
    # 3. Assign each point to the closest centroid
    closest_centroids = np.argmin(np.linalg.norm(x - centroids[:, np.newaxis], axis=2), axis=1)
    return closest_centroids


# kmeans steps:
# 1) Initialize k points (corresponding to k clusters) randomly from the data. We call these points centroids.
# 2) For each data point, measure the L2 distance from the centroid. Assign each data point to the centroid for which it has the shortest distance. In other words, assign the closest centroid to each data point.
# 3) Now each data point assigned to a centroid forms an individual cluster. For k centroids, we will have k clusters. Update the value of the centroid of each cluster by the mean of all the data points present in that particular cluster.
# 4) Repeat steps 1-3 until the maximum change in centroids for each iteration falls below a threshold value, or the clustering error converges.

def kMeans(X:np.array, K:int, max_iters:int =10, plot_progress=None ) -> np.array:

    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(max_iters):
        # cluster assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # move centroids step
        centroids = [X[C==k].mean(axis=0) for k in range(K)]
        if plot_progress != None:
            plot_progress(X, C, np.array(centroids))
        return np.array(centroids), C
import sys
import matplotlib.pyplot as plt


def show(X, C, centroids, keep = False):
    import time
    time.sleep(0.5)
    plt.plot(X[C == 0, 0], X[C == 0, 1], '*b',
         X[C == 1, 0], X[C == 1, 1], '*r',
         X[C == 2, 0], X[C == 2, 1], '*g')
    plt.plot(centroids[:,0],centroids[:,1],'*m',markersize=20)
    if keep :
        plt.show()

# generate 3 cluster data
# data = np.genfromtxt('data1.csv', delimiter=',')
m1, cov1 = [9, 8], [[1.5, 2], [1, 2]]
m2, cov2 = [5, 13], [[2.5, -1.5], [-1.5, 1.5]]
m3, cov3 = [3, 7], [[0.25, 0.5], [-0.1, 0.5]]
data1 = np.random.multivariate_normal(m1, cov1, 250)
data2 = np.random.multivariate_normal(m2, cov2, 180)
data3 = np.random.multivariate_normal(m3, cov3, 100)
X = np.vstack((data1,np.vstack((data2,data3))))
np.random.shuffle(X)

centroids, C = kMeans(X, K = 3, plot_progress = show)
show(X, C, centroids, True)


#https://github.com/hhundiwala/hierarchical-clustering/blob/master/Hierarchical%20Clustering%20Explanation.ipynb
def hierarchical_clustering(data, linkage, no_of_clusters):
    intital_distances =  np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    np.fill_diagonal(intital_distances, np.inf)
    clusters = find_clusster(intital_distances, linkage)

    return clusters

def find_clusters(input, linkage):
    clusters = {}
    row_index = -1
    col_index = -1
    array = []

    for n in range(input.shape[0]):
        array.append(n)

    clusters[0] = array.copy()

    # finding the minimum value from the distance matrix
    for k in range(1, input.shape[0]):
        min_val = np.inf

        for i in range(0, input.shape[0]]):
            for j in range(0, input.shape[1]):
                if input[i][j] <= min_val:
                    min_val = input[i][j]
                    row_index = i
                    col_index = j

        if linkage == "single":
            for i in range(0, input.shape[0]):
                if i != col_index:
                    temp = min(input[row_index][i], input[col_index][i])
                    input[row_index][i] = temp
                    input[i][col_index] = temp
        elif linkage == "average":
            for i in range(0, input.shape[0]):
                if i != col_index and i != row_index:
                    temp = (input[row_index][i] + input[col_index][i]) / 2
                    input[row_index][i] = temp
                    input[i][col_index] = temp
        elif linkage == "complete":
            for i in range(0,input.shape[0]):
                if i != col_index and i != row_index:
                    temp = min(input[row_index][i], input[col_index][i])
                    input[row_index][i] = temp
                    input[i][col_index] = temp

        #set the rows and columns for the cluster with higher index i.e. the row index to infinity
        for i in range(0, input.shape[0]):
            input[row_index][i] = np.inf
            input[i][col_index] = np.inf
        #Manipulating the dictionary to keep track of cluster formation in each step
        #if k=0,then all datapoints are clusters
        minimum = min(row_index, col_index)
        maximum = max(row_index, col_index)
        for n in range(len(array)):
            if array[n] == maximum:
                array[n] = minimum
        clusters[k] = array.copy()

    return clusters
