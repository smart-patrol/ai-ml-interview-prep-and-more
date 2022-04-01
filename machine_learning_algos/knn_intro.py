import numpy as np
from matplotlib.pyplot as plt

# source: https://tomaszgolan.github.io/introduction_to_machine_learning/markdown/introduction_to_machine_learning_01_knn/introduction_to_machine_learning_01_knn/

class NearestNeighbor():
    """Nerest Neighbor Classifier"""

    def __init__(self) -> None:
        """set distance defition: 0 - L1, 1 - L2"""
        if distance == 0:
            self.distance = np.abs
        elif distance == 1:
            self.distance = np.square
        else:
            raise Exception("Distance not defined.")
    
    def train(self,x:np.array, y:np.array) -> None:
        """Train the classifier (save training data)
        
        x -- feature vectors (N x D)
        y -- labels (N x 1)
        """
        self.x_train = x
        self.y_train = y
    
    def predict(self,x:np.array) -> np.array:
        """Predict labels for test data using the classifier
        
        x -- feature vectors (N x D)
        """
        predictions = []

        # loop over each testing sample
        for x_test in x:
            # array of distances between current teste and all training samples
            distances = np.sum(self.distance(self.x_train - x_test), axis=1)
            # get the closest one
            min_index = np.argmin(distances)
            # add corresponding label
            predictions.append(self.y_train[min_index])

        return predictions


class Analysis():
  """Apply NearestNeighbor to generated (uniformly) test samples."""

  def __init__(self, *x, distance):
    """Generate labels and initilize classifier

    x -- feature vectors arrays
    distance -- 0 for L1, 1 for L2    
    """
    # get number of classes
    self.nof_classes = len(x)

    # create lables array
    # np.ones creates an array of given shape filled with 1 of given type
    # we apply consecutive integer numbers as class labels
    # ravel return flatten array
    y = [i * np.ones(_x.shape[0], dtype=np.int) for i, _x in enumerate(x)]
    y = np.array(y).ravel()

    # save training samples to plot them later
    self.x_train = x

    # merge feature vector arrays for NearestNeighbor
    x = np.concatenate(x, axis=0)

    # train classifier
    self.nn = NearestNeighbor(distance)
    self.nn.train(x, y)


  def prepare_test_samples(self, low=0, high=2, step=0.01):
    """Generate a grid with test points (from low to high with step)"""
    # remember range
    self.range = [low, high]

    # start with grid of points from [low, high] x [low, high]
    grid = np.mgrid[low:high+step:step, low:high+step:step]

    # convert to an array of 2D points
    self.x_test = np.vstack([grid[0].ravel(), grid[1].ravel()]).T


  def analyse(self):
    """Run classifier on test samples and split them according to labels."""

    # find labels for test samples 
    self.y_test = self.nn.predict(self.x_test)

    self.classified = []  # [class I test points, class II test ...]

    # loop over available labels
    for label in range(self.nof_classes):
      # if i-th label == current label -> add test[i]
      class_i = np.array([self.x_test[i] \
                          for i, l in enumerate(self.y_test) \
                          if l == label])
      self.classified.append(class_i)


  def plot(self, t=''):
    """Visualize the result of classification"""
    plot = init_plot(self.range, self.range)
    plot.set_title(t)
    plot.grid(False)

    # plot training samples
    for i, x in enumerate(self.x_train):
      plot.plot(*x.T, mpl_colors[i] + 'o')

    # plot test samples
    for i, x in enumerate(self.classified):
      plot.plot(*x.T, mpl_colors[i] + ',')

class kNearestNeighbors(NearestNeighbor):
    """k-Nearest Neighbor Classifier"""

    def __init__(self) -> None:
        super().__init__()(self, k=1, distance=0)
        """set distance defition: 0 - L1, 1 - L2"""
        super().__init__(distance)
        self.k = k

    def predict(self, x:np.array) -> None:
        """Predict and return lables for each feature vector from x
        
        x -- feature vectors (N X D)
        """
        predictions = []
        # no of classes = max label
        nof_classes = np.amax(self.y_train) + 1
        # loop over all test samples
        for x_test in x:
            # array of distances between current test sample and all training samples
            distances = np.sum(self.distance(self.x_train - x_test), axis=1)
            # placeholder for labels votes
            votes = np.zeros(nof_classes,dtype=np.int)
            # find k closet neighbors and vote
            # # argsort returns the indices that would sort an array
            # # so indices of nearest neighbors
            # # we take self.k first
            for neighbor_id in np.argsort(distances)[:self.k]:
                # closest neighbor label
                neighbor_label = self.y_train[neighbor_id]
                # update votes array
                votes[neighbor_label] += 1
            
            # predicted label is one with most votes
            predictions.append(np.argmax(votes))

        return predictions

class kAnalysis(Analysis):
    """Apply kNearestNeighbor to generated (uniformly) test samples."""
    
    def __init__(self, *x, k=1, distance=1):
        """Generate labels and initilize classifier

            x -- feature vectors arrays
            k -- number of nearest neighbors
            distance -- 0 for L1, 1 for L2    
        """
        # get number of classes
        self.nof_classes = len(x)

        # create lables array
        y = [i * np.ones(_x.shape[0], dtype=np.int) for i, _x in enumerate(x)]
        y = np.array(y).ravel()

        # save training samples to plot them later
        self.x_train = x

        # merge feature vector arrays for NearestNeighbor
        x = np.concatenate(x, axis=0)

        # train classifier (knn this time)
        self.nn = kNearestNeighbors(k, distance)
        self.nn.train(x, y)


def generate_random_points(size=10, low=0, high=1):
    """Generate a set of random 2D point
    
        size -- number of points to generate
        low  -- min value
        high -- max value
    """
    # random_sample([size]) returns random numbers with shape defined by size
    # e.g.
    # >>> np.random.random_sample((2, 3))
    #
    # array([[ 0.44013807,  0.77358569,  0.64338619],
    #        [ 0.54363868,  0.31855232,  0.16791031]])
    #
    return (high - low) * np.random.random_sample((size, 2)) + low

def generate4(n=50):
    """Generate 4 sets of random points."""
    # points from [0, 1] x [0, 1]
    X1 = generate_random_points(n, 0, 1)
    # points from [1, 2] x [1, 2]
    X2 = generate_random_points(n, 1, 2)
    # points from [0, 1] x [1, 2]
    X3 = np.array([[x, y+1] for x,y in generate_random_points(n, 0, 1)])
    # points from [1, 2] x [0, 1]
    X4 = np.array([[x, y-1] for x,y in generate_random_points(n, 1, 2)])
    
    return X1, X2, X3, X4


if __name__ == "__main__":
# generate some data
    x1 = np.random.randn(100, 2) + np.array([-2, 2])
    x2 = np.random.randn(100, 2) + np.array([2, 2])
    x3 = np.random.randn(100, 2) + np.array([0, -2])
    
    print("L1 test")
    # create an analysis object
    analysis = Analysis(x1, x2, x3)
    # prepare test samples
    analysis.prepare_test_samples(low=-5, high=5, step=0.1)
    # run classifier
    analysis.analyse()
    # plot the result
    analysis.plot('KNN classification')

    print("L2 test")
    analysis = Analysis(x1, x2, x3, distance=1)
    analysis.prepare_test_samples(low=-5, high=5, step=0.1)
    analysis.analyse()
    analysis.plot('KNN classification')

    # generate 4 classes of 2D points
    X1, X2, X3, X4 = generate4()

    # add some noise by applying gaussian to every point coordinates
    noise = lambda x, y: [np.random.normal(x, 0.1), np.random.normal(y, 0.1)]

    X1 = np.array([noise(x, y) for x, y in X1])
    X2 = np.array([noise(x, y) for x, y in X2])
    X3 = np.array([noise(x, y) for x, y in X3])
    X4 = np.array([noise(x, y) for x, y in X4])

    # apply kNN with k=1 on the same set of training samples
    knn = kAnalysis(X1, X2, X3, X4, k=1, distance=1)
    knn.prepare_test_samples()
    knn.analyse()
    knn.plot()

    # training size = 50
    # let's check a few values between 1 and 50
    for k in (1, 5, 10, 50):
        knn = kAnalysis(X1, X2, X3, X4, k=k, distance=1)
        knn.prepare_test_samples()
        knn.analyse()
        knn.plot("k = {}".format(k))