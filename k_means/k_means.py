import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


# --- Some utility functions
def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 

    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.

    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))


class KMeans:

    def __init__(self, k=2):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        # TODO: Should be randomized and take k as argument for size
        self.k = k
        self.centroids = None
        pass

    def fit(self, X: pd.DataFrame):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """

        m = X.shape[0]  # Number of rows
        n = X.shape[1]  # Dimensions of data (in this case 2)

        clusters = {}

        # Generate random centroids
        self.centroids = np.random.uniform(
            low=X.min(), high=X.max(), size=(self.k, n))
        distortion = np.inf

        # Repeat until convergence
        while distortion > euclidean_distortion(X, self.predict(X)) + 1e-5:
            # "Old" distortion
            distortion = euclidean_distortion(X, self.predict(X))

            # Reset all clusters to be empty
            for j in range(self.k):
                clusters[j] = []

            # Distance to nearest centroid for each point, initially infinity
            c = np.full(m, np.inf, float)

            # For every datapoint
            for i, row in X.iterrows():
                x0, x1 = row
                point = np.array([x0, x1])

                # Find closest centroid
                for j in range(self.k):
                    # If this centroid is closer than previous, set it as closest
                    if c[i] > euclidean_distance(point, self.centroids[j]):
                        c[i] = euclidean_distance(point, self.centroids[j])
                        nearest_centroid = j
                # Add point to cluster of nearest centroid
                clusters[nearest_centroid].append([point])

            # Update centroid positions to mean of all points in their cluster
            for j in range(self.k):
                self.centroids[j] = np.concatenate(clusters[j]).mean(axis=0)

    def predict(self, X: pd.DataFrame):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement
        cluster_assignments = np.full(X.shape[0], -1, int)

        for i, row in X.iterrows():
            x0, x1 = row
            point = np.array([x0, x1])
            distance = np.inf
            for j in range(self.k):
                if (distance > euclidean_distance(point, self.centroids[j])):
                    distance = euclidean_distance(point, self.centroids[j])
                    cluster_assignments[i] = j
        return cluster_assignments

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


if __name__ == "__main__":
    data_1 = pd.read_csv(
        'C:/Users/Håvard/Github/machine_learning/task-1/k_means/data_1.csv')
    X = data_1[['x0', 'x1']]
    model_1 = KMeans(k=3)  # <-- Should work with default constructor
    model_1.fit(X)
    z = model_1.predict(X)
