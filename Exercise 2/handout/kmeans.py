import numpy as np


def kmeans_fit(data, k, n_iter=100, tol=1.e-4):
    """
    Fit kmeans
    
    Args:
        data ... Array of shape n_samples x n_features
        k    ... Number of clusters
        
    Returns:
        centers   ... Cluster centers. Array of shape k x n_features
    """
    n_samples, n_features = data.shape
    
    # Create a random number generator
    # Use this to avoid fluctuation in k-means performance due to initialisation
    rng = np.random.default_rng(6174)
    
    # Initialise clusters (choose k random samples as initial centroid)
    centroids = data[rng.choice(n_samples, k, replace=False)]
    
    # Iterate the k-means update steps

    for i in range(n_iter):
      indices = kmeans_predict_idx(data, centroids)
      prev_centroids = centroids.copy()
      for c in range(k):
        in_current_cluster = data[indices == c]
        centroids[c] = np.mean(in_current_cluster, axis=0)
      if np.max(centroids - prev_centroids) < tol:
        break
            
    # Return cluster centers
    return centroids


def compute_distance(data, clusters):
    """
    Compute all distances of every sample in data, to every center in clusters.
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features
        
    Returns:
        distances ... n_samples x n_clusters
    """
    
    distances = np.linalg.norm(data[:, np.newaxis] - clusters, axis=2)

    return distances


def kmeans_predict_idx(data, clusters):
    """
    Predict index of closest cluster for every sample
    
    Args:
        data     ... n_samples x n_features
        clusters ... n_clusters x n_features

    Returns:
        indices  ... n_samples
    """

    distances = compute_distance(data, clusters)
    return np.argmin(distances, axis=1)