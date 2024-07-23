import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
np.random.seed(0)

# Generate random data for three clusters
cluster_1 = np.random.randn(100, 2) + np.array([2, 2])
cluster_2 = np.random.randn(100, 2) + np.array([-2, -2])
cluster_3 = np.random.randn(100, 2) + np.array([2, -2])

# Combine the clusters to create the dataset
X = np.vstack([cluster_1, cluster_2, cluster_3])

# Step 2: Implement the K-Means Algorithm
def kmeans(X, k, max_iters=100, tol=1e-4):
    # Randomly initialize centroids
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_indices]

    for _ in range(max_iters):
        # Assign clusters based on the closest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        cluster_labels = np.argmin(distances, axis=1)

        # Calculate new centroids as the mean of assigned points
        new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return cluster_labels, centroids

# Run k-means clustering
k = 3
cluster_labels, centroids = kmeans(X, k)

# Step 3: Visualize the Clusters
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', marker='o', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering with NumPy')
plt.legend()
plt.show()