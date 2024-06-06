import numpy as np


def initialize_centers(points, k):
    # Randomly select k points as initial centers
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(points.shape[0], k, replace=False)
    return points[indices]


def assign_points_to_clusters(points, centers):
    distances = np.sqrt(((points - centers[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)
def update_centers(points, labels, k):
    # Update cluster centers based on the mean of points assigned to each cluster
    new_centers = np.zeros((k, points.shape[1]))
    for i in range(k):
        new_centers[i] = np.mean(points[labels == i], axis=0)
    return new_centers


def kmeans(points, k, max_iterations=100, tol=1e-4):
    # Initialize cluster centers
    centers = initialize_centers(points, k)

    for _ in range(max_iterations):
        # Assign points to clusters
        labels = assign_points_to_clusters(points, centers)

        # Update cluster centers
        new_centers = update_centers(points, labels, k)

        # Check convergence
        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    return centers, labels


# Given points and initial cluster centers
points = np.array([[2, 2],
                   [1, 1],
                   [1.5, 0.5],
                   [3, 1],
                   [3, 2]])

initial_centers = np.array([[2, 2],
                            [1, 1]])

# Perform K-means clustering
cluster_centers, labels = kmeans(points, k=2)

# Print cluster centers and labels
print("Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i + 1}: {center}")

print("\nCluster Labels:")
for i, label in enumerate(labels):
    print(f"Point {chr(65 + i)}: Cluster {label + 1}")
