# 4th q
import numpy as np
import matplotlib.pyplot as plt

# Given points
points = np.array([
    [2, 10],
    [2, 5],
    [8, 4],
    [5, 8],
    [7, 5],
    [6, 4],
    [1, 2],
    [4, 9]
])

# Initial cluster centers
initial_centers = np.array([
    [2, 10],
    [5, 8],
    [1, 2]
])

# Function to compute the Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Function to assign points to the nearest cluster center
def assign_clusters(points, centers):
    clusters = {}
    for i in range(len(centers)):
        clusters[i] = []
    for point in points:
        distances = [euclidean_distance(point, center) for center in centers]
        cluster = distances.index(min(distances))
        clusters[cluster].append(point)
    return clusters

# Function to recalculate the cluster centers
def recalculate_centers(clusters):
    new_centers = []
    for cluster in clusters.values():
        new_center = np.mean(cluster, axis=0)
        new_centers.append(new_center)
    return new_centers

# K-means clustering algorithm
def kmeans(points, initial_centers, max_iterations=100):
    centers = initial_centers
    for _ in range(max_iterations):
        clusters = assign_clusters(points, centers)
        new_centers = recalculate_centers(clusters)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return centers, clusters

# Run the K-means algorithm
final_centers, clusters = kmeans(points, initial_centers)

print("Final cluster centers:")
for i, center in enumerate(final_centers):
    print(f"Cluster {i+1} center: {center}")

print("\nCluster assignments:")
for i, cluster in clusters.items():
    print(f"Cluster {i+1}: {cluster}")

# Convert final_centers to a numpy array
final_centers = np.array(final_centers)

# Plot the results
colors = ['r', 'g', 'b']
for i, cluster in clusters.items():
    cluster_points = np.array(cluster)
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
plt.scatter(final_centers[:, 0], final_centers[:, 1], color='k', marker='x', s=100, label='Centers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('K-means Clustering')
plt.show()