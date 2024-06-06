import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




# Convert to DataFrame
df = pd.read_csv("two.csv")

# Save the DataFrame to a CSV file
# df.to_csv('ids.csv', index=False)

print(df.head())

# Extract the features for clustering
X = df[['var1', 'var2']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add the cluster labels to the DataFrame
df['cluster'] = kmeans.labels_

print("Cluster assignments:")
print(df)

# Visualize the clusters
plt.scatter(df['var1'], df['var2'], c=df['cluster'], cmap='viridis', marker='o')
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('K-Means Clustering')
plt.colorbar(label='Cluster')
plt.show()

