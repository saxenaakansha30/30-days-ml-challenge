# Problem: Cluster grocery store customers based on purchase history with K-Means
# Dataset: https://archive.ics.uci.edu/dataset/292/wholesale+customers
# Unsupervised learning.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Load the data
data = pd.read_csv('dataset/wholesale_customers_data.csv')
# print(data.head())

# Step 2: Preprocess the data
# Check for missing values.
# print(data.isnull().sum())

# If no missing value, proceed with scaling the features (Normalization)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Apply K-Means Clustering

# Trying different numbers of clusters (K)
sum_sq_dist_pt = [] # Sum of squared distances of each K
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(data_scaled)
    sum_sq_dist_pt.append(kmeans.inertia_) # Inertia is the sum of squared distances to the nearest cluster center


# Plot the elbow curve to know about the optimal K
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), sum_sq_dist_pt, marker='o',)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of squared distances (Inertia)')
plt.title('Elbow method for optimal K')
plt.show()
# From the Elbow curve 7 looks to be the Optimal K.

# Step 4: Select the optimal K and train K-Means
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
kmeans.fit(data_scaled)

# Add the cluster label to original dataset.
data['cluster'] = kmeans.labels_
print(data.head())

# Step 5: Visualize the clusters.

# Here, you initialize the PCA object and specify n_components=2, which means you want to reduce
# the data down to 2 dimensions.
#
# Why?: Since we cannot visualize data with more than 3 dimensions easily, PCA helps us reduce the dataset to two
# principal components (2D), capturing the majority of the variance in the data.
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10,7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans.labels_, cmap='viridis', label=kmeans.labels_)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means clustering with PCA reduced data')
plt.show()