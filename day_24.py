# Problem: K-Means clustering to segment customers based on behavior
# Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the data
data = pd.read_csv('dataset/Mall_Customers.csv')
# print(data.info())
# print(data.head())

# Step 2: Data Preprocessing
# Drop CustomerID as it is not useful for clustering.
data.drop('CustomerID', axis=1)

# Encode Gender, male - 0, female - 1
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Handle missing value.
#print(data.isnull().sum()) # We don't have any missing value.

# Step 3: Create features for Cluster
X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Apply Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply Elbow method to find the optimal number of clusters (K)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    # What k-means++ Does: The k-means++ initialization method is a smart way of choosing the initial centroids.
    # Instead of choosing random points, k-means++ spreads out the initial centroids as follows:
    #
    # It first randomly selects one centroid from the data points.
    # Then, for each remaining centroid, it selects the next centroid from the data points that are far from the
    # already chosen centroids. The probability of choosing a data point as a centroid is proportional to its
    # distance from the nearest already chosen centroid.

# Plot the Elbow curve
plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Curve')
plt.xlabel('K value')
plt.ylabel('Inertia')
plt.show()

# Step 6: Apply K Means on the optimal number of clusters (K = 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 7: Visualize the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_scaled[:, 2], y=X_scaled[:, 3], hue=y_kmeans, palette="viridis", s=100)
# We are plotting the Annual Income (x-axis) against the Spending Score (y-axis) for each customer.
plt.title('Customer Segments (K Means Cluster)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# Step 8: Add the clusters information for each row to original data
data['Clusters'] = y_kmeans
print(data)