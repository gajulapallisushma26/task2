import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer purchase data (replace this with your own dataset)
purchase_data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'AmountSpent': [100, 150, 200, 300, 400],
    'Frequency': [5, 10, 8, 12, 15]
}

# Convert to DataFrame
df = pd.DataFrame(purchase_data)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['AmountSpent', 'Frequency']])

# Define the number of clusters
num_clusters = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(scaled_data)

# Assign clusters to the data points
df['Cluster'] = kmeans.labels_

# Print centroids of the clusters
print("Centroids:")
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
print(centroids)

# Display the clustered data
print("\nClustered Data:")
print(df)
