from data_generation import generate_data
from dbscan import dbscan
from plotting import plot_data
from accuracy import calculate_accuracy
from sklearn.cluster import KMeans, DBSCAN
import hdbscan

# Data generation
df = generate_data()  # Get DataFrame with true labels

# Converting data into array
X = df[['Feature 1', 'Feature 2']].values

epsilon = 2
min_pts = 17


# Custom DBSCAN
labels_custom, core_points, noise_points = dbscan(X, epsilon, min_pts)

# Adding labels from custom DBSCAN to the DataFrame
df['Cluster_custom'] = labels_custom

# K-Means clustering
kmeans = KMeans(n_clusters=len(df['Label'].unique()), random_state=0)
labels_kmeans = kmeans.fit_predict(X)

# Adding labels from K-Means to the DataFrame
df['Cluster_kmeans'] = labels_kmeans

# dbscan clustering
dbscan_sklearn = DBSCAN(eps=epsilon, min_samples=min_pts)
labels_dbscan_sklearn = dbscan_sklearn.fit_predict(X)
df['Cluster_dbscan_sklearn'] = labels_dbscan_sklearn

# HDBSCAN clustering
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)  # Ustawienia HDBSCAN
labels_hdbscan = hdbscan_clusterer.fit_predict(X)
df['Cluster_hdbscan'] = labels_hdbscan

# Calculate accuracy for custom DBSCAN
accuracy_custom = calculate_accuracy(df['Label'].values, df['Cluster_custom'].values)
print(f"Accuracy of Custom DBSCAN: {accuracy_custom:.2f}")

# Calculate accuracy for K-Means
accuracy_kmeans = calculate_accuracy(df['Label'].values, df['Cluster_kmeans'].values)
print(f"Accuracy of K-Means: {accuracy_kmeans:.2f}")

# Calculate accuracy for scikit-learn DBSCAN
accuracy_dbscan_sklearn = calculate_accuracy(df['Label'].values, df['Cluster_dbscan_sklearn'].values)
print(f"Accuracy of Scikit-Learn DBSCAN: {accuracy_dbscan_sklearn:.2f}")

# Calculate accuracy for HDBSCAN
accuracy_hdbscan = calculate_accuracy(df['Label'].values, df['Cluster_hdbscan'].values)
print(f"Accuracy of HDBSCAN: {accuracy_hdbscan:.2f}")

# Plotting results
plot_data(df, noise_points, 'Cluster_custom', 'Cluster_kmeans', 'Cluster_dbscan_sklearn', 'Cluster_hdbscan')

