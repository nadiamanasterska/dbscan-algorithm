from sklearn.metrics import accuracy_score
import numpy as np


def calculate_accuracy(original_labels, cluster_labels):
    # Create a mapping of cluster labels to true labels
    unique_clusters = np.unique(cluster_labels)
    true_labels = []

    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points
            continue
        # Find indices of points in the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        # Find the most common original label in this cluster
        most_common_label = np.bincount(original_labels[cluster_indices]).argmax()
        true_labels.extend([most_common_label] * len(cluster_indices))

    # Create a prediction array based on the original labels for the points assigned to clusters
    predictions = np.full_like(original_labels, -1)  # Default to -1 for noise
    for cluster in unique_clusters:
        if cluster == -1:
            continue
        cluster_indices = np.where(cluster_labels == cluster)[0]
        predictions[cluster_indices] = np.bincount(original_labels[cluster_indices]).argmax()

    # Calculate accuracy, ignoring noise points (-1)
    return accuracy_score(original_labels[original_labels != -1], predictions[original_labels != -1])