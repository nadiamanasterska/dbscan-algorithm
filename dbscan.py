import numpy as np
from collections import deque


def dbscan(X, epsilon, min_pts):
    labels = np.zeros(X.shape[0], dtype=int)  # Unassigned points labeled as 0
    core_points = []  # Core points
    noise_points = []  # Noise points
    cluster_id = 0

    def region_query(point):
        return np.where(np.linalg.norm(X - point, axis=1) < epsilon)[0]

    def expand_cluster(point_idx, neighbors):
        nonlocal cluster_id
        labels[point_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()

            if labels[neighbor_idx] == -1:  # Change noise point to current cluster
                labels[neighbor_idx] = cluster_id

            if labels[neighbor_idx] == 0:  # If the point is unvisited
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(X[neighbor_idx])

                if len(new_neighbors) >= min_pts:
                    queue.extend(new_neighbors)

    for i in range(len(X)):
        if labels[i] != 0:  # Skip if point is already assigned to a cluster
            continue

        neighbors = region_query(X[i])

        if len(neighbors) < min_pts:
            noise_points.append(i)
            labels[i] = -1  # Mark as noise
        else:
            core_points.append(i)
            cluster_id += 1  # Start a new cluster
            expand_cluster(i, neighbors)

    return labels, core_points, noise_points
