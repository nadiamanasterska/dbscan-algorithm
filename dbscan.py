# dbscan.py

import numpy as np
from collections import deque

def dbscan(X, epsilon, min_pts):
    labels = np.full(X.shape[0], -1)  # Unassigned points
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

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            if labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(X[neighbor_idx])

                if len(new_neighbors) >= min_pts:
                    queue.extend(new_neighbors)

    for i in range(len(X)):
        if labels[i] != -1:
            continue

        neighbors = region_query(X[i])

        if len(neighbors) < min_pts:
            noise_points.append(i)
            labels[i] = -1
        else:
            core_points.append(i)
            cluster_id += 1
            expand_cluster(i, neighbors)

    return labels, core_points, noise_points
