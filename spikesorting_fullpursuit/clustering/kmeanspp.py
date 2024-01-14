import numpy as np


def initial_cluster_farthest(
        data,
        median_cluster_size,
        choose_percentile=0.95,
        n_random=0):
    """
    Create distance based cluster labels along the rows of data.

    Returns a vector containing the labels for each data point.

    Data are iteratively clustered based on Euclidean distance until the median
    number of points in each cluster is <= median_cluster_size or each point is
    the only member of its own cluster. For each iteration, a new cluster center
    is chosen. First, the distance of each point from its nearest cluster center
    is computed. Second, from these distances, the point lying at the 99th
    percentile is chosen to be a new cluster center.  Finally, all points closer
    to this center than their current cluster center are assigned to the new
    cluster and the process is repeated.  This is similar to the k-means++
    algorithm except that it is deterministic, always choosing at the 99th
    percentile.

    Parameters
    ----------
    data : numpy ndarray
        Each row of data will be treated as an observation and each column as a
        dimension over which distance will be computed.  Must be two dimensional.
    median_cluster_size : {int, float, ndarray etc.}
        Must be a single, scalar value regardless of type. New cluster centers
        will be added until the median number of points from data that are
        nearest a cluster center is less than or equal to this number (see
        Notes below).

    Returns
    -------
    labels : numpy ndarray of dtype int64
        A new array holding the numerical labels indicating the membership of
        each point input in data. Array is the same size as data.shape[0].
    """
    if data.ndim <= 1 or data.size == 1:
        # Only 1 spike so return 1 label!
        return np.zeros(1, dtype=np.int64)

    # Begin with a single cluster (all data belong to the same cluster)
    labels = np.zeros((data.shape[0]), dtype=np.int64)
    label_counts = labels.size
    current_num_centers = 0
    if labels.size <= median_cluster_size:
        return labels
    if median_cluster_size <= 2:
        labels = np.arange(0, labels.size, dtype=np.int64)
        return labels
    centers = np.mean(data, axis=0)
    distances = np.sum((data - centers)**2, axis=1)
    if np.all(np.all(distances == 0)):
        # All scores are the same, so return all same label
        return labels

    if n_random > 0:
        if n_random >= labels.size:
            return np.arange(0, labels.size, dtype=np.int64)
        n_random = np.ceil(n_random).astype(np.int64)
        for nl in range(0, n_random):
            rand_ind = np.random.choice(data.shape[0], 1,
                            p=(distances/np.sum(distances)), replace=False)
            current_num_centers += 1
            new_center = data[rand_ind, :]
            centers = np.vstack((centers, new_center))
            temp_distance = np.sum((data - new_center)**2, axis=1)
            select = temp_distance < distances
            labels[select] = current_num_centers
            distances[select] = temp_distance[select]
            _, label_counts = np.unique(labels, return_counts=True)
            if current_num_centers == labels.size:
                break
    pre_centers = current_num_centers

    _, label_counts = np.unique(labels, return_counts=True)
    # Convert percentile to an index
    n_percentile = np.ceil((labels.size-1) * (1 - choose_percentile)).astype(np.int64)
    while np.median(label_counts) > median_cluster_size and current_num_centers < labels.size:
        current_num_centers += 1
        # Partition the negative distances (ascending partition)
        new_index = np.argpartition(-distances, n_percentile)[n_percentile]
        # Choose data at percentile index as the center of the next cluster
        new_center = data[new_index, :]
        centers = np.vstack((centers, new_center))
        temp_distance = np.sum((data - new_center)**2, axis=1)
        # Add any points closer to new center than their previous center
        select = temp_distance < distances
        labels[select] = current_num_centers
        distances[select] = temp_distance[select]
        _, label_counts = np.unique(labels, return_counts=True)

    return labels
