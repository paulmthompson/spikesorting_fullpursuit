import numpy as np


def reorder_labels(labels):
    """
    Rename labels from 0 to n-1, where n is the number of unique labels.

    Returns None. Input labels is altered in place.

    Following sorting, the labels for individual spikes can be any number up to
    the maximum number of clusters used for initial sorting (worst case
    scenario, this could be 0:M-1, where M is the number of spikes). This
    function reorders the labels so that they nicely go from 0:num_clusters.

    Parameters
    ----------
    labels : numpy ndarray
        A one dimensional array of numerical labels.

    Returns
    -------
    None :
        The array labels is changed in place.
    """

    if labels.size == 0:
        return
    unique_labels = np.unique(labels)
    new_label = 0
    for old_label in unique_labels:
        labels[labels == old_label] = new_label
        new_label += 1

    return None
