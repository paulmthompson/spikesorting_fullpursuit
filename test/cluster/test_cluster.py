import numpy as np
from generate_data.generate_cluster import generate_test_cluster_easy
from spikesorting_fullpursuit.clustering.kmeanspp import initial_cluster_farthest
from spikesorting_fullpursuit.clustering.isocut import merge_clusters
import matplotlib.pyplot as plt

import pytest

def cluster_default(scores):
    p_value_cut_thresh = 0.01

    median_cluster_size = 100

    neuron_labels = initial_cluster_farthest(scores, median_cluster_size, n_random=0)
    neuron_labels = merge_clusters(
        scores,
        neuron_labels,
        split_only=False,
        p_value_cut_thresh=p_value_cut_thresh,
        match_cluster_size=False,
        check_splits=False,
    )

    return neuron_labels

@pytest.fixture(scope="module")
def easy_scores():
    scores, ground_truth_labels = generate_test_cluster_easy()
    yield scores

@pytest.fixture(scope="module")
def easy_ground_truth():
    scores, ground_truth_labels = generate_test_cluster_easy()
    yield ground_truth_labels

@pytest.fixture()
def cluster_labels(easy_scores):
    neuron_labels = cluster_default(easy_scores)
    yield neuron_labels


def test_cluster_size(cluster_labels, easy_ground_truth):

    assert cluster_labels.size == easy_ground_truth.size


def test_cluster_num_easy(cluster_labels, easy_ground_truth):
    label_ids = np.unique(cluster_labels)

    assert len(label_ids) == len(np.unique(easy_ground_truth))
    """
    for g in label_ids:
        ix = np.where(neuron_labels == g)
        plt.scatter(scores[ix,0], scores[ix,1])
    plt.show()
    """
