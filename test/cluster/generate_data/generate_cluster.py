import numpy as np
import matplotlib.pyplot as plt

def generate_cluster_2d(
    rng,
    n_samples: int,
    u_x: float,
    u_y: float,
    sigma_xx: float,
    sigma_yy: float,
    sigma_xy: float,
):
    cluster = rng.multivariate_normal(
        [u_x, u_y], [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]], size=n_samples
    )

    return cluster


def generate_test_cluster_easy():

    rng = np.random.default_rng(seed=42)

    cluster1 = generate_cluster_2d(
        rng, 1000, 0.0, 4.0, 2.0, 1.0, 1.2)

    cluster2 = generate_cluster_2d(
        rng, 1000, 1.0, 0.0, 1.0, 1.0, 0.0)

    cluster3 = generate_cluster_2d(
        rng, 1000, 0.0, -4.0, 2.0, 1.0, -1.2
    )

    """
    plt.scatter(cluster1[:,0], cluster1[:,1])
    plt.scatter(cluster2[:,0], cluster2[:,1])
    plt.scatter(cluster3[:,0], cluster3[:,1])
    plt.show()
    """

    data = np.concatenate((cluster1, cluster2, cluster3), axis=0)

    labels = np.concatenate(
        (np.ones(1000), np.ones(1000) * 2, np.ones(1000) * 3), axis=0
    )
    """
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    """
    return data, labels