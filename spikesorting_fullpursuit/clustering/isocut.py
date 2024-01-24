import numpy as np
from numpy import linalg as la
from sklearn.neighbors import BallTree

from spikesorting_fullpursuit.c_cython import sort_cython
from spikesorting_fullpursuit.clustering import isotonic, multinomial_gof


def max_sse_window(observed, expected):
    """
    Compute an approximate KS statistic for Kolmogorov-Smirnov test from binned
    data.

    Returns an approximate KS statistic and its approximate p-value as well as
    the point of maximum difference from the input distributions.

    This is intended as a helper function for the current sorting algorithm and
    is NOT a proper Kolmogorov-Smirnov test in its own right.  It is inspired
    by the KS test but violates some fundamental assumptions. The actual KS test
    and associated distributions require that the KS statistic be computed from
    data that are not binned or smoothed.  This function only approximates these
    tests for the sake of speed by by using binned data in the face of
    potentially very large sample sizes.  It does this by computing the sample
    size N as the number of observations in each binned distribution (the sum
    of counts) and penalizing  by a degrees of freedom equal to the number of
    bins (the length of counts). In the current context the number of bins is
    the square root of N.

    Parameters
    ----------
    counts_1, counts_2 : numpy ndarray
        These vectors contain the binned observation counts for the underlying
        distributions to be compared.  It is assumed that both were binned over
        the same range, and are orded in accord with this axis and with each
        other.  The CDF of each distribution will be computed and compared
        between them under this assumption.

    Returns
    -------
    ks : float
        The pseudo KS test statistic as computed from the binned data in
        counts_1 and counts_2 using N values adjusted by the number of bins used.
    I : numpy ndarray of type int
        The index into the counts where the maximum difference in the empirical
        CDF was found (based on the input binned data).  Used to indicate the
        point of maximum divergence between the distributions represented by
        counts_1 and counts_2.
    p_value : float
        The p-value for the KS test between the distributions represented by
        counts_1 and counts_2 as returned by compute_ks4_p_value.

    See Also
    --------
    compute_ks4_p_value

    Indices are sliceable, and at least 0, 1. They are adjusted to include the
    point just before and just after the most deviant SSE window if
    possible."""

    max_sse = 0
    best_start = 0
    best_stop = observed.size
    search_start = True
    d_observed = np.diff(observed)
    d_expected = np.diff(expected)
    check_ind = 0
    while check_ind < observed.size:
        if search_start:
            if check_ind == observed.size - 1:
                break  # Can't look for start at last index
            elif check_ind == 0:
                if observed[check_ind] != expected[check_ind]:
                    # Will act as an "OR" operator when paired with "if" below
                    curr_start = check_ind
                    search_start = False
            if d_observed[check_ind] != d_expected[check_ind]:
                curr_start = check_ind
                search_start = False
        else:
            if check_ind == observed.size - 1:
                # Hit last index so end window and check it
                curr_stop = check_ind + 1
                curr_sse = np.sum(
                    (observed[curr_start:curr_stop] - expected[curr_start:curr_stop])
                    ** 2
                )
                if curr_sse >= max_sse:  # Equal allows finding last most point
                    best_start = curr_start
                    best_stop = curr_stop
                    max_sse = curr_sse
                break
            if (d_observed[check_ind] == d_expected[check_ind]) and (
                d_observed[check_ind] != 0
            ):
                curr_stop = min(check_ind + 1, observed.size)
                curr_sse = np.sum(
                    (observed[curr_start:curr_stop] - expected[curr_start:curr_stop])
                    ** 2
                )
                if curr_sse >= max_sse:  # Equal allows finding last most point
                    best_start = curr_start
                    best_stop = curr_stop
                    max_sse = curr_sse
                search_start = True
        check_ind += 1

    return max_sse, best_start, best_stop


def choose_optimal_cutpoint(
    cutpoint_ind,
    residual_densities,
    x_axis,
):
    """ """
    if cutpoint_ind == 0:
        first_same_index = 0
        last_same_index = 0
        check_ind = 0
        while residual_densities[check_ind + 1] == residual_densities[cutpoint_ind]:
            last_same_index = check_ind + 1
            check_ind += 1
            if check_ind == residual_densities.shape[0] - 1:
                break
    elif cutpoint_ind == residual_densities.shape[0] - 1:
        first_same_index = residual_densities.shape[0] - 1
        last_same_index = residual_densities.shape[0] - 1
        check_ind = residual_densities.shape[0] - 1
        while residual_densities[check_ind - 1] == residual_densities[cutpoint_ind]:
            first_same_index = check_ind - 1
            check_ind -= 1
            if check_ind == 0:
                break
    else:
        first_same_index = cutpoint_ind
        check_ind = cutpoint_ind
        while residual_densities[check_ind - 1] == residual_densities[cutpoint_ind]:
            first_same_index = check_ind - 1
            check_ind -= 1
            if check_ind == 0:
                break
        last_same_index = cutpoint_ind
        check_ind = cutpoint_ind
        while residual_densities[check_ind + 1] == residual_densities[cutpoint_ind]:
            last_same_index = check_ind + 1
            check_ind += 1
            if check_ind == residual_densities.shape[0] - 1:
                break

    # Do not want to cut at 0 or between bin zero and 1; same for end
    # first_same_index = max(first_same_index , 1)
    # last_same_index = min(last_same_index, residual_densities.shape[0] - 1)
    cutpoint = (
        x_axis[first_same_index]
        + (x_axis[last_same_index] - x_axis[first_same_index]) / 2
    )

    return cutpoint


def iso_cut(
    projection: np.ndarray,
    p_value_cut_thresh: float,
):
    """
    This helper function determines the optimal cutpoint given a distribution.
    First, it tests to determine whether the histogram has a single peak
    If not, it returns the optimal cut point.

    Args:
        projection (np.ndarray[double]): [description]
        p_value_cut_thresh (float): [description]
    """
    N = projection.size
    if N <= 2:
        # Don't try any comparison with only two samples since can't split
        # merge_clusters shouldn't get to this point
        return 1.0, None

    num_bins = np.ceil(np.sqrt(N)).astype(np.int64)
    if num_bins < 2:
        num_bins = 2

    smooth_pdf, x_axis, _ = sort_cython.kde(projection, num_bins)
    # smooth_pdf, x_axis, _ = kde_builtin(projection, num_bins)
    if x_axis.size == 1:
        # All data are in same bin so merge (shouldn't happen)
        return 1.0, None
    # Output density of kde does not sum to one, so normalize it.
    smooth_pdf = smooth_pdf / np.sum(smooth_pdf)
    # kde function estimates at power of two spacing levels so compute num_points
    num_points = smooth_pdf.size
    if np.any(np.isnan(smooth_pdf)):
        return 1.0, None  # (shouldn't happen)

    # Approximate observations per spacing used for computing n for statistics
    smooth_pdf[smooth_pdf < 0] = 0
    obs_counts = np.around(smooth_pdf * N).astype(np.int64)
    remove_counts = np.logical_and(obs_counts <= 0, smooth_pdf <= 0)
    smooth_pdf = smooth_pdf[~remove_counts]
    obs_counts = obs_counts[~remove_counts]
    x_axis = x_axis[~remove_counts]
    num_points = obs_counts.shape[0]

    # Generate a triange weighting to bias regression towards center of distribution
    if num_points % 2 == 0:
        iso_fit_weights = np.hstack(
            (np.arange(1, num_points // 2 + 1), np.arange(num_points // 2, 0, -1))
        )
    else:
        iso_fit_weights = np.hstack(
            (np.arange(1, num_points // 2 + 1), np.arange(num_points // 2 + 1, 0, -1))
        )
    (
        densities_unimodal_fit,
        peak_density_ind,
    ) = isotonic.unimodal_prefix_isotonic_regression_l2(smooth_pdf, iso_fit_weights)
    densities_unimodal_fit = densities_unimodal_fit / np.sum(densities_unimodal_fit)
    null_counts = np.around(densities_unimodal_fit * N).astype(np.int64)

    if np.all(obs_counts == null_counts):
        return 1.0, None
    if num_points <= 4:
        m_gof = multinomial_gof.MultinomialGOF(
            obs_counts, densities_unimodal_fit, p_threshold=p_value_cut_thresh
        )
        p_value = m_gof.twosided_exact_test()
        cutpoint = None
        if p_value < p_value_cut_thresh:
            cutpoint_ind = np.argmax(null_counts - obs_counts)
            cutpoint = choose_optimal_cutpoint(
                cutpoint_ind, null_counts - obs_counts, x_axis
            )
            # print("Early EXACT critical cut at p=", p_value,"!")
        return p_value, cutpoint

    sse_left, left_start, left_stop = max_sse_window(
        obs_counts[0 : peak_density_ind + 1], null_counts[0 : peak_density_ind + 1]
    )
    sse_right, right_start, right_stop = max_sse_window(
        obs_counts[peak_density_ind:][-1::-1], null_counts[peak_density_ind:][-1::-1]
    )

    if sse_left > sse_right:
        # max_sse_window returns sliceable indices so don't adjust
        critical_range = np.arange(left_start, left_stop)
        citical_side = "left"
    else:
        # right side values were computed backward in max_sse_window so fix it
        flip_right_start = len(x_axis) - right_stop
        flip_right_stop = len(x_axis) - right_start
        critical_range = np.arange(flip_right_start, flip_right_stop)
        citical_side = "right"

    m_gof = multinomial_gof.MultinomialGOF(
        obs_counts[critical_range],
        densities_unimodal_fit[critical_range],
        p_threshold=p_value_cut_thresh,
    )
    log_combinations = m_gof.get_log_n_total_combinations()
    n_perms = np.int64(np.ceil(1 / p_value_cut_thresh) * 100)
    if log_combinations <= np.log(n_perms):
        # Fewer than n_perms combinations exist so do exact test
        p_value = m_gof.twosided_exact_test()
    else:
        p_value = m_gof.random_perm_test(n_perms=n_perms)

    # Only compute cutpoint if we plan on using it, also skipped if p_value is np.nan
    cutpoint = None
    if p_value < p_value_cut_thresh:
        residual_densities = obs_counts - null_counts
        # Multiply by negative residual densities since isotonic.unimodal_prefix_isotonic_regression_l2 only does UP-DOWN
        residual_densities_fit, _ = isotonic.unimodal_prefix_isotonic_regression_l2(
            -1 * residual_densities[critical_range], np.ones_like(critical_range)
        )
        # A full critical range suggests possible bad fit because distribution is extremely bimodal so check them
        found_dip = False
        if (peak_density_ind < obs_counts.size / 2) and (
            critical_range[-1] == obs_counts.size - 1
        ):
            min_null_fit = np.amin(null_counts)
            for cutpoint_ind in range(0, residual_densities_fit.size):
                if (null_counts[cutpoint_ind] == min_null_fit) and (
                    residual_densities_fit[cutpoint_ind] < 0
                ):
                    # This means the null fit missed a second hump because it was so extreme
                    found_dip = True
                    break
            if cutpoint_ind > 0:
                cutpoint_ind -= 1
        elif (peak_density_ind > obs_counts.size / 2) and (
            critical_range[-1] == obs_counts.size - 1
        ):
            min_null_fit = np.amin(null_counts)
            for cutpoint_ind in range(residual_densities_fit.size - 1, -1, -1):
                if (null_counts[cutpoint_ind] == min_null_fit) and (
                    residual_densities_fit[cutpoint_ind] < 0
                ):
                    # This means the null fit missed a second hump because it was so extreme
                    found_dip = True
                    break
            if cutpoint_ind < (residual_densities_fit.size - 1):
                cutpoint_ind += 1
        if not found_dip:
            if citical_side == "left":
                # Ensure we cut closest to center in event of a tie
                cutpoint_ind = np.argmax(residual_densities_fit[-1::-1])
                cutpoint_ind = len(critical_range) - cutpoint_ind - 1
            else:
                cutpoint_ind = np.argmax(residual_densities_fit)
        cutpoint = choose_optimal_cutpoint(
            cutpoint_ind, residual_densities_fit, x_axis[critical_range]
        )

    return p_value, cutpoint


def merge_clusters(
    data,
    labels,
    p_value_cut_thresh=0.01,
    whiten_clusters=True,
    merge_only=False,
    split_only=False,
    max_iter=20000,
    match_cluster_size=False,
    check_splits=False,
    verbose=False,
) -> np.ndarray:
    """
    merge_clusters(data, labels [, ])

    This is the workhorse function when performing clustering. It joins individual
    clusters together to form larger clusters until it reaches convergence.  Returns
    the cluster labels for each spike input.

    Explanation of parameters:
    - comparison_pca. When we choose two clusters to compare, should we re-compute
    the principle components (using these "local" scores rather than the global
    components). This can help separate larger clusters into smaller clusters.
     - merge_only. Only perform merges, do not split.

    Args:
        data (np.ndarray[double]): These are typically scores from dimensionality
            reduction, (samples x dimensions)
        labels (np.ndarray[int64]): Label for each sample (samples)
    Returns:
        labels (np.ndarray[int64]): Labels for each sample
    """

    def whiten_cluster_pairs(
        scores: np.ndarray,
        labels: np.ndarray,
        c1: int,
        c2: int,
    ) -> np.ndarray:
        """

        Args:
            scores (np.ndarray[double]): [description]
            labels (np.ndarray[int64]): [description]
            c1 (int): [description]
            c2 (int): [description]

        Returns:
            np.ndarray[double]
        """
        centroid_1 = sort_cython.compute_cluster_centroid(scores, labels, c1)
        centroid_2 = sort_cython.compute_cluster_centroid(scores, labels, c2)
        V = centroid_2 - centroid_1

        avg_cov = np.cov(
            scores[np.logical_or(labels == c1, labels == c2), :], rowvar=False
        )

        if np.abs(la.det(avg_cov)) > 1e-6:
            inv_average_covariance = la.inv(avg_cov)
            V = np.matmul(V, inv_average_covariance)

        return V

    def create_matched_cluster(
        scores: np.ndarray,
        labels: np.ndarray,
        c1: int,
        c2: int,
    ) -> np.ndarray:
        """
        Finds the smallest and largest cluster of the labels c1 and c2.
        Then chooses a matched number of points from
        the larger cluster that are nearest the smallest cluster centroid.
        These are assigned a new label (the maximum
        value of a numpy.int64) so that merge and isocut can be performed
        on this matched subset of nearest points. Returns
        the label of the smallest cluster input and the new cluster.

        Args:
            scores (np.ndarray[double]): [description]
            labels (np.ndarray[int64]): [description]
            c1 (int): [description]
            c2 (int): [description]

        Returns:
            np.ndarray[double]
        """
        # Use an output label equal to the maximum integer value, assuming there are not nearly this many labels used in "labels"
        matched_label = np.iinfo(np.int64).max
        if np.count_nonzero(labels == c1) >= np.count_nonzero(labels == c2):
            larger_cluster = c1
            smaller_cluster = c2
        else:
            larger_cluster = c2
            smaller_cluster = c1
        small_centroid = sort_cython.compute_cluster_centroid(
            scores, labels, smaller_cluster
        )

        n_small = np.count_nonzero(labels == smaller_cluster)
        select_large = labels == larger_cluster
        indices_large = np.where(select_large)[0]
        large_distances = np.sum(
            (scores[select_large, :] - small_centroid) ** 2, axis=1
        )
        large_order = np.argsort(large_distances)[0:n_small]
        # Give the n_small nearest points from the large cluster a new label
        labels[indices_large[large_order]] = matched_label

        return smaller_cluster, matched_label

    def merge_test(
        scores: np.ndarray,
        labels: np.ndarray,
        c1: int,
        c2: int,
        match_cluster=False,
        check_iso_splits=False,
    ) -> bool:
        """
        This helper function determines if we should perform merging of two
        clusters. This function returns a boolean if we should merge the
        clusters, and reassigns labels in-place according to the iso_cut
        split otherwise

        Args:
            scores (np.ndarray[double]): [description]
            labels (np.ndarray[int64]): [description]
            c1 (int): [description]
            c2 (int): [description]
            match_cluster:
            check_iso_splits:

        Returns:
            bool
        """
        # Save the labels so we can revert to them at the end if needed
        original_labels = np.copy(labels)
        if match_cluster:
            smaller_cluster, matched_label = create_matched_cluster(
                scores, labels, c1, c2
            )
        else:
            if np.count_nonzero(labels == c1) >= np.count_nonzero(labels == c2):
                matched_label = c1
                smaller_cluster = c2
            else:
                matched_label = c2
                smaller_cluster = c1
        if scores.shape[1] > 1:
            # Get V, the vector connecting the two centroids either
            # with or without whitening
            if whiten_clusters:
                V = whiten_cluster_pairs(scores, labels, smaller_cluster, matched_label)
            else:
                centroid_1 = sort_cython.compute_cluster_centroid(
                    scores, labels, smaller_cluster
                )
                centroid_2 = sort_cython.compute_cluster_centroid(
                    scores, labels, matched_label
                )
                V = centroid_2 - centroid_1

            norm_V = la.norm(V)
            if norm_V == 0:
                # The two cluster centroids are identical so merge
                return True

            # Scale by the magnitude to get a unit vector in the
            # appropriate direction
            V = V / norm_V
            # Compute the projection of all points from C1 and C2 onto the line
            projection = np.matmul(scores, V)

        else:
            # Can't whiten one and project one dimensional scores,
            # they are already the 1D projection
            projection = np.squeeze(np.copy(scores))

        p_value, optimal_cut = iso_cut(
            projection[
                np.logical_or(labels == smaller_cluster, labels == matched_label)
            ],
            p_value_cut_thresh,
        )
        # Now that we have done the iso_cut test and found any cutpoints,
        # we don't need matched labels anymore so revert
        if smaller_cluster == c1:
            # Larger cluster must have been c2
            labels[labels == matched_label] = c2
        elif smaller_cluster == c2:
            # Larger cluster must have been c1
            labels[labels == matched_label] = c1
        else:
            # I must have made a horrible mistake
            raise RuntimeError("Lost track of the labels")
        if p_value >= p_value_cut_thresh:  # or np.isnan(p_value):
            # These two clusters should be combined
            if split_only:
                return False
            else:
                return True
        elif merge_only:
            # These clusters should be split, but our options say
            # no with merge only.
            return False
        else:
            # Reassign based on the optimal value
            select_greater = np.logical_and(
                np.logical_or(labels == c1, labels == c2),
                (projection > optimal_cut + 1e-6),
            )
            select_less = np.logical_and(
                np.logical_or(labels == c1, labels == c2), ~select_greater
            )

            # Get mean and distance measures for the original labels so we can check this split and assign labels
            center_1_orig = np.mean(projection[labels == c1])
            center_2_orig = np.mean(projection[labels == c2])
            if check_iso_splits:
                # We will first check how the current split alters the distances between the nearest neighboring points of clusters
                # and if it looks bad, revert the split
                clust1 = projection[labels == c1].reshape(-1, 1)
                clust2 = projection[labels == c2].reshape(-1, 1)
                if clust1.shape[0] >= clust2.shape[0]:
                    ball_array = clust1
                    test_array = clust2
                else:
                    ball_array = clust2
                    test_array = clust1
                tree = BallTree(ball_array)
                distances, _ = tree.query(test_array, k=1)
                test_10_percent = int(np.ceil(0.10 * test_array.shape[0]))
                raw_dist_between_orig = np.mean(
                    np.sort(distances.ravel())[:test_10_percent]
                )

                # No that we have reassigned labels according to the split, get the new distance info
                clust1 = projection[labels == c1].reshape(-1, 1)
                clust2 = projection[labels == c2].reshape(-1, 1)
                if clust1.shape[0] >= clust2.shape[0]:
                    ball_array = clust1
                    test_array = clust2
                else:
                    ball_array = clust2
                    test_array = clust1
                tree = BallTree(ball_array)
                distances, _ = tree.query(test_array, k=1)
                test_10_percent = min(
                    test_10_percent, int(np.ceil(0.10 * test_array.shape[0]))
                )
                raw_dist_between_post = np.mean(
                    np.sort(distances.ravel())[:test_10_percent]
                )
            else:
                # This will force skipping the revert labels below and go straight to reassignment accordng to cut point
                raw_dist_between_post = 1
                raw_dist_between_orig = 0

            if raw_dist_between_post <= raw_dist_between_orig:
                # If the 10% nearest neighbors are closer after split than before, this might be a bad split so undo
                labels[:] = original_labels
            else:
                # Reassign labels to cluster keeping label numbers that minimize
                # projection distance between original and new center
                if center_1_orig >= center_2_orig:
                    labels[select_greater] = c1
                    labels[select_less] = c2
                else:
                    labels[select_greater] = c2
                    labels[select_less] = c1

            if (
                np.count_nonzero(labels == c1) == 0
                or np.count_nonzero(labels == c2) == 0
            ):
                # Our optimal split forced a merge. This can happen even with
                # 'split_only' set to True.
                return True
            return False

    # START ACTUAL OUTER FUNCTION
    if labels.size == 0:
        return labels
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size <= 1:
        # Return all labels merged into most prevalent label
        labels[:] = unique_labels[np.argmax(u_counts)]
        return labels
    elif data is None:
        return labels
    elif data.ndim == 1 or data.shape[0] == 1:
        return labels
    if data.size == 1:
        # PCA failed because there is no variance between data points
        # so return all labels as the same, merged into most common label
        labels[:] = unique_labels[np.argmax(u_counts)]
        return labels

    previously_compared_pairs = []
    num_iter = 0
    none_merged = True
    while True:
        if num_iter > max_iter:
            print("Maximum number of iterations exceeded")
            return labels

        minimum_distance_pairs = sort_cython.identify_clusters_to_compare(
            data, labels, previously_compared_pairs
        )
        if len(minimum_distance_pairs) == 0 and none_merged:
            break  # Done, no more clusters to compare
        none_merged = True
        for c1, c2 in minimum_distance_pairs:
            if verbose:
                print("Comparing ", c1, " with ", c2)

            n_c1 = np.count_nonzero(labels == c1)
            n_c2 = np.count_nonzero(labels == c2)
            if (n_c1 > 1) and (n_c2 > 1):
                # Need more than 1 spike in each matched cluster
                merge = merge_test(
                    data,
                    labels,
                    c1,
                    c2,
                    match_cluster=match_cluster_size,
                    check_iso_splits=check_splits,
                )
            elif (n_c1 > 1) or (n_c2 > 1):
                merge = merge_test(
                    data,
                    labels,
                    c1,
                    c2,
                    match_cluster=False,
                    check_iso_splits=check_splits,
                )
            else:
                # c1 and c2 have one spike each so merge them (algorithm can't
                # split in this case and they are mutually closest pairs)
                merge = True
            if merge:
                # Combine the two clusters together, merging into larger cluster label
                if n_c1 >= n_c2:
                    labels[labels == c2] = c1
                else:
                    labels[labels == c1] = c2
                if verbose:
                    print(
                        "Iter: ",
                        num_iter,
                        ", Unique clusters: ",
                        np.unique(labels).size,
                    )
                none_merged = False
                # labels changed, so any previous comparison is no longer valid and is removed
                for ind, pair in reversed(list(enumerate(previously_compared_pairs))):
                    if c1 in pair or c2 in pair:
                        del previously_compared_pairs[ind]
            else:
                previously_compared_pairs.append((c1, c2))
                if verbose:
                    print("split clusters, ", c1, c2)
        num_iter += 1

    return labels
