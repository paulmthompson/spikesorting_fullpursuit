import numpy as np

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
                break # Can't look for start at last index
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
                curr_stop = check_ind+1
                curr_sse = np.sum((observed[curr_start:curr_stop] - expected[curr_start:curr_stop]) ** 2)
                if curr_sse >= max_sse: # Equal allows finding last most point
                    best_start = curr_start
                    best_stop = curr_stop
                    max_sse = curr_sse
                break
            if (d_observed[check_ind] == d_expected[check_ind]) and (d_observed[check_ind] != 0):
                curr_stop = min(check_ind + 1, observed.size)
                curr_sse = np.sum((observed[curr_start:curr_stop] - expected[curr_start:curr_stop]) ** 2)
                if curr_sse >= max_sse: # Equal allows finding last most point
                    best_start = curr_start
                    best_stop = curr_stop
                    max_sse = curr_sse
                search_start = True
        check_ind += 1

    return max_sse, best_start, best_stop


def choose_optimal_cutpoint(cutpoint_ind, residual_densities, x_axis):
    """
    """
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
    cutpoint = x_axis[first_same_index] + (
                    x_axis[last_same_index] - x_axis[first_same_index])/2

    return cutpoint


def iso_cut(
        projection: np.ndarray,
        p_value_cut_thresh: float
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
        return 1., None

    num_bins = np.ceil(np.sqrt(N)).astype(np.int64)
    if num_bins < 2:
        num_bins = 2

    smooth_pdf, x_axis, _ = sort_cython.kde(projection, num_bins)
    # smooth_pdf, x_axis, _ = kde_builtin(projection, num_bins)
    if x_axis.size == 1:
        # All data are in same bin so merge (shouldn't happen)
        return 1., None
    # Output density of kde does not sum to one, so normalize it.
    smooth_pdf = smooth_pdf / np.sum(smooth_pdf)
    # kde function estimates at power of two spacing levels so compute num_points
    num_points = smooth_pdf.size
    if np.any(np.isnan(smooth_pdf)):
        return 1., None  # (shouldn't happen)

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
        iso_fit_weights = np.hstack((np.arange(1, num_points // 2 + 1), np.arange(num_points // 2, 0, -1)))
    else:
        iso_fit_weights = np.hstack((np.arange(1, num_points // 2 + 1), np.arange(num_points // 2 + 1, 0, -1)))
    densities_unimodal_fit, peak_density_ind = isotonic.unimodal_prefix_isotonic_regression_l2(smooth_pdf, iso_fit_weights)
    densities_unimodal_fit = densities_unimodal_fit / np.sum(densities_unimodal_fit)
    null_counts = np.around(densities_unimodal_fit * N).astype(np.int64)

    if np.all(obs_counts == null_counts):
        return 1., None
    if num_points <= 4:
        m_gof = multinomial_gof.MultinomialGOF(
                    obs_counts,
                    densities_unimodal_fit,
                    p_threshold=p_value_cut_thresh)
        p_value = m_gof.twosided_exact_test()
        cutpoint = None
        if p_value < p_value_cut_thresh:
            cutpoint_ind = np.argmax(null_counts - obs_counts)
            cutpoint = choose_optimal_cutpoint(
                cutpoint_ind,
                null_counts - obs_counts,
                x_axis)
            # print("Early EXACT critical cut at p=", p_value,"!")
        return p_value, cutpoint

    sse_left, left_start, left_stop = max_sse_window(
        obs_counts[0:peak_density_ind+1],
        null_counts[0:peak_density_ind+1]
        )
    sse_right, right_start, right_stop = max_sse_window(
        obs_counts[peak_density_ind:][-1::-1],
        null_counts[peak_density_ind:][-1::-1]
        )

    if sse_left > sse_right:
        # max_sse_window returns sliceable indices so don't adjust
        critical_range = np.arange(left_start, left_stop)
        citical_side = 'left'
    else:
        # right side values were computed backward in max_sse_window so fix it
        flip_right_start = len(x_axis) - right_stop
        flip_right_stop = len(x_axis) - right_start
        critical_range = np.arange(flip_right_start, flip_right_stop)
        citical_side = 'right'

    m_gof = multinomial_gof.MultinomialGOF(
                obs_counts[critical_range],
                densities_unimodal_fit[critical_range],
                p_threshold=p_value_cut_thresh)
    log_combinations = m_gof.get_log_n_total_combinations()
    n_perms = np.int64(np.ceil(1/p_value_cut_thresh) * 100)
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
        residual_densities_fit, _ = isotonic.unimodal_prefix_isotonic_regression_l2(-1 * residual_densities[critical_range], np.ones_like(critical_range))
        # A full critical range suggests possible bad fit because distribution is extremely bimodal so check them
        found_dip = False
        if ( (peak_density_ind < obs_counts.size/2) and (critical_range[-1] == obs_counts.size - 1) ):
            min_null_fit = np.amin(null_counts)
            for cutpoint_ind in range(0, residual_densities_fit.size):
                if ( (null_counts[cutpoint_ind] == min_null_fit) and (residual_densities_fit[cutpoint_ind] < 0) ):
                    # This means the null fit missed a second hump because it was so extreme
                    found_dip = True
                    break
            if cutpoint_ind > 0:
                cutpoint_ind -= 1
        elif ( (peak_density_ind > obs_counts.size/2) and (critical_range[-1] == obs_counts.size - 1) ):
            min_null_fit = np.amin(null_counts)
            for cutpoint_ind in range(residual_densities_fit.size - 1, -1, -1):
                if ( (null_counts[cutpoint_ind] == min_null_fit) and (residual_densities_fit[cutpoint_ind] < 0) ):
                    # This means the null fit missed a second hump because it was so extreme
                    found_dip = True
                    break
            if cutpoint_ind < (residual_densities_fit.size - 1):
                cutpoint_ind += 1
        if not found_dip:
            if citical_side == 'left':
                # Ensure we cut closest to center in event of a tie
                cutpoint_ind = np.argmax(residual_densities_fit[-1::-1])
                cutpoint_ind = len(critical_range) - cutpoint_ind - 1
            else:
                cutpoint_ind = np.argmax(residual_densities_fit)
        cutpoint = choose_optimal_cutpoint(
            cutpoint_ind,
            residual_densities_fit,
            x_axis[critical_range]
            )

    return p_value, cutpoint
