import numpy as np
from numpy import linalg as la
from isosplit6 import isosplit6
import spikesorting_fullpursuit.clustering
import spikesorting_fullpursuit.dim_reduce

from spikesorting_fullpursuit.c_cython import sort_cython


def pca_scores(
    spikes, compute_pcs=None, pcs_as_index=True, return_V=False, return_E=False
):
    """
    Given a set of spikes which is an MxN matrix (M spikes x N timepoints), we
    determine the principle components given a set of spikes. The principle
    components are returned in order of decreasing variance (i.e, the
    first components returned have the highest variance).

    Each column in the returned output corresponds to one principle component.
    To compute the "weight" for each PCA, simply multiply matrix wise.
    pca(spikes)[:, 1] * spikes[1, :]
    to get the weight of the first principle component for the first spike.
    If pc_count is a scalar
    """
    if spikes.ndim != 2:
        raise ValueError("Input 'spikes' must be a 2 dimensional array to compute PCA")
    if spikes.shape[0] <= 1:
        raise ValueError("Must input more than 1 spike to compute PCA")
    if compute_pcs is None:
        compute_pcs = spikes.shape[1]
    spike_std = np.std(spikes, axis=0)

    if np.all(spike_std != 0):
        spike_copy = np.copy(spikes)
        spike_copy -= np.mean(spike_copy, axis=0)
        spike_copy /= spike_std
        C = np.cov(spike_copy, rowvar=False)
        E, V = la.eigh(C)

        # If pcs_as_index is true, treat compute_pcs as index of specific components
        # else compute_pcs must be a scalar and we index from 0:compute_pcs
        if pcs_as_index:
            key = np.argsort(E)[::-1][compute_pcs]
        else:
            key = np.argsort(E)[::-1][:compute_pcs]

        E, V = E[key], V[:, key]
        U = np.matmul(spike_copy, V)
    else:
        # No variance, all PCs are equal! Set to None(s)
        U = None
        V = None
        E = None

    if return_V and return_E:
        return U, V, E
    elif return_V:
        return U, V
    elif return_E:
        return U, E
    else:
        return U


def optimal_reconstruction_pca_order(
    spikes, check_components=None, max_components=None, min_components=0
):
    """
    Used as an alternative to 'max_pca_components_cross_validation'.
    This function computes the reconstruction based on each principal component
    separately and then reorders the principal components according to their
    reconstruction accuracy rather than variance accounted for.  It then
    iterates through the reconstructions adding one PC at a time in this
    new order and at each step computing the ratio of improvement
    from the addition of a PC.  All PCs up to and including the first local
    maxima of this VAF function are output as the the optimal ones to use.
    """
    # Limit max-components based on the size of the dimensions of spikes
    if max_components is None:
        max_components = spikes.shape[1]
    if check_components is None:
        check_components = spikes.shape[1]
    max_components = np.amin([max_components, spikes.shape[1]])
    check_components = np.amin([check_components, spikes.shape[1]])

    # Get residual sum of squared error for each PC separately
    resid_error = np.zeros(check_components)
    _, components = pca_scores(
        spikes, check_components, pcs_as_index=False, return_V=True
    )
    if components is None:
        # Couldn't compute PCs
        return np.zeros(1, dtype=np.int64), True
    for comp in range(0, check_components):
        reconstruction = (spikes @ components[:, comp][:, None]) @ components[:, comp][
            :, None
        ].T
        RESS = np.mean(np.mean((reconstruction - spikes) ** 2, axis=1), axis=0)
        resid_error[comp] = RESS

    # Optimal order of components based on reconstruction accuracy
    comp_order = np.argsort(resid_error)

    # Find improvement given by addition of each ordered PC
    vaf = np.zeros(check_components)
    PRESS = np.mean(np.mean((spikes) ** 2, axis=1), axis=0)
    RESS = np.mean(
        np.mean((spikes - np.mean(np.mean(spikes, axis=0))) ** 2, axis=1), axis=0
    )
    vaf[0] = 1.0 - RESS / PRESS

    PRESS = RESS
    for comp in range(1, check_components):
        reconstruction = (spikes @ components[:, comp_order[0:comp]]) @ components[
            :, comp_order[0:comp]
        ].T
        RESS = np.mean(np.mean((reconstruction - spikes) ** 2, axis=1), axis=0)
        vaf[comp] = 1.0 - RESS / PRESS
        PRESS = RESS
        # Choose first local maxima
        if vaf[comp] < vaf[comp - 1]:
            break
        if comp == max_components:
            # Won't use more than this so break
            break

    max_vaf_components = comp

    is_worse_than_mean = False
    if vaf[1] < 0:
        # First PC is worse than the mean
        is_worse_than_mean = True
        max_vaf_components = 1

    # This is to account for slice indexing and edge effects
    if max_vaf_components >= vaf.size - 1:
        # This implies that we found no maxima before reaching the end of vaf
        if vaf[-1] > vaf[-2]:
            # vaf still increasing so choose last point
            max_vaf_components = vaf.size
        else:
            # vaf has become flat so choose second to last point
            max_vaf_components = vaf.size - 1
    if max_vaf_components < min_components:
        max_vaf_components = min_components
    if max_vaf_components > max_components:
        max_vaf_components = max_components

    return comp_order[0:max_vaf_components], is_worse_than_mean


def compute_pca(
    clips,
    check_components,
    max_components,
    add_peak_valley=False,
    curr_chan_inds=None,
    n_samples=1e5,
) -> np.ndarray:
    """
    Compute principal components from spike clips.
    Args:
        clips:
        check_components:
        max_components:
        add_peak_valley:
        curr_chan_inds:
        n_samples:

    Returns:
        PCA from spike clips (samples x n_components
    """
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError(
            "Must supply indices for the main channel if using peak valley"
        )
    # Get a sample of the current clips for PCA reconstruction so that memory
    # usage does not explode. Copy from memmap clips to memory
    # PCA order functions use double precision and are compiled that way
    mem_order = "F" if clips.flags["F_CONTIGUOUS"] else "C"
    if n_samples > clips.shape[0]:
        sample_clips = np.empty(clips.shape, dtype=np.float64, order=mem_order)
        np.copyto(sample_clips, clips)
    else:
        sample_clips = np.empty(
            (int(n_samples), clips.shape[1]), dtype=np.float64, order=mem_order
        )
        sel_inds = np.random.choice(clips.shape[0], int(n_samples), replace=True)
        np.copyto(sample_clips, clips[sel_inds, :])

    if mem_order == "C":
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(
            sample_clips, check_components, max_components
        )
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(
            sample_clips, check_components, max_components
        )
    # Return the PC matrix computed over sampled data
    scores, V = pca_scores(
        sample_clips, use_components, pcs_as_index=True, return_V=True
    )
    if scores is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        return np.zeros((clips.shape[0], 1))
    else:
        # Convert ALL clips to scores
        scores = np.matmul(clips, V)
    if add_peak_valley:
        peak_valley = (
            np.amax(clips[:, curr_chan_inds], axis=1)
            - np.amin(clips[:, curr_chan_inds], axis=1)
        ).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley))  # Normalized from -1 to 1
        peak_valley *= np.amax(
            np.amax(np.abs(scores))
        )  # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))

    return scores


def compute_pca_by_channel(
    clips,
    curr_chan_inds,
    check_components,
    max_components,
    add_peak_valley=False,
    n_samples=1e5,
):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError(
            "Must supply indices for the main channel if using peak valley"
        )
    pcs_by_chan = []
    eigs_by_chan = []
    # Do current channel first
    # use_components, _ = optimal_reconstruction_pca_order(clips[:, curr_chan_inds], check_components, max_components, min_components=0)
    # NOTE: Slicing SWITCHES C and F ordering so check!
    mem_order = (
        "F"
        if clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1].flags["F_CONTIGUOUS"]
        else "C"
    )
    if n_samples > clips.shape[0]:
        sample_clips = np.empty(
            (clips.shape[0], len(curr_chan_inds)), dtype=np.float64, order=mem_order
        )
        np.copyto(sample_clips, clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1])
        sel_inds = np.arange(0, clips.shape[0])
    else:
        sample_clips = np.empty(
            (int(n_samples), len(curr_chan_inds)), dtype=np.float64, order=mem_order
        )
        sel_inds = np.random.choice(clips.shape[0], int(n_samples), replace=True)
        np.copyto(
            sample_clips, clips[sel_inds, curr_chan_inds[0] : curr_chan_inds[-1] + 1]
        )

    if mem_order == "C":
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(
            sample_clips, check_components, max_components
        )
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(
            sample_clips, check_components, max_components
        )
    scores, V, eigs = pca_scores(
        sample_clips, use_components, pcs_as_index=True, return_E=True, return_V=True
    )
    if scores is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        return np.zeros((clips.shape[0], 1))
    else:
        # Convert ALL clips to scores
        scores = np.matmul(clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1], V)
    if add_peak_valley:
        peak_valley = (
            np.amax(clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1], axis=1)
            - np.amin(clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1], axis=1)
        ).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley))  # Normalized from -1 to 1
        peak_valley *= np.amax(
            np.amax(np.abs(scores))
        )  # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))
    pcs_by_chan.append(scores)
    eigs_by_chan.append(eigs)
    n_curr_max = use_components.size

    samples_per_chan = curr_chan_inds.size
    n_estimated_chans = clips.shape[1] // samples_per_chan
    for ch in range(0, n_estimated_chans):
        if ch * samples_per_chan == curr_chan_inds[0]:
            continue
        ch_win = [ch * samples_per_chan, (ch + 1) * samples_per_chan]
        # Copy channel data to memory in sample clips using same clips as before
        np.copyto(sample_clips, clips[sel_inds, ch_win[0] : ch_win[1]])
        if mem_order == "C":
            (
                use_components,
                is_worse_than_mean,
            ) = sort_cython.optimal_reconstruction_pca_order(
                sample_clips, check_components, max_components
            )
        else:
            (
                use_components,
                is_worse_than_mean,
            ) = sort_cython.optimal_reconstruction_pca_order_F(
                sample_clips, check_components, max_components
            )
        if is_worse_than_mean:
            # print("Automatic component detection (get by channel) chose !NO! PCA components.", flush=True)
            continue
        scores, V, eigs = pca_scores(
            clips[:, ch_win[0] : ch_win[1]],
            use_components,
            pcs_as_index=True,
            return_E=True,
            return_V=True,
        )
        if scores is not None:
            # Convert ALL clips to scores
            scores = np.matmul(clips[:, ch_win[0] : ch_win[1]], V)
        pcs_by_chan.append(scores)
        eigs_by_chan.append(eigs)

    # Keep only the max components by eigenvalue
    pcs_by_chan = np.hstack(pcs_by_chan)
    if pcs_by_chan.shape[1] > max_components:
        eigs_by_chan = np.hstack(eigs_by_chan)
        comp_order = np.argsort(eigs_by_chan)
        pcs_by_chan = pcs_by_chan[:, comp_order]
        pcs_by_chan = pcs_by_chan[:, 0:max_components]
        pcs_by_chan = np.ascontiguousarray(pcs_by_chan)

    return pcs_by_chan


def compute_template_pca(
    clips,
    labels,
    curr_chan_inds,
    check_components,
    max_components,
    add_peak_valley=False,
    use_weights=True,
):
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError(
            "Must supply indices for the main channel if using peak valley"
        )
    # Compute the weights using the PCA for neuron templates
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 1:
        # Can't do PCA on one template
        return None
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.mean(clips[labels == l, :], axis=0)
        if use_weights:
            templates[ind, :] *= np.sqrt(u_counts[ind] / labels.size)

    # use_components, _ = optimal_reconstruction_pca_order(templates, check_components, max_components)
    # PCA order functions use double precision and are compiled that way, so cast
    # here and convert back afterward instead of carrying two copies. Scores
    # will then be output as doubles.
    clip_dtype = clips.dtype
    clips = clips.astype(np.float64)
    if templates.flags["C_CONTIGUOUS"]:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(
            templates, check_components, max_components
        )
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(
            templates, check_components, max_components
        )
    # print("Automatic component detection (FULL TEMPLATES) chose", use_components, "PCA components.")
    _, score_mat = pca_scores(
        templates, use_components, pcs_as_index=True, return_V=True
    )
    if score_mat is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        return np.zeros((clips.shape[0], 1))
    scores = clips @ score_mat
    if add_peak_valley:
        peak_valley = (
            np.amax(clips[:, curr_chan_inds], axis=1)
            - np.amin(clips[:, curr_chan_inds], axis=1)
        ).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley))  # Normalized from -1 to 1
        peak_valley *= np.amax(
            np.amax(np.abs(scores))
        )  # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))
    clips = clips.astype(clip_dtype)

    return scores


def compute_template_pca_by_channel(
    clips,
    labels,
    curr_chan_inds,
    check_components,
    max_components,
    add_peak_valley=False,
    use_weights=True,
):
    if curr_chan_inds is None:
        raise ValueError(
            "Must supply indices for the main channel for computing PCA by channel"
        )
    # Compute the weights using the PCA for neuron templates
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 1:
        # Can't do PCA on one template
        return None
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.median(clips[labels == l, :], axis=0)
        if use_weights:
            templates[ind, :] *= np.sqrt(u_counts[ind] / labels.size)

    pcs_by_chan = []
    # Do current channel first
    # use_components, _ = optimal_reconstruction_pca_order(templates[:, curr_chan_inds], check_components, max_components)
    is_c_contiguous = templates[:, curr_chan_inds].flags["C_CONTIGUOUS"]
    # PCA order functions use double precision and are compiled that way, so cast
    # here and convert back afterward instead of carrying two copies. Scores
    # will then be output as doubles.
    clip_dtype = clips.dtype
    clips = clips.astype(np.float64)
    if is_c_contiguous:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order(
            templates[:, curr_chan_inds], check_components, max_components
        )
    else:
        use_components, _ = sort_cython.optimal_reconstruction_pca_order_F(
            templates[:, curr_chan_inds], check_components, max_components
        )
    # print("Automatic component detection (TEMPLATES by channel) chose", use_components, "PCA components.")
    _, score_mat = pca_scores(
        templates[:, curr_chan_inds], use_components, pcs_as_index=True, return_V=True
    )

    if score_mat is None:
        # Usually means all clips were the same or there was <= 1, so PCs can't
        # be computed. Just return everything as the same score
        print("FAILED TO FIND PCS !!!")
        return np.zeros((clips.shape[0], 1))
    scores = clips[:, curr_chan_inds] @ score_mat
    if add_peak_valley:
        peak_valley = (
            np.amax(clips[:, curr_chan_inds], axis=1)
            - np.amin(clips[:, curr_chan_inds], axis=1)
        ).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley))  # Normalized from -1 to 1
        peak_valley *= np.amax(
            np.amax(np.abs(scores))
        )  # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))
    pcs_by_chan.append(scores)
    n_curr_max = use_components.size

    samples_per_chan = curr_chan_inds.size
    for ch in range(0, clips.shape[1] // samples_per_chan):
        if ch * samples_per_chan == curr_chan_inds[0]:
            continue
        ch_inds = np.arange(ch * samples_per_chan, (ch + 1) * samples_per_chan)
        # use_components, is_worse_than_mean = optimal_reconstruction_pca_order(templates[:, ch_inds], check_components, max_components)
        if is_c_contiguous:
            (
                use_components,
                is_worse_than_mean,
            ) = sort_cython.optimal_reconstruction_pca_order(
                templates[:, ch_inds], check_components, max_components
            )
        else:
            (
                use_components,
                is_worse_than_mean,
            ) = sort_cython.optimal_reconstruction_pca_order_F(
                templates[:, ch_inds], check_components, max_components
            )
        if is_worse_than_mean:
            # print("Automatic component detection (TEMPLATES by channel) chose !NO! PCA components.", flush=True)
            continue
        # if use_components.size > n_curr_max:
        #     use_components = use_components[0:n_curr_max]
        # print("Automatic component detection (TEMPLATES by channel) chose", use_components, "PCA components.")
        _, score_mat = pca_scores(
            templates[:, ch_inds], use_components, pcs_as_index=True, return_V=True
        )
        scores = clips[:, ch_inds] @ score_mat
        pcs_by_chan.append(scores)
    clips = clips.astype(clip_dtype)

    return np.hstack(pcs_by_chan)


def branch_pca_2_0(
    neuron_labels,
    clips,
    curr_chan_inds,
    p_value_cut_thresh=0.01,
    add_peak_valley=False,
    check_components=None,
    max_components=None,
    use_rand_init=True,
    method="pca",
    match_cluster_size=False,
    check_splits=False,
):
    """ """
    neuron_labels_copy = np.copy(neuron_labels)
    clusters_to_check = [ol for ol in np.unique(neuron_labels_copy)]
    next_label = int(np.amax(clusters_to_check) + 1)
    while len(clusters_to_check) > 0:
        curr_clust = clusters_to_check.pop()
        curr_clust_bool = neuron_labels_copy == curr_clust
        clust_clips = clips[curr_clust_bool, :]
        if clust_clips.ndim == 1:
            clust_clips = np.expand_dims(clust_clips, 0)
        if clust_clips.shape[0] <= 1:
            # Only one spike so don't try to sort
            continue
        median_cluster_size = min(100, int(np.around(clust_clips.shape[0] / 1000)))

        # Re-cluster and sort using only clips from current cluster
        if method.lower() == "pca":
            scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca(
                clust_clips,
                check_components,
                max_components,
                add_peak_valley=add_peak_valley,
                curr_chan_inds=curr_chan_inds,
            )
        elif method.lower() == "chan_pca":
            scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca_by_channel(
                clust_clips,
                curr_chan_inds,
                check_components,
                max_components,
                add_peak_valley=add_peak_valley,
            )
        else:
            raise ValueError("Branch method must be either 'pca', or 'chan_pca'.")
        n_random = (
            max(100, np.around(clust_clips.shape[0] / 100)) if use_rand_init else 0
        )
        clust_labels = (
            spikesorting_fullpursuit.clustering.kmeanspp.initial_cluster_farthest(
                scores, median_cluster_size, n_random=n_random
            )
        )
        #clust_labels = isosplit6(scores)
        # clust_labels = isosplit6(scores, initial_labels=clust_labels)

        clust_labels = spikesorting_fullpursuit.clustering.isocut.merge_clusters(
            scores,
            clust_labels,
            p_value_cut_thresh=p_value_cut_thresh,
            match_cluster_size=match_cluster_size,
            check_splits=check_splits,
        )

        new_labels = np.unique(clust_labels)
        if new_labels.size > 1:
            # Found at least one new cluster within original so reassign labels
            for nl in new_labels:
                temp_labels = neuron_labels_copy[curr_clust_bool]
                temp_labels[clust_labels == nl] = next_label
                neuron_labels_copy[curr_clust_bool] = temp_labels
                clusters_to_check.append(next_label)
                next_label += 1

    return neuron_labels_copy
