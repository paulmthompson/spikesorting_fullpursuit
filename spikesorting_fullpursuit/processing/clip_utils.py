import numpy as np


def minimal_redundancy_template_order(
        spikes,
        templates,
        max_templates=None,
        first_template_ind=None):
    """
    """
    if max_templates is None:
        max_templates = templates.shape[0]

    templates_copy = np.copy(templates).T
    new_temp_order = np.zeros_like(templates_copy)
    temps_remaining = [int(x) for x in range(0, templates_copy.shape[1])]

    first_template_ind = first_template_ind / np.sum(first_template_ind)
    template_sizes = np.sum(templates_copy ** 2, axis=0) * first_template_ind

    # if first_template_ind is None:
    #     first_template_ind = np.argmax(template_sizes)
    # new_temp_order[:, 0] = templates_copy[:, first_template_ind]
    # temps_remaining.remove(first_template_ind)
    # total_function = [template_sizes[first_template_ind]]

    f_ind = np.argmax(template_sizes)
    new_temp_order[:, 0] = templates_copy[:, f_ind]
    temps_remaining.remove(f_ind)
    total_function = [template_sizes[f_ind]]

    temp_seeking = 1
    while len(temps_remaining) > 0:
        test_templates = new_temp_order[:, 0:temp_seeking]
        dot_products = np.zeros(len(temps_remaining))
        for t_ind, t in enumerate(temps_remaining):
            # dot_products[t_ind] = np.abs(template_sizes[t] - np.median(templates_copy[:, t] @ test_templates))
            # dot_products[t_ind] = np.amin(np.abs(template_sizes[t] - templates_copy[:, t] @ test_templates))
            dot_products[t_ind] = np.sum(templates_copy[:, t] @ test_templates) / template_sizes[t]
        # next_best_temp = temps_remaining[np.argmax(dot_products)]
        # total_function.append((dot_products[np.argmax(dot_products)]))
        next_best_temp = temps_remaining[np.argmin(dot_products)]
        total_function.append((np.abs(dot_products[np.argmin(dot_products)])))
        new_temp_order[:, temp_seeking] = templates_copy[:, next_best_temp]
        temps_remaining.remove(next_best_temp)

        temp_seeking += 1

    total_function = np.hstack(total_function)
    vaf = np.zeros_like(total_function)
    total_function[0] = np.inf
    for df in range(1, total_function.size):
        this_vaf = total_function[df] / total_function[df-1]
        vaf[df] = this_vaf
        # if vaf[df] < vaf[df-1]:
        #     break
    df = np.argmax(vaf) + 1
    if df < 2:
        df = 2
    if df > max_templates:
        df = max_templates
    new_temp_order = new_temp_order[:, 0:df]
    print("CHOSE", df, "TEMPLATES FOR PROJECTION")

    return new_temp_order.T


def compute_template_projection(
        clips,
        labels,
        curr_chan_inds,
        add_peak_valley=False,
        max_templates=None):
    """
    """
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError("Must supply indices for the main channel if using peak valley")
    # Compute the weights using projection onto each neurons' template
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if max_templates is None:
        max_templates = unique_labels.size
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.mean(clips[labels == l, :], axis=0)
    templates = minimal_redundancy_template_order(
        clips,
        templates,
        max_templates=max_templates,
        first_template_ind=u_counts
        )
    # Keep at most the max_templates templates
    templates = templates[0:min(templates.shape[0], max_templates), :]
    scores = clips @ templates.T

    if add_peak_valley:
        peak_valley = (np.amax(clips[:, curr_chan_inds], axis=1)
                       - np.amin(clips[:, curr_chan_inds], axis=1)
                       ).reshape(clips.shape[0], -1)
        peak_valley /= np.amax(np.abs(peak_valley))  # Normalized from -1 to 1
        peak_valley *= np.amax(np.amax(np.abs(scores)))  # Normalized to same range as PC scores
        scores = np.hstack((scores, peak_valley))

    return scores


def keep_max_on_main(clips, main_chan_inds):

    keep_clips = np.ones(clips.shape[0], dtype="bool")
    for c in range(0, clips.shape[0]):
        max_ind = np.argmax(np.abs(clips[c, :]))
        if max_ind < main_chan_inds[0] or max_ind > main_chan_inds[-1]:
            keep_clips[c] = False

    return keep_clips


def cleanup_clusters(clips, neuron_labels):

    keep_clips = np.ones(clips.shape[0], dtype="bool")

    total_SSE_clips = np.sum(clips ** 2, axis=1)
    total_mean_SSE_clips = np.mean(total_SSE_clips)
    total_STD_clips = np.std(total_SSE_clips)
    overall_deviant = np.logical_or(total_SSE_clips > total_mean_SSE_clips + 5*total_STD_clips,
                        total_SSE_clips < total_mean_SSE_clips - 5*total_STD_clips)
    keep_clips[overall_deviant] = False

    for nl in np.unique(neuron_labels):
        select_nl = neuron_labels == nl
        select_nl[overall_deviant] = False
        nl_template = np.mean(clips[select_nl, :], axis=0)
        nl_SSE_clips = np.sum((clips[select_nl, :] - nl_template) ** 2, axis=1)
        nl_mean_SSE_clips = np.mean(nl_SSE_clips)
        nl_STD_clips = np.std(nl_SSE_clips)
        for nl_ind in range(0, clips.shape[0]):
            if not select_nl[nl_ind]:
                continue
            curr_SSE = np.sum((clips[nl_ind, :] - nl_template) ** 2)
            if np.logical_or(curr_SSE > nl_mean_SSE_clips + 2*nl_STD_clips,
                             curr_SSE < nl_mean_SSE_clips - 2*nl_STD_clips):
                keep_clips[nl_ind] = False

    return keep_clips


def calculate_robust_template(clips):

    if clips.shape[0] == 1 or clips.ndim == 1:
        # Only 1 clip so nothing to average over
        return np.squeeze(clips) # Return 1D array
    robust_template = np.zeros((clips.shape[1], ), dtype=clips.dtype)
    sample_medians = np.median(clips, axis=0)
    for sample in range(0, clips.shape[1]):
        # Compute MAD with standard deviation conversion factor
        sample_MAD = np.median(np.abs(clips[:, sample] - sample_medians[sample])) / 0.6745
        # Samples within 1 MAD
        select_1MAD = np.logical_and(clips[:, sample] > sample_medians[sample] - sample_MAD,
                                     clips[:, sample] < sample_medians[sample] + sample_MAD)
        if ~np.any(select_1MAD):
            # Nothing within 1 MAD STD so just fall back on median
            robust_template[sample] = sample_medians[sample]
        else:
            # Robust template as median of samples within 1 MAD
            robust_template[sample] = np.median(clips[select_1MAD, sample])

    return robust_template


def keep_cluster_centroid(clips, neuron_labels, n_keep=100):
    keep_clips = np.ones(clips.shape[0], dtype="bool")
    if n_keep > clips.shape[0]:
        # Everything will be kept no matter what so just exit
        return keep_clips
    for nl in np.unique(neuron_labels):
        select_nl = neuron_labels == nl
        nl_template = np.mean(clips[select_nl, :], axis=0)
        nl_distances = np.sum((clips[select_nl, :] - nl_template[None, :]) ** 2, axis=1)
        dist_order = np.argpartition(nl_distances, min(n_keep, nl_distances.shape[0]-1))[0:min(n_keep, nl_distances.shape[0])]
        select_dist = np.zeros(nl_distances.shape[0], dtype="bool")
        select_dist[dist_order] = True
        keep_clips[select_nl] = select_dist

    return keep_clips
