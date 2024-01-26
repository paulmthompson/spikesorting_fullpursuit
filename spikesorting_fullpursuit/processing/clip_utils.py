from os import path

import numpy as np

from spikesorting_fullpursuit.processing.conversions import time_window_to_samples
from spikesorting_fullpursuit.utils.memmap_close import MemMapClose


def minimal_redundancy_template_order(
    spikes,
    templates,
    max_templates=None,
    first_template_ind=None,
):
    """ """
    if max_templates is None:
        max_templates = templates.shape[0]

    templates_copy = np.copy(templates).T
    new_temp_order = np.zeros_like(templates_copy)
    temps_remaining = [int(x) for x in range(0, templates_copy.shape[1])]

    first_template_ind = first_template_ind / np.sum(first_template_ind)
    template_sizes = np.sum(templates_copy**2, axis=0) * first_template_ind

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
            dot_products[t_ind] = (
                np.sum(templates_copy[:, t] @ test_templates) / template_sizes[t]
            )
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
        this_vaf = total_function[df] / total_function[df - 1]
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
    max_templates=None,
):
    """ """
    if add_peak_valley and curr_chan_inds is None:
        raise ValueError(
            "Must supply indices for the main channel if using peak valley"
        )
    # Compute the weights using projection onto each neurons' template
    unique_labels, u_counts = np.unique(labels, return_counts=True)
    if max_templates is None:
        max_templates = unique_labels.size
    templates = np.empty((unique_labels.size, clips.shape[1]))
    for ind, l in enumerate(unique_labels):
        templates[ind, :] = np.mean(clips[labels == l, :], axis=0)
    templates = minimal_redundancy_template_order(
        clips, templates, max_templates=max_templates, first_template_ind=u_counts
    )
    # Keep at most the max_templates templates
    templates = templates[0 : min(templates.shape[0], max_templates), :]
    scores = clips @ templates.T

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


def keep_max_on_main(clips, main_chan_inds):
    keep_clips = np.ones(clips.shape[0], dtype="bool")
    for c in range(0, clips.shape[0]):
        max_ind = np.argmax(np.abs(clips[c, :]))
        if max_ind < main_chan_inds[0] or max_ind > main_chan_inds[-1]:
            keep_clips[c] = False

    return keep_clips


def cleanup_clusters(clips, neuron_labels):
    keep_clips = np.ones(clips.shape[0], dtype="bool")

    total_SSE_clips = np.sum(clips**2, axis=1)
    total_mean_SSE_clips = np.mean(total_SSE_clips)
    total_STD_clips = np.std(total_SSE_clips)
    overall_deviant = np.logical_or(
        total_SSE_clips > total_mean_SSE_clips + 5 * total_STD_clips,
        total_SSE_clips < total_mean_SSE_clips - 5 * total_STD_clips,
    )
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
            if np.logical_or(
                curr_SSE > nl_mean_SSE_clips + 2 * nl_STD_clips,
                curr_SSE < nl_mean_SSE_clips - 2 * nl_STD_clips,
            ):
                keep_clips[nl_ind] = False

    return keep_clips


def calculate_robust_template(clips):
    if clips.shape[0] == 1 or clips.ndim == 1:
        # Only 1 clip so nothing to average over
        return np.squeeze(clips)  # Return 1D array
    robust_template = np.zeros((clips.shape[1],), dtype=clips.dtype)
    sample_medians = np.median(clips, axis=0)
    for sample in range(0, clips.shape[1]):
        # Compute MAD with standard deviation conversion factor
        sample_MAD = (
            np.median(np.abs(clips[:, sample] - sample_medians[sample])) / 0.6745
        )
        # Samples within 1 MAD
        select_1MAD = np.logical_and(
            clips[:, sample] > sample_medians[sample] - sample_MAD,
            clips[:, sample] < sample_medians[sample] + sample_MAD,
        )
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
        dist_order = np.argpartition(
            nl_distances, min(n_keep, nl_distances.shape[0] - 1)
        )[0 : min(n_keep, nl_distances.shape[0])]
        select_dist = np.zeros(nl_distances.shape[0], dtype="bool")
        select_dist[dist_order] = True
        keep_clips[select_nl] = select_dist

    return keep_clips


def calculate_templates(clips, neuron_labels):
    """
    Computes the median template from clips for each unique
    label in neuron_labels.

    Output is a list of templates and a numpy array of the unique
    labels to which each template corresponds.
    """

    labels = np.unique(neuron_labels)
    # Use clips to get templates for each label in order
    templates = []
    for n in labels:
        templates.append(np.nanmean(clips[neuron_labels == n, :], axis=0))

    return templates, labels


def get_singlechannel_clips(
    probe_dict,
    chan_voltage,
    event_indices,
    clip_width_s,
    use_memmap=False,
) -> tuple[np.ndarray | MemMapClose, np.ndarray]:
    """

    Given a probe and the threshold crossings, return a matrix of clips for a
    given set of threshold crossings. We center the clip on the
    threshold crossing index. The width of the segment is passed
    to the function in units of seconds. This is done over all channels
    on probes, so threshold crossings input in event_indices must
    be a list where each element contains a numpy array of
    threshold crossing indices for the corresponding channel in the Probe
    object.
    event_indices that yield a clip width beyond data boundaries are ignored.
    event_indices MUST BE ORDERED or else edge cases will not correctly be
    accounted for and an error may result.
    'get_clips' below should be preferred as it is more versatile but
    this function is kept for ease and backward compatibility.

    Args:
        item_dict = {'sampling_rate',
                     'n_samples',
                     'thresholds': 1D numpy array of thresholds for each channel
                     'v_dtype',
                     'ID',
                     'memmap_dir',
                     'memmap_fID'
        chan_voltage: 2D numpy array of voltage values for segment
            (channels x samples)
        event_indices: 1D numpy array of indices of threshold crossings
        clip_width_s: list of two floats, time window in seconds to align
        use_memmap: bool, whether to use memmap for clips

    Returns:
        spike_clips: waveforms for each event in 2D array
            (events x window_samples)
            This can be either a numpy array or a memmap object
            The first dimension has length equal to the number of *VALID* indices,
            which is not necessarily the length of event_indices
        valid_event_indices: np.ndarray of booleans, which specify
            which of event indices are valid
    """

    sampling_rate = probe_dict["sampling_rate"]
    n_samples = probe_dict["n_samples"]

    window, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    clip_width_samples = window[1] - window[0]

    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones(event_indices.shape[0], dtype="bool")

    start_ind, stop_ind = check_edge_cases(
        event_indices, n_samples, valid_event_indices, window
    )

    if np.all(valid_event_indices == False):
        return None, valid_event_indices

    if use_memmap:
        clip_fname = path.join(
            probe_dict["memmap_dir"],
            "{0}clips_{1}.bin".format(probe_dict["memmap_fID"], str(probe_dict["ID"])),
        )
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict["v_dtype"],
            mode="w+",
            shape=(
                np.count_nonzero(valid_event_indices),
                clip_width_samples,
            ),
        )
    else:
        spike_clips = np.empty(
            (
                np.count_nonzero(valid_event_indices),
                clip_width_samples,
            ),
            dtype=probe_dict["v_dtype"],
        )

    for out_ind, spk in enumerate(
        range(start_ind, stop_ind + 1)
    ):  # Add 1 to index through last valid index
        spike_clips[out_ind, :] = chan_voltage[
            event_indices[spk] + window[0] : event_indices[spk] + window[1]
        ]

    if use_memmap:
        if isinstance(spike_clips, np.memmap):
            spike_clips.flush()
            spike_clips._mmap.close()
            del spike_clips
        # Make output read only
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict["v_dtype"],
            mode="r",
            shape=(
                np.count_nonzero(valid_event_indices),
                clip_width_samples,
            ),
        )

    return spike_clips, valid_event_indices


def check_edge_cases(
    event_indices,
    n_samples,
    valid_event_indices,
    window,
):
    start_ind = validate_first_indices(event_indices, 0, valid_event_indices, window)
    stop_ind = validate_last_indices(
        event_indices, n_samples, valid_event_indices, window
    )
    return start_ind, stop_ind


def validate_last_indices(
    event_indices,
    n_samples,
    valid_event_indices,
    window,
):
    stop_ind = event_indices.shape[0] - 1
    n = event_indices[stop_ind]
    while n + window[1] > n_samples:
        valid_event_indices[stop_ind] = False
        stop_ind -= 1
        if stop_ind < 0:
            # There are no valid indices
            valid_event_indices[:] = False
            break

        n = event_indices[stop_ind]
    return stop_ind


def validate_first_indices(
    event_indices,
    start_ind,
    valid_event_indices,
    window,
):
    n = event_indices[start_ind]
    while n + window[0] < 0:
        valid_event_indices[start_ind] = False
        start_ind += 1
        if start_ind == event_indices.size:
            # There are no valid indices
            valid_event_indices[:] = False
            break

        n = event_indices[start_ind]
    return start_ind


def get_clips(
    probe_dict,
    voltage,
    neighbors: np.ndarray,
    event_indices,
    clip_width_s: list,
    use_memmap=False,
    check_valid=True,
) -> tuple[np.ndarray | MemMapClose, np.ndarray]:
    """
    This is like get_singlechannel_clips except it concatenates the clips for
    each channel input in the list 'neighbors' in the order that they appear.
    Also works for single channel clips.

    Event indices is a single one dimensional array of indices over which
    clips from all input
    channels given by neighbors will be aligned. event_indices MUST BE ORDERED
    or else edge cases will not correctly be accounted for and an error may
    result. If no edge cases are present, check_valid can be set False and
    edges will not be checked (an indexing error will occur if this assumption
    is incorrect).

    Spike clips are output in the order of the input event indices.

    Args:
        item_dict = {'sampling_rate',
                     'n_samples',
                     'thresholds': 1D numpy array of thresholds for each channel
                     'v_dtype',
                     'ID',
                     'memmap_dir',
                     'memmap_fID'
        voltage: 2D numpy array of voltage values for segment
            (channels x samples)
        neighbors: np.ndarray of channel indices to use for clips
        event_indices: 1D numpy array of indices of threshold crossings
        clip_width_s: list of two floats, time window in seconds to align
        use_memmap:
        check_valid:

    Returns:
        spike_clips: waveforms for each event in 2D array
            (events x window_samples)
            This can be either a numpy array or a memmap object
            The first dimension has length equal to the number of *VALID* indices,
            which is not necessarily the length of event_indices
        valid_event_indices: np.ndarray of booleans, which specify
            which of event indices are valid
    """
    if event_indices.ndim > 1:
        raise ValueError("Event_indices must be one dimensional array of indices")

    sampling_rate = probe_dict["sampling_rate"]
    n_samples = probe_dict["n_samples"]

    window, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    clip_width_samples = (window[1] - window[0]) * len(neighbors)

    if len(event_indices) == 0:
        # No indices input
        return np.zeros((0, clip_width_samples), dtype=probe_dict["v_dtype"]), np.ones(
            0, dtype="bool"
        )

    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices: np.ndarray = np.ones(event_indices.shape[0], dtype="bool")
    start_ind = 0
    if check_valid:
        start_ind, stop_ind = check_edge_cases(
            event_indices, n_samples, valid_event_indices, window
        )

        if np.all(valid_event_indices == False):
            return None, valid_event_indices

    else:
        stop_ind = len(event_indices) - 1

    if use_memmap:
        clip_fname = path.join(
            probe_dict["memmap_dir"],
            "{0}clips_{1}.bin".format(probe_dict["memmap_fID"], str(probe_dict["ID"])),
        )
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict["v_dtype"],
            mode="w+",
            shape=(
                np.count_nonzero(valid_event_indices),
                clip_width_samples,
            ),
        )
    else:
        spike_clips: np.ndarray = np.empty(
            (
                np.count_nonzero(valid_event_indices),
                clip_width_samples,
            ),
            dtype=probe_dict["v_dtype"],
        )

    for out_ind, spk in enumerate(
        range(start_ind, stop_ind + 1)
    ):  # Add 1 to index through last valid index
        start = 0
        for n_ind, chan in enumerate(neighbors):
            stop = (n_ind + 1) * (window[1] - window[0])
            spike_clips[out_ind, start:stop] = voltage[
                chan, event_indices[spk] + window[0] : event_indices[spk] + window[1]
            ]
            start = stop

    if use_memmap:
        if isinstance(spike_clips, np.memmap):
            spike_clips.flush()
            spike_clips._mmap.close()
            del spike_clips
        # Make output read only
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict["v_dtype"],
            mode="r",
            shape=(
                np.count_nonzero(valid_event_indices),
                clip_width_samples,
            ),
        )

    return spike_clips, valid_event_indices


def get_windows_and_indices(
    clip_width_s,
    sampling_rate,
    channel,
    neighbors,
):
    """
    Computes some basic info used in many functions about how clips are
    are formatted and provides window indices and clip indices.
    """

    curr_chan_win, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    chan_neighbor_ind = next(
        (idx[0] for idx, val in np.ndenumerate(neighbors) if val == channel), None
    )
    samples_per_chan = curr_chan_win[1] - curr_chan_win[0]
    curr_chan_inds = np.arange(
        samples_per_chan * chan_neighbor_ind,
        samples_per_chan * chan_neighbor_ind + samples_per_chan,
        1,
    )

    return (
        clip_width_s,
        chan_neighbor_ind,
        curr_chan_win,
        samples_per_chan,
        curr_chan_inds,
    )
