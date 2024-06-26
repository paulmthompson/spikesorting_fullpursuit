import numpy as np
from copy import copy, deepcopy
from scipy.stats import norm
from spikesorting_fullpursuit.overlap import (
    neuron_separability,
    binary_pursuit_parallel,
)
from spikesorting_fullpursuit.overlap.consolidate import SegSummary
from spikesorting_fullpursuit.processing.clip_utils import (
    calculate_robust_template,
    get_clips,
)
from spikesorting_fullpursuit.parallel.segment_parallel import memmap_to_mem
from spikesorting_fullpursuit.processing.conversions import time_window_to_samples
from spikesorting_fullpursuit.c_cython import sort_cython
from spikesorting_fullpursuit.utils.memmap_close import MemMapClose


def get_binary_pursuit_clip_width(
    seg_w_items,
    clips_dict,
    voltage,
    data_dict,
    sort_info,
):
    """
    Determines a clip width to use for binary pursuit by asking how much
    the current clip width must be increased so that it returns near a median
    voltage of 0 at both ends of the clip width. This is constrained by the
    sorting parameter max_binary_pursuit_clip_width_factor.
    """

    n_samples_per_chan = sort_info["n_samples_per_chan"]
    n_channels = sort_info["n_channels"]
    clip_width_original = sort_info["clip_width"]
    sampling_rate = sort_info["sampling_rate"]
    max_binary_pursuit_clip_width_factor = sort_info[
        "max_binary_pursuit_clip_width_factor"
    ]

    # Maximim factor by which the clip width can be increased
    if max_binary_pursuit_clip_width_factor <= 1.0:
        # Do not use expanded clip widths, just return
        (
            bp_clip_width,
            original_clip_starts,
            original_clip_stops,
        ) = get_original_clip_width_start_stop(
            n_samples_per_chan,
            n_channels,
            clip_width_original,
        )
        return bp_clip_width, original_clip_starts, original_clip_stops

    # Start by building the set of all clips for all units in segment
    all_events = []
    for w_item in seg_w_items:
        if w_item["ID"] in data_dict["results_dict"].keys():
            if len(data_dict["results_dict"][w_item["ID"]][0]) == 0:
                # This work item found nothing (or raised an exception)
                continue
            all_events.append(data_dict["results_dict"][w_item["ID"]][0])
    if len(all_events) == 0:
        # No events found so just return input clip width
        (
            bp_clip_width,
            original_clip_starts,
            original_clip_stops,
        ) = get_original_clip_width_start_stop(
            n_samples_per_chan,
            n_channels,
            clip_width_original,
        )
        return bp_clip_width, original_clip_starts, original_clip_stops

    all_events = np.hstack(all_events)
    all_events.sort()  # Must be sorted for get multichannel clips to work
    # Find the average clip for our max output clip width, double the original
    bp_clip_width = [
        max_binary_pursuit_clip_width_factor * v for v in clip_width_original
    ]
    all_clips, valid_event_indices = get_clips(
        clips_dict,
        voltage,
        np.arange(0, voltage.shape[0]),
        all_events,
        clip_width_s=bp_clip_width,
    )

    if np.count_nonzero(valid_event_indices) == 0:
        (
            bp_clip_width,
            original_clip_starts,
            original_clip_stops,
        ) = get_original_clip_width_start_stop(
            n_samples_per_chan,
            n_channels,
            clip_width_original,
        )
        return bp_clip_width, original_clip_starts, original_clip_stops

    mean_clip = np.mean(all_clips, axis=0)
    bp_samples_per_chan = all_clips.shape[1] // n_channels
    first_indices = np.arange(
        0,
        bp_samples_per_chan * (n_channels - 1) + 1,
        bp_samples_per_chan,
        dtype=np.int64,
    )
    last_indices = np.arange(
        bp_samples_per_chan - 1,
        bp_samples_per_chan * n_channels + 1,
        bp_samples_per_chan,
        dtype=np.int64,
    )

    # Randomly choose 10 seconds worth of time points

    noise_sample_inds = np.random.choice(voltage.shape[1], 10 * sampling_rate)
    median_noise = np.median(np.median(np.abs(voltage[:, noise_sample_inds]), axis=1))
    clip_end_tolerance = 0.05 * median_noise

    bp_chan_win_samples, _ = time_window_to_samples(bp_clip_width, sampling_rate)
    chan_win_samples, _ = time_window_to_samples(clip_width_original, sampling_rate)

    # Find the most we can increase the first indices to
    # chan_win_samples[0] is negative, we want positve here
    max_pre_samples = (
        -1 * bp_chan_win_samples[0] + chan_win_samples[0]
    )  # Don't shrink past original
    while np.all(np.abs(mean_clip[first_indices]) < clip_end_tolerance):
        if first_indices[0] >= max_pre_samples:
            break
        first_indices += 1

    # This is what's left of bp_chan_win_samples after we moved
    bp_clip_width[0] = (
        -1 * (-1 * bp_chan_win_samples[0] - first_indices[0]) / sampling_rate
    )

    # Most we can decrease the last indices to
    min_post_samples = (
        (bp_samples_per_chan - bp_chan_win_samples[1]) + chan_win_samples[1] - 1
    )  # Don't shrink past original
    while np.all(np.abs(mean_clip[last_indices]) < clip_end_tolerance):
        if last_indices[0] <= min_post_samples:
            break
        last_indices -= 1
    bp_clip_width[1] = (
        bp_chan_win_samples[1] - (bp_samples_per_chan - last_indices[0])
    ) / sampling_rate

    # Compute the indices required to slice the new bp_clip_width clips back to
    # their original input sort_info['clip_width'] size
    clip_start_ind = (
        -1 * bp_chan_win_samples[0] + chan_win_samples[0]
    ) - first_indices[0]
    clip_stop_ind = clip_start_ind + (chan_win_samples[1] - chan_win_samples[0])
    clip_n = last_indices[0] - first_indices[0]  # New expanded clip width
    original_clip_starts = np.arange(
        clip_start_ind,
        clip_n * (n_channels),
        clip_n,
        dtype=np.int64,
    )
    original_clip_stops = np.arange(
        clip_stop_ind,
        (clip_n + 1) * n_channels,
        clip_n,
        dtype=np.int64,
    )
    print(1, bp_chan_win_samples)
    print(2, chan_win_samples)
    print(3, max_pre_samples)
    print(4, first_indices)
    print(5, bp_clip_width)
    print(6, min_post_samples)
    print(7, last_indices)
    print(8, bp_samples_per_chan)
    print(9, clip_start_ind)
    print(10, clip_stop_ind)
    print(11, clip_n)
    print(12, original_clip_starts)
    print(13, original_clip_stops)
    # Make clip_width symmetric
    bp_cw_max = max(np.abs(bp_clip_width))
    bp_clip_width = [-1 * bp_cw_max, bp_cw_max]

    return bp_clip_width, original_clip_starts, original_clip_stops


def get_original_clip_width_start_stop(
    n_samples_per_chan,
    n_channels,
    clip_width,
):
    """

    Parameters
    ----------
    n_samples_per_chan: int
    n_channels: int
    clip_width: float

    Returns
    -------

    """
    original_clip_starts = np.arange(
        0,
        n_samples_per_chan * (n_channels),
        n_samples_per_chan,
        dtype=np.int64,
    )
    original_clip_stops = np.arange(
        n_samples_per_chan,
        (n_samples_per_chan + 1) * n_channels,
        n_samples_per_chan,
        dtype=np.int64,
    )

    bp_cw_max = max(np.abs(clip_width))
    bp_clip_width = [-1 * bp_cw_max, bp_cw_max]
    return bp_clip_width, original_clip_starts, original_clip_stops


def full_binary_pursuit(
    seg_w_items,
    data_dict,
    seg_number,
    sort_info,
    v_dtype,
    overlap_ratio_threshold,
    absolute_refractory_period,
    kernels_path=None,
    max_gpu_memory=None,
    use_memmap=False,
    output_clips=False,
):
    """
    This is the main function that runs binary pursuit. It first handles
    the unit and template consolidation and the removal of noise templates to
    create the final template set for binary pursuit derived from the input
    segment sorted data. Then the templates are input to binary pursuit. Output
    is finally formatted for final output.
    """
    # Get numpy view of voltage for clips and binary pursuit
    voltage = get_voltage_for_bp(
        data_dict,
        seg_number,
        use_memmap,
        v_dtype,
    )

    all_chan_nbrs = np.arange(0, voltage.shape[0], dtype=np.int64)

    # Reset neighbors to all channels for full binary pursuit
    original_neighbors = []
    for w_item in seg_w_items:
        if w_item["ID"] in data_dict["results_dict"].keys():
            original_neighbors.append(w_item["neighbors"])
            w_item["neighbors"] = np.copy(all_chan_nbrs)

    original_clip_width = [s for s in sort_info["clip_width"]]
    original_n_samples_per_chan = copy(sort_info["n_samples_per_chan"])

    # Max shift indices to check for binary pursuit overlaps
    n_max_shift_inds = original_n_samples_per_chan - 1

    # Make a dictionary with all info needed for get_clips
    clips_dict = {
        "sampling_rate": sort_info["sampling_rate"],
        "n_samples": seg_w_items[0]["n_samples"],
        "v_dtype": v_dtype,
    }

    # Need to build this in format used for consolidate functions
    seg_summary = create_segment_summary(
        absolute_refractory_period,
        clips_dict,
        data_dict,
        seg_w_items,
        sort_info,
        v_dtype,
        voltage,
    )

    if len(seg_summary.summaries) == 0:
        print("Found no neuron templates for binary pursuit")
        return [[[], [], [], [], None]]

    if sort_info["verbose"]:
        print(
            "Entered with",
            len(seg_summary.summaries),
            "templates in segment",
            seg_number + 1,
        )

    # Need this chan_win before assigning binary pursuit clip width. Used for
    # find_overlap_templates
    chan_win, clip_width = time_window_to_samples(
        sort_info["clip_width"], sort_info["sampling_rate"]
    )

    # Reassign binary pursuit clip width to clip width
    # (This is slightly confusing but keeps certain code compatability.
    # We will reset it to original value at the end.)
    (
        bp_reduction_samples_per_chan,
        original_clip_starts,
        original_clip_stops,
    ) = assign_bp_clip_width(
        clips_dict,
        data_dict,
        original_clip_width,
        original_n_samples_per_chan,
        seg_w_items,
        sort_info,
        voltage,
    )

    # Gather the bp_templates for each unit and the clip-template residuals for
    # computing the separability metrics for each unit
    bp_templates = []
    clip_template_residuals = []
    n_template_spikes = []
    for neuron_summary in seg_summary.summaries:
        clips, _ = get_clips(
            clips_dict,
            voltage,
            all_chan_nbrs,
            neuron_summary["spike_indices"],
            clip_width_s=sort_info["clip_width"],  # binary_pursuit_clip_width
        )
        robust_template = calculate_robust_template(clips)
        bp_templates.append(robust_template)
        clip_template_residuals.append(clips - robust_template)
        n_template_spikes.append(neuron_summary["spike_indices"].shape[0])

    bp_templates = np.vstack(bp_templates)  # N_units X template_width

    chan_covariance_mats = get_channel_covariance_matrix(
        clip_template_residuals,
        sort_info["n_channels"],
        sort_info["n_samples_per_chan"],  # bp_n_samples_per_chan
        sort_info["n_cov_samples"],
        sort_info["verbose"],
    )

    n_template_spikes = np.array(n_template_spikes, dtype=np.int64)
    # The overlap check input here is hard coded to look at shifts +/- the
    # original input chan win (clip_width). This is arbitrary
    templates_to_check = sort_cython.find_overlap_templates(
        bp_templates,
        sort_info["n_samples_per_chan"],  # bp_n_samples_per_chan
        sort_info["n_channels"],
        np.int64(np.abs(chan_win[0]) - 1),
        np.int64(np.abs(chan_win[1]) - 1),
        n_template_spikes,
    )

    # Go through suspect bp_templates in templates_to_check
    templates_to_delete = np.zeros(bp_templates.shape[0], dtype="bool")
    # Use the sigma lower bound to decide the acceptable level of
    # misclassification between template sums
    confusion_threshold = norm.sf(sort_info["sigma"])
    for t_info in templates_to_check:
        # templates_to_check is not length of bp_templates so need to find the
        # correct index of the template being checked
        t_ind = t_info[0]
        shift_temp = t_info[1]
        p_confusion = neuron_separability.check_template_pair(
            bp_templates[t_ind, :],
            shift_temp,
            chan_covariance_mats,
            sort_info["n_channels"],
            sort_info["n_samples_per_chan"],  # bp_n_samples_per_chan
        )
        if p_confusion > confusion_threshold:
            templates_to_delete[t_ind] = True

    # Remove units corresponding to overlap bp_templates from summary
    bp_templates = bp_templates[~templates_to_delete, :]
    for x in reversed(range(0, len(seg_summary.summaries))):
        if templates_to_delete[x]:
            del seg_summary.summaries[x]

    if sort_info["verbose"]:
        print(
            "Removing sums reduced number of templates to", len(seg_summary.summaries)
        )

    # Return the original neighbors to the work items that were reset
    orig_neigh_ind = 0
    for w_item in seg_w_items:
        if w_item["ID"] in data_dict["results_dict"].keys():
            w_item["neighbors"] = original_neighbors[orig_neigh_ind]
            orig_neigh_ind += 1

    if len(seg_summary.summaries) == 0:
        # All data this segment found nothing (or raised an exception)
        seg_data = []
        for chan in range(0, sort_info["n_channels"]):
            curr_item = None
            for w_item in seg_w_items:
                if w_item["channel"] == chan:
                    curr_item = w_item
                    break
            if curr_item is None:
                # This should never be possible, but just to be sure
                raise RuntimeError("Could not find a matching work item for unit")
            seg_data.append([[], [], [], [], curr_item["ID"]])
        # Set these back to match input values
        sort_info["clip_width"] = original_clip_width
        sort_info["n_samples_per_chan"] = bp_reduction_samples_per_chan
        return seg_data

    # Perform final sharpening, this time including channel covariance matrices
    # and the full binary pursuit templates so that confusion between binary
    # pursuit templates is accounted for
    seg_summary.set_bp_templates(bp_templates)
    seg_summary.sharpen_across_chans(chan_covariance_mats)
    bp_templates = [
        neuron_summary["bp_template"] for neuron_summary in seg_summary.summaries
    ]
    if sort_info["verbose"]:
        print(
            "Removing confused pairs reduced number of templates to", len(bp_templates)
        )

    del seg_summary  # No longer needed so clear memory

    separability_metrics = neuron_separability.compute_separability_metrics(
        bp_templates,
        chan_covariance_mats,
        sort_info,
    )
    # Identify templates similar to noise and decide what to do with them
    noisy_templates = neuron_separability.find_noisy_templates(
        separability_metrics, sort_info
    )
    separability_metrics = neuron_separability.set_bp_threshold(separability_metrics)
    separability_metrics, noisy_templates = neuron_separability.check_noise_templates(
        separability_metrics,
        sort_info,
        noisy_templates,
    )
    separability_metrics = neuron_separability.delete_noise_units(
        separability_metrics, noisy_templates
    )

    if separability_metrics["templates"].shape[0] == 0:
        # All data this segment found nothing (or raised an exception)
        seg_data = []
        for chan in range(0, sort_info["n_channels"]):
            curr_item = None
            for w_item in seg_w_items:
                if w_item["channel"] == chan:
                    curr_item = w_item
                    break
            if curr_item is None:
                # This should never be possible, but just to be sure
                raise RuntimeError("Could not find a matching work item for unit")
            seg_data.append([[], [], [], [], curr_item["ID"]])
        # Set these back to match input values
        sort_info["clip_width"] = original_clip_width
        sort_info["n_samples_per_chan"] = bp_reduction_samples_per_chan
        return seg_data

    if sort_info["verbose"]:
        print(
            "Starting full binary pursuit search with",
            separability_metrics["templates"].shape[0],
            "templates in segment",
            seg_number + 1,
        )

    (
        crossings,
        neuron_labels,
        is_binary_pursuit_spike,
        clips,
    ) = binary_pursuit_parallel.binary_pursuit(
        voltage,
        v_dtype,
        sort_info,
        separability_metrics,
        n_max_shift_inds=n_max_shift_inds,
        kernels_path=None,
        max_gpu_memory=max_gpu_memory,
    )

    # Returns spike clips after the waveforms of any potentially overlapping spikes have been removed.
    if not sort_info["get_adjusted_clips"]:
        clips, _ = get_clips(
            clips_dict,
            voltage,
            all_chan_nbrs,
            crossings,
            clip_width_s=sort_info["clip_width"],  # binary_pursuit_clip_width
        )

    if sort_info["output_separability_metrics"]:
        # Save the separability metrics as used (and output) by binary_pursuit
        sort_info["separability_metrics"][seg_number] = separability_metrics

    chans_to_template_labels = assign_unit_channel_to_max_snr(
        clips,
        neuron_labels,
        seg_w_items[0]["thresholds"],
        sort_info["sigma"],
        sort_info["n_samples_per_chan"],  # bp_n_samples_per_chan
        sort_info["n_channels"],
    )

    # Set these back to match input values
    sort_info["clip_width"] = original_clip_width
    sort_info["n_samples_per_chan"] = bp_reduction_samples_per_chan

    # Need to convert binary pursuit output to standard sorting output. This
    # requires data from every channel, even if it is just empty
    seg_data = convert_binary_pursuit_output_to_seg_data(
        chans_to_template_labels,
        clips,
        crossings,
        is_binary_pursuit_spike,
        neuron_labels,
        original_clip_starts,
        original_clip_stops,
        output_clips,
        seg_w_items,
        sort_info,
        v_dtype,
    )

    return seg_data


def assign_unit_channel_to_max_snr(
    clips,
    neuron_labels,
    thresholds,
    sigma,
    n_samples_per_chan,
    n_channels,
):
    """

    Parameters
    ----------
    clips:
    neuron_labels:
    thresholds:
    sigma: float
    n_samples_per_chan: int
    n_channels: int

    Returns
    -------

    """

    chans_to_template_labels = {}
    for chan in range(0, n_channels):
        chans_to_template_labels[chan] = []

    for unit in np.unique(neuron_labels):
        # Find this unit's channel as the channel with max SNR of template
        curr_template = np.mean(clips[neuron_labels == unit, :], axis=0)
        unit_best_snr = -1.0
        unit_best_chan = None
        for chan in range(0, n_channels):
            background_noise_std = thresholds[chan] / sigma
            chan_win = [
                n_samples_per_chan * chan,
                n_samples_per_chan * (chan + 1),
            ]
            chan_template = curr_template[chan_win[0] : chan_win[1]]
            temp_range = np.amax(chan_template) - np.amin(chan_template)
            chan_snr = temp_range / (3 * background_noise_std)
            if chan_snr > unit_best_snr:
                unit_best_snr = chan_snr
                unit_best_chan = chan
        chans_to_template_labels[unit_best_chan].append(unit)
    return chans_to_template_labels


def convert_binary_pursuit_output_to_seg_data(
    chans_to_template_labels,
    clips,
    crossings,
    is_binary_pursuit_spike,
    neuron_labels,
    original_clip_starts,
    original_clip_stops,
    output_clips,
    seg_w_items,
    sort_info,
    v_dtype,
):
    seg_data = []
    for chan in range(0, sort_info["n_channels"]):
        curr_item = None
        for w_item in seg_w_items:
            if w_item["channel"] == chan:
                curr_item = w_item
                break

        if curr_item is None:
            # This should never be possible, but just to be sure
            raise RuntimeError("Could not find a matching work item for unit")

        if len(chans_to_template_labels[chan]) > 0:
            # Set data to empty defaults and append if they exist
            (
                chan_events,
                chan_labels,
                chan_is_binary_pursuit_spike,
                chan_clips,
            ) = (
                [],
                [],
                [],
                [],
            )
            for unit in chans_to_template_labels[chan]:
                is_current_unit = neuron_labels == unit
                chan_events.append(crossings[is_current_unit])
                chan_labels.append(neuron_labels[is_current_unit])
                chan_is_binary_pursuit_spike.append(
                    is_binary_pursuit_spike[is_current_unit]
                )

                if output_clips:
                    # Get clips for this unit over all channels
                    unit_clips = get_unit_clips(
                        clips,
                        curr_item,
                        is_current_unit,
                        original_clip_starts,
                        original_clip_stops,
                        sort_info,
                        v_dtype,
                    )
                    chan_clips.append(unit_clips)
                else:
                    chan_clips.append([])

            # Adjust crossings for seg start time
            chan_events = np.hstack(chan_events)
            chan_events += curr_item["index_window"][0]
            # Append list of crossings, labels, clips, binary pursuit spikes
            seg_data.append(
                [
                    chan_events,
                    np.hstack(chan_labels),
                    np.vstack(chan_clips),
                    np.hstack(chan_is_binary_pursuit_spike),
                    curr_item["ID"],
                ]
            )
        else:
            # This work item found nothing (or raised an exception)
            seg_data.append([[], [], [], [], curr_item["ID"]])
    return seg_data


def get_unit_clips(
    clips,
    curr_item,
    is_current_unit,
    original_clip_starts,
    original_clip_stops,
    sort_info,
    v_dtype,
):
    unit_clips = np.zeros(
        (
            np.count_nonzero(is_current_unit),
            curr_item["neighbors"].shape[0] * sort_info["n_samples_per_chan"],
        ),
        dtype=v_dtype,
    )
    # Map clips from all channels to current channel neighborhood
    for neigh in range(0, curr_item["neighbors"].shape[0]):
        chan_ind = curr_item["neighbors"][neigh]
        unit_clips[
            :,
            neigh
            * sort_info["n_samples_per_chan"] : (neigh + 1)
            * sort_info["n_samples_per_chan"],
        ] = clips[
            is_current_unit,
            original_clip_starts[chan_ind] : original_clip_stops[chan_ind],
        ]
    return unit_clips


def create_segment_summary(
    absolute_refractory_period,
    clips_dict,
    data_dict,
    seg_w_items,
    sort_info,
    v_dtype,
    voltage,
):
    seg_data = []
    for w_item in seg_w_items:
        if w_item["ID"] in data_dict["results_dict"].keys():
            if len(data_dict["results_dict"][w_item["ID"]][0]) == 0:
                # This work item found nothing (or raised an exception)
                seg_data.append([[], [], [], [], w_item["ID"]])
                continue

            clips, _ = get_clips(
                clips_dict,
                voltage,
                w_item["neighbors"],
                data_dict["results_dict"][w_item["ID"]][0],  # crossings
                clip_width_s=sort_info["clip_width"],
            )

            # Insert list of crossings, labels, clips, binary pursuit spikes
            seg_data.append(
                [
                    data_dict["results_dict"][w_item["ID"]][0],  # crossings
                    data_dict["results_dict"][w_item["ID"]][1],  # labels
                    clips,
                    np.zeros(
                        len(data_dict["results_dict"][w_item["ID"]][0]), dtype="bool"
                    ),
                    w_item["ID"],
                ]
            )
            if type(seg_data[-1][0][0]) == np.ndarray:
                if seg_data[-1][0][0].size > 0:
                    # Adjust crossings for segment start time
                    seg_data[-1][0][0] += w_item["index_window"][0]
        else:
            # This work item found nothing (or raised an exception)
            seg_data.append([[], [], [], [], w_item["ID"]])
    # Pass a copy of current state of sort info to seg_summary. Actual sort_info
    # will be altered later but SegSummary must follow original data
    seg_summary = SegSummary(
        seg_data,
        seg_w_items,
        deepcopy(sort_info),
        v_dtype,
        absolute_refractory_period=absolute_refractory_period,
        verbose=False,
    )
    return seg_summary


def assign_bp_clip_width(
    clips_dict,
    data_dict,
    original_clip_width,
    original_n_samples_per_chan,
    seg_w_items,
    sort_info,
    voltage,
):
    (
        bp_clip_width,
        original_clip_starts,
        original_clip_stops,
    ) = get_binary_pursuit_clip_width(
        seg_w_items,
        clips_dict,
        voltage,
        data_dict,
        sort_info,
    )

    sort_info["clip_width"] = bp_clip_width

    # Store newly assigned binary pursuit clip width for final output
    if "binary_pursuit_clip_width" not in sort_info:
        sort_info["binary_pursuit_clip_width"] = [0, 0]

    sort_info["binary_pursuit_clip_width"][0] = min(
        bp_clip_width[0], sort_info["binary_pursuit_clip_width"][0]
    )
    sort_info["binary_pursuit_clip_width"][1] = max(
        bp_clip_width[1], sort_info["binary_pursuit_clip_width"][1]
    )
    bp_chan_win, _ = time_window_to_samples(bp_clip_width, sort_info["sampling_rate"])

    sort_info["n_samples_per_chan"] = bp_chan_win[1] - bp_chan_win[0]

    # This should be same as input samples per chan but could probably
    # be off by one due to rounding error of the clip width so
    # need to recompute
    bp_reduction_samples_per_chan = original_clip_stops[0] - original_clip_starts[0]
    if bp_reduction_samples_per_chan != original_n_samples_per_chan:
        # This should be coded so this never happens, but if it does it could be a difficult to notice disaster during consolidate
        raise RuntimeError(
            "Template reduction from binary pursuit does not have the same number of samples as original!"
        )

    if sort_info["verbose"]:
        print(
            "Binary pursuit clip width is",
            bp_clip_width,
            "from",
            original_clip_width,
        )
    if sort_info["verbose"]:
        print(
            "Binary pursuit samples per chan",
            sort_info["n_samples_per_chan"],
            "from",
            original_n_samples_per_chan,
        )

    return bp_reduction_samples_per_chan, original_clip_starts, original_clip_stops


def get_voltage_for_bp(
    data_dict,
    seg_number,
    use_memmap,
    v_dtype,
):
    """

    Parameters
    ----------
    data_dict
    seg_number: int
    use_memmap: bool
    v_dtype:

    Returns
    -------

    """
    if use_memmap:
        voltage_mmap = MemMapClose(
            data_dict["seg_v_files"][seg_number][0],
            dtype=data_dict["seg_v_files"][seg_number][1],
            mode="r",
            shape=data_dict["seg_v_files"][seg_number][2],
        )
        voltage = memmap_to_mem(
            voltage_mmap, dtype=data_dict["seg_v_files"][seg_number][1]
        )
        if isinstance(voltage_mmap, np.memmap):
            voltage_mmap._mmap.close()
            del voltage_mmap
    else:
        seg_volts_buffer = data_dict["segment_voltages"][seg_number][0]
        seg_volts_shape = data_dict["segment_voltages"][seg_number][1]
        voltage = np.frombuffer(seg_volts_buffer, dtype=v_dtype).reshape(
            seg_volts_shape
        )
    return voltage


def get_channel_covariance_matrix(
    clip_template_residuals,
    n_channels,
    n_samples_per_chan,
    n_cov_samples,
    verbose,
):
    """
    Finds the covariance matrix for each channel by using
    residual variation between clips and robust clip templates

    Parameters
    ----------
    clip_template_residuals: list
        Length of the number of units
        Each element is a numpy array of shape (event_indices vs template_width (all channels)
    n_channels: int
    n_samples_per_chan: int
    n_cov_samples: int
    verbose: bool


    Returns
    -------

    """
    clip_template_residuals = np.vstack(
        clip_template_residuals
    )  # (N_units x event_indices) X template_width

    # Get the noise covariance over time within the binary pursuit clip width
    if verbose:
        print(
            "Computing clip noise covariance for each channel with",
            n_cov_samples,
            "clip samples",
        )
    chan_covariance_mats = []
    for chan in range(0, n_channels):
        t_win = [
            chan * n_samples_per_chan,
            (chan + 1) * n_samples_per_chan,
        ]

        if n_cov_samples >= clip_template_residuals.shape[0]:
            cov_sample_inds = np.arange(0, clip_template_residuals.shape[0])
        else:
            cov_sample_inds = np.random.randint(
                0,
                clip_template_residuals.shape[0],
                n_cov_samples,
            )

        chan_covariance_mats.append(
            np.cov(
                clip_template_residuals[cov_sample_inds, t_win[0] : t_win[1]],
                rowvar=False,
                ddof=0,
            )
        )
    return chan_covariance_mats
