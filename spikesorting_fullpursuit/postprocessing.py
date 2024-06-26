import numpy as np

import spikesorting_fullpursuit.dim_reduce.pca
from spikesorting_fullpursuit.clustering.isocut import merge_clusters
from spikesorting_fullpursuit import analyze_spike_timing
from spikesorting_fullpursuit.processing.clip_utils import get_clips
from spikesorting_fullpursuit.overlap import neuron_separability

import warnings


def delete_neurons_by_snr_mua(
    neurons,
    snr_thresh=2.0,
    mua_thresh=0.10,
    operator="and",
):
    """Delete neurons above SNR threshold and/or above MUA threshold.
    All others are deleted in output list."""
    neurons_to_delete = []
    if operator.lower() == "and":
        for n_ind, n in enumerate(neurons):
            if n["fraction_mua"] < mua_thresh:
                continue
            # has bad mua if made it here so check SNR
            if n["snr"]["average"] > snr_thresh:
                continue
            neurons_to_delete.append(n_ind)
    elif operator.lower() == "or":
        for n_ind, n in enumerate(neurons):
            bad_snr = False
            if n["fraction_mua"] > mua_thresh:
                bad_snr = True
            else:
                # has good mua if made it here so check SNR
                if n["snr"]["average"] < snr_thresh:
                    bad_snr = True
            if bad_snr:
                neurons_to_delete.append(n_ind)
    else:
        raise ValueError("Input operator type must be 'and' or 'or'.")
    for dn in reversed(neurons_to_delete):
        del neurons[dn]

    return neurons


def delete_neurons_by_min_duration(neurons, min_duration):
    """Deletes any units whose spikes span less than min_duration, in seconds."""
    neurons_to_delete = []
    for n_ind, n in enumerate(neurons):
        if n["duration_s"] < min_duration:
            neurons_to_delete.append(n_ind)
    for dn in reversed(neurons_to_delete):
        del neurons[dn]

    return neurons


def delete_neurons_by_min_firing_rate(neurons, min_firing_rate):
    """Deletes any units whose firing rate is less than min_firing_rate."""
    neurons_to_delete = []
    for n_ind, n in enumerate(neurons):
        if n["firing_rate"] < min_firing_rate:
            neurons_to_delete.append(n_ind)
    for dn in reversed(neurons_to_delete):
        del neurons[dn]

    return neurons


def merge_units(neurons, n1_ind, n2_ind):
    """Merge the units corresponding to input indices and outputs the combined
    unit in the lowest index, deleting the neuron from the highest index."""
    if n1_ind < 0 or n2_ind < 0:
        raise ValueError("Neuron indices must be input as positive integers")
    if n1_ind > n2_ind:
        n1_ind, n2_ind = n2_ind, n1_ind
    if n1_ind == n2_ind:
        print("Indices are the same unit so already combined")
        return neurons

    neurons[n1_ind] = combine_two_neurons(neurons[n1_ind], neurons[n2_ind])
    del neurons[n2_ind]

    return neurons


def combine_two_neurons(neuron1, neuron2):
    """Perfrom the combining of two neuron dictionaries into one. Intended as
    the workhorse function for merge_units rather than called directly."""
    combined_neuron = {}
    combined_neuron["sort_info"] = neuron1["sort_info"]
    # Start with values set to neuron1 and merge neuron2 into them
    combined_neuron["channel"] = [x for x in neuron1["channel"]]  # Needs to copy
    combined_neuron["neighbors"] = neuron1["neighbors"]
    combined_neuron["chan_neighbor_ind"] = neuron1["chan_neighbor_ind"]
    combined_neuron["main_windows"] = neuron1["main_windows"]
    n_total_spikes = (
        neuron1["spike_indices"].shape[0] + neuron2["spike_indices"].shape[0]
    )
    max_clip_samples = max(neuron1["clips"].shape[1], neuron2["clips"].shape[1])
    half_clip_inds = int(
        round(
            np.amax(np.abs(neuron1["sort_info"]["clip_width"]))
            * neuron1["sort_info"]["sampling_rate"]
        )
    )
    combined_neuron["duplicate_tol_inds"] = max(
        neuron1["duplicate_tol_inds"], neuron2["duplicate_tol_inds"]
    )
    # Merge neuron2 data channel-wise
    for chan in neuron2["channel"]:
        if chan not in combined_neuron["channel"]:
            combined_neuron["channel"].append(chan)
            combined_neuron["neighbors"][chan] = neuron2["neighbors"][chan]
            combined_neuron["chan_neighbor_ind"][chan] = neuron2["chan_neighbor_ind"][
                chan
            ]
            combined_neuron["main_windows"][chan] = neuron2["main_windows"][chan]

    channel_selector = []
    indices_by_unit = []
    clips_by_unit = []
    bp_spike_bool_by_unit = []
    snr_by_unit = []
    for unit in [neuron1, neuron2]:
        n_unit_events = unit["spike_indices"].shape[0]
        if n_unit_events == 0:
            continue
        indices_by_unit.append(unit["spike_indices"])
        clips_by_unit.append(unit["clips"])
        bp_spike_bool_by_unit.append(unit["binary_pursuit_bool"])
        # Make and append a bunch of book keeping numpy arrays by channel
        chan_select = np.zeros(n_unit_events)
        snr_unit = np.zeros(n_unit_events)
        for chan in unit["channel"]:
            chan_select[unit["channel_selector"][chan]] = chan
            snr_unit[unit["channel_selector"][chan]] = unit["snr"][chan]
        channel_selector.append(chan_select)
        # NOTE: redundantly piling this up makes it easy to track and gives
        # a weighted SNR as its final result
        snr_by_unit.append(snr_unit)

    # Now combine everything into one
    channel_selector = np.hstack(channel_selector)
    snr_by_unit = np.hstack(snr_by_unit)
    combined_neuron["spike_indices"] = np.hstack(indices_by_unit)
    # Need to account for fact that different channels can have different
    # neighborhood sizes. So make all clips start from beginning, and
    # remainder zeroed out if it has no data
    combined_neuron["clips"] = np.zeros(
        (combined_neuron["spike_indices"].shape[0], max_clip_samples),
        neuron1["clips"].dtype,
    )
    clip_start_ind = 0
    for clips in clips_by_unit:
        combined_neuron["clips"][
            clip_start_ind : clips.shape[0] + clip_start_ind, 0 : clips.shape[1]
        ] = clips
        clip_start_ind += clips.shape[0]
    combined_neuron["binary_pursuit_bool"] = np.hstack(bp_spike_bool_by_unit)

    # Cleanup some memory that wasn't overwritten during concatenation
    del indices_by_unit
    del clips_by_unit
    del bp_spike_bool_by_unit

    # NOTE: This still needs to be done even though segments
    # were ordered because of overlap!
    # Ensure everything is ordered. Must use 'stable' sort for
    # output to be repeatable because overlapping segments and
    # binary pursuit can return slightly different dupliate spikes
    spike_order = np.argsort(combined_neuron["spike_indices"], kind="stable")
    combined_neuron["spike_indices"] = combined_neuron["spike_indices"][spike_order]
    combined_neuron["clips"] = combined_neuron["clips"][spike_order, :]
    combined_neuron["binary_pursuit_bool"] = combined_neuron["binary_pursuit_bool"][
        spike_order
    ]
    channel_selector = channel_selector[spike_order]
    snr_by_unit = snr_by_unit[spike_order]

    # # Remove duplicates found in binary pursuit
    # keep_bool = keep_binary_pursuit_duplicates(combined_neuron["spike_indices"],
    #                 combined_neuron["binary_pursuit_bool"],
    #                 tol_inds=half_clip_inds)
    # combined_neuron["spike_indices"] = combined_neuron["spike_indices"][keep_bool]
    # combined_neuron["binary_pursuit_bool"] = combined_neuron["binary_pursuit_bool"][keep_bool]
    # combined_neuron['clips'] = combined_neuron['clips'][keep_bool, :]
    # channel_selector = channel_selector[keep_bool]
    # snr_by_unit = snr_by_unit[keep_bool]

    # Get each spike's channel of origin and the clips on main channel
    combined_neuron["channel_selector"] = {}
    combined_neuron["template"] = {}
    for chan in combined_neuron["channel"]:
        chan_select = channel_selector == chan
        combined_neuron["channel_selector"][chan] = chan_select
        combined_neuron["template"][chan] = np.mean(
            combined_neuron["clips"][chan_select, :], axis=0
        ).astype(neuron1["clips"].dtype)

    # Remove any identical index duplicates (either from error or
    # from combining overlapping segments), preferentially keeping
    # the waveform most similar to its channel's template
    keep_bool = analyze_spike_timing.remove_spike_event_duplicates_across_chans(
        combined_neuron
    )
    combined_neuron["spike_indices"] = combined_neuron["spike_indices"][keep_bool]
    combined_neuron["binary_pursuit_bool"] = combined_neuron["binary_pursuit_bool"][
        keep_bool
    ]
    combined_neuron["clips"] = combined_neuron["clips"][keep_bool, :]
    for chan in combined_neuron["channel"]:
        combined_neuron["channel_selector"][chan] = combined_neuron["channel_selector"][
            chan
        ][keep_bool]
    channel_selector = channel_selector[keep_bool]
    snr_by_unit = snr_by_unit[keep_bool]

    # Recompute things of interest like SNR and templates by channel as the
    # average over all data from that channel and store for output
    combined_neuron["template"] = {}
    combined_neuron["snr"] = {}
    chans_to_remove = []
    for chan in combined_neuron["channel"]:
        if snr_by_unit[combined_neuron["channel_selector"][chan]].size == 0:
            # Spikes contributing from this channel have been removed so
            # remove all its data below
            chans_to_remove.append(chan)
        else:
            combined_neuron["template"][chan] = np.mean(
                combined_neuron["clips"][combined_neuron["channel_selector"][chan], :],
                axis=0,
            ).astype(neuron1["clips"].dtype)
            combined_neuron["snr"][chan] = np.mean(
                snr_by_unit[combined_neuron["channel_selector"][chan]]
            )
    for chan_ind in reversed(range(0, len(chans_to_remove))):
        chan_num = chans_to_remove[chan_ind]
        del combined_neuron["channel"][chan_ind]  # A list so use index
        del combined_neuron["neighbors"][chan_num]  # Rest of these are all
        del combined_neuron["chan_neighbor_ind"][chan_num]  # dictionaries so
        del combined_neuron["channel_selector"][chan_num]  #  use value
        del combined_neuron["main_windows"][chan_num]
    # Weighted average SNR over both units
    combined_neuron["snr"]["average"] = np.mean(snr_by_unit)
    combined_neuron["duration_s"] = (
        combined_neuron["spike_indices"][-1] - combined_neuron["spike_indices"][0]
    ) / (combined_neuron["sort_info"]["sampling_rate"])
    combined_neuron["firing_rate"] = (
        combined_neuron["spike_indices"].shape[0] / combined_neuron["duration_s"]
    )
    combined_neuron["fraction_mua"] = analyze_spike_timing.calc_fraction_mua_to_peak(
        combined_neuron["spike_indices"],
        combined_neuron["sort_info"]["sampling_rate"],
        combined_neuron["duplicate_tol_inds"],
        combined_neuron["sort_info"]["absolute_refractory_period"],
    )

    return combined_neuron


def delete_spike_clips(neurons):
    """Removes the spike clips from each neuron dictionary in neurons to reduce
    memory usage for final output. Should only be used after post-processing steps
    are completed."""
    for n in neurons:
        del n["clips"]

    return neurons


class WorkItemSummary(object):
    """Main class that handles and organizes output of spike_sort function.

    duplicate_tol_inds
        NOTE: this will be added to half spike width, which is roughly the max
        spike sorter shift error
    Really relies on the half clip inds being less than absolute refractory period
    and and that spike alignment shifts do not exceed this value. Should all be
    true for a reasonable choice of clip width.

    Inputs for voltage_file should be binary files '.bin', requiring specification
    of the datatype, or numpy array files '.npy' where datatype is already saved.
    If clips are input, AND a voltage file is given, the voltage file is ignored.
    It is assumed that the entire voltage file can be read into memory. It is
    otherwise very slow to randomly index the entire voltage array to find clips.
    Voltage will be shaped into the n_channels x n_samples as specified in
    sort_info.
    """

    def __init__(
        self,
        sort_data,
        work_items,
        sort_info,
        absolute_refractory_period=12e-4,
        max_mua_ratio=0.2,
        min_snr=1.5,
        min_overlapping_spikes=0.75,
        stitch_overlap_only=True,
        skip_organization=False,
        n_segments=None,
        verbose=False,
        voltage_file=None,
        voltage_dtype=None,
    ):
        if (
            (voltage_file is not None)
            and (voltage_dtype is None)
            and (voltage_file[-4:] != ".npy")
        ):
            raise ValueError(
                "Must specify the voltage data type for input voltage files of type '{0}'.".format(
                    voltage_file[-4:]
                )
            )
        self.voltage_file = voltage_file
        if not skip_organization:
            self.check_input_data(sort_data, work_items)
        else:
            self.sort_data = sort_data
            self.work_items = work_items
        self.sort_info = sort_info
        # These are used so frequently they are aliased for convenience
        self.n_chans = self.sort_info["n_channels"]
        if n_segments is None:
            self.n_segments = self.sort_info["n_segments"]
        else:
            self.n_segments = n_segments
        self.new2orig_seg_labels = [{} for x in range(0, self.n_segments)]
        self.half_clip_inds = int(
            round(
                np.amax(np.abs(self.sort_info["clip_width"]))
                * self.sort_info["sampling_rate"]
            )
        )
        self.absolute_refractory_period = absolute_refractory_period
        self.max_mua_ratio = max_mua_ratio
        self.min_snr = min_snr
        self.print_duplicate_tol_fail = True
        if stitch_overlap_only:
            if not skip_organization:
                self.overlap_indices = work_items[0]["overlap"]
            else:
                self.overlap_indices = work_items[0][0]["overlap"]
            if self.overlap_indices <= 0:
                print(
                    "No overlap between segments found. Switching stitch_overlap_only to False."
                )
                stitch_overlap_only = False
                self.overlap_indices = 0
        else:
            self.overlap_indices = np.inf
        self.stitch_overlap_only = stitch_overlap_only
        self.min_overlapping_spikes = min_overlapping_spikes
        self.is_stitched = False  # Repeated stitching can change results so track
        self.last_overlap_ratio_threshold = (
            np.inf
        )  # Track how much overlap deleted units
        self.verbose = verbose
        # Organize sort_data to be arranged for stitching and summarizing
        if not skip_organization:
            self.organize_sort_data()
            self.temporal_order_sort_data()
        else:
            self.neuron_summary_seg_inds = []
            for seg in range(0, self.n_segments):
                self.neuron_summary_seg_inds.append(
                    self.work_items[0][seg]["index_window"]
                )
        if self.voltage_file is not None:
            # Load voltage into memory
            if self.voltage_file[-4:] == ".bin":
                self.voltage = np.fromfile(self.voltage_file, dtype=voltage_dtype)
                self.voltage = np.reshape(
                    self.voltage, (sort_info["n_channels"], sort_info["n_samples"])
                )
            elif self.voltage_file[-4:] == ".npy":
                self.voltage = np.load(self.voltage_file)
            else:
                raise ValueError(
                    "Unrecognized voltage file type. Accepted types are '.bin' and '.npy'."
                )
            self.clip_dict = {
                "sampling_rate": sort_info["sampling_rate"],
                "n_samples": sort_info["n_samples"],
                "v_dtype": self.voltage.dtype,
            }
        else:
            self.voltage = None
        if not skip_organization:
            self.delete_bad_mua_snr_units()

    def check_input_data(self, sort_data, work_items):
        """Quick check to see if everything looks right with sort_data and
        work_items before proceeding."""
        if len(sort_data) != len(work_items):
            raise ValueError(
                "Sort data and work items must be lists of the same length"
            )
        # Make sure sort_data and work items are ordered one to one
        # by ordering their IDs
        sort_data.sort(key=lambda x: x[4])
        work_items.sort(key=lambda x: x["ID"])
        for s_data, w_item in zip(sort_data, work_items):
            if len(s_data) != 5:
                raise ValueError(
                    "sort_data for item", s_data[4], "is not the right size"
                )
            if s_data[4] != w_item["ID"]:
                raise ValueError(
                    "Could not match work order of sort data and work items. First failure on sort_data",
                    s_data[4],
                    "work item ID",
                    w_item["ID"],
                )
            empty_count = 0
            n_spikes = len(s_data[0])
            if n_spikes > 1:
                if (s_data[2].size == 0) or (len(s_data[2]) == 0):
                    if self.voltage_file is None:
                        raise ValueError(
                            "Spike clips not found and no voltage file specified. Must include a voltage file to continue!"
                        )
                else:
                    if (self.voltage_file is not None) and (
                        s_data[2].shape[0] == n_spikes
                    ):
                        print(
                            "Both clips and voltage file given. Ignoring voltage file."
                        )
                        self.voltage_file = None
            for di in range(0, 4):
                if len(s_data[di]) == 0:
                    empty_count += 1
                else:
                    # This is caught below by empty_count !=4 ...
                    pass
            if empty_count > 0:
                # Require that if any data element is empty, all are empty (this
                # is how they should be coming out of sorter)
                for di in range(0, 4):
                    s_data[di] = []
                if empty_count != 4:
                    sort_data_message = (
                        "One element of sort data, but not "
                        "all, indicated no spikes for work "
                        "item {0} on channel {1}, segment "
                        "{2}. All data for this item are "
                        "being removed.".format(
                            w_item["ID"], w_item["channel"], w_item["seg_number"]
                        )
                    )
                    warnings.warn(sort_data_message, RuntimeWarning, stacklevel=2)
        self.sort_data = sort_data
        self.work_items = work_items

    def organize_sort_data(self):
        """Takes the output of the spike sorting algorithms and organizes is by
        channel in segment order. Also organizes the labels of units within
        segments and maps to original label indices from 0-N within each segment.
        Parameters
        ----------
        sort_data: list
            Each element contains a list of the 4 outputs of spike_sort_segment,
            and the ID of the work item that created it, split up by
            individual channels into 'work_items' (or spike_sort_parallel output)
            "crossings, labels, clips, binary_pursuit", in this order, for a
            single segment and a single channel of sorted data.
            len(sort_data) = number segments * channels == len(work_items)
        work_items: list
            Each list element contains the dictionary of information pertaining to
            the corresponding element in sort_data.
        n_chans: int
            Total number of channels sorted. If None this number is inferred as
            the maximum channel found in work_items['channel'] + 1.
        Returns
        -------
        Returns None but reassigns class attributes as follows:
        self.sort_data : sort_data list of dictionaries
            A list of the original elements of sort_data and organzied such that
            each list index corresponds to a channel number, and within each channel
            sublist data are in ordered by segment number as indicated in work_items.
        self.work_items : work_items list of dictionaries
            Same as for organized data but for the elements of work_items.
        """
        organized_data = [[] for x in range(0, self.n_chans)]
        organized_items = [[] for x in range(0, self.n_chans)]
        for chan in range(0, self.n_chans):
            chan_items, chan_data = zip(
                *[
                    [x, y]
                    for x, y in sorted(
                        zip(self.work_items, self.sort_data),
                        key=lambda pair: pair[0]["seg_number"],
                    )
                    if x["channel"] == chan
                ]
            )
            organized_data[chan].extend(chan_data)
            organized_items[chan].extend(chan_items)
        self.sort_data = organized_data
        self.work_items = organized_items

        self.neuron_summary_seg_inds = []
        for seg in range(0, self.n_segments):
            self.neuron_summary_seg_inds.append(self.work_items[0][seg]["index_window"])
            all_seg_labels = []
            # First find all the seg labels
            for chan in range(0, self.n_chans):
                if len(self.sort_data[chan][seg][0]) == 0:
                    continue  # No spikes in this segment
                all_seg_labels = np.hstack(
                    (all_seg_labels, np.unique(self.sort_data[chan][seg][1]))
                )
            if len(all_seg_labels) == 0:
                continue
            # Go through labels and reassign to lowest label starting with 0
            next_new_label = 0
            # Unique sorts so going through lowest to highest
            for curr_label in np.unique(all_seg_labels):
                for chan in range(0, self.n_chans):
                    if np.any(self.sort_data[chan][seg][1] == curr_label):
                        # We found the channel with current lowest label, reassign
                        curr_select = self.sort_data[chan][seg][1] == curr_label
                        self.sort_data[chan][seg][1][curr_select] = next_new_label
                        next_new_label += 1
                        break

    def temporal_order_sort_data(self):
        """Places all data within each segment in temporal order according to
        the spike event times. Use 'stable' sort for output to be repeatable
        because overlapping segments and binary pursuit can return identical
        dupliate spikes that become sorted in different orders. Moves labels
        to be non-overlapping across segments and updates dict tracking of this."""

        min_seg_label = 0
        max_seg_label = 0
        for seg in range(0, self.n_segments):
            for chan in range(0, self.n_chans):
                if len(self.sort_data[chan][seg][0]) == 0:
                    continue  # No spikes in this segment
                spike_order = np.argsort(self.sort_data[chan][seg][0], kind="stable")
                self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][spike_order]
                self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][spike_order]
                self.sort_data[chan][seg][
                    1
                ] += min_seg_label  # Increment to get unique labels for each segment
                # Update segment label tracking
                for n_label in np.unique(self.sort_data[chan][seg][1]):
                    self.new2orig_seg_labels[seg][n_label] = n_label - min_seg_label
                curr_max_label = np.amax(self.sort_data[chan][seg][1])
                if curr_max_label > max_seg_label:
                    max_seg_label = curr_max_label
                if self.voltage_file is None:
                    self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][
                        spike_order, :
                    ]
                self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][spike_order]
            min_seg_label = max_seg_label + 1

    def delete_label(self, chan, seg, label):
        """Remove the unit corresponding to label from input segment and channel."""
        # NOTE: If all items are deleted, data will become empty numpy array
        # rather than the empty list style of original input
        keep_indices = self.sort_data[chan][seg][1] != label
        self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][keep_indices]
        self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][keep_indices]
        if self.voltage is None:
            self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][keep_indices, :]
        self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][keep_indices]
        return keep_indices

    def delete_bad_mua_snr_units(self):
        """Goes through each sorted item and removes any units with an SNR less
        than input min_snr or fraction MUA greater than max_mua_ratio.

        Function tracks and prints the lowest fraction_mua that was removed and
        the maximum SNR that was removed. This allows the user to understand
        whether any units of interest had segments that were 'just barely'
        removed under the current criteria."""
        min_removed_mua = np.inf
        min_removed_mua_chan = None
        min_removed_mua_seg = None
        max_removed_snr = -np.inf
        max_removed_snr_chan = None
        max_removed_snr_seg = None
        for chan in range(0, self.n_chans):
            for seg in range(0, self.n_segments):
                # NOTE: This will just skip if no data in segment
                for l in np.unique(self.sort_data[chan][seg][1]):
                    select = self.sort_data[chan][seg][1] == l
                    if self.voltage is None:
                        curr_template = np.mean(
                            self.sort_data[chan][seg][2][select, :], axis=0
                        )
                    else:
                        curr_template, _ = get_clips(
                            self.clip_dict,
                            self.voltage,
                            self.work_items[chan][seg]["neighbors"],
                            self.sort_data[chan][seg][0][select],
                            self.sort_info["clip_width"],
                            check_valid=False,
                        )
                        curr_template = np.mean(curr_template, axis=0)
                    mua_ratio = self.get_fraction_mua_to_peak(
                        chan, seg, l, curr_template
                    )
                    if mua_ratio > self.max_mua_ratio:
                        self.delete_label(chan, seg, l)
                        if mua_ratio < min_removed_mua:
                            min_removed_mua = mua_ratio
                            min_removed_mua_chan = chan
                            min_removed_mua_seg = seg
                        # Will skip computing SNR, but will not track max
                        # removed accurately
                        continue

                    snr = self.get_snr(chan, seg, curr_template)
                    if snr < self.min_snr:
                        self.delete_label(chan, seg, l)
                        if snr > max_removed_snr:
                            max_removed_snr = snr
                            max_removed_snr_chan = chan
                            max_removed_snr_seg = seg
        print(
            "Least MUA removed was",
            min_removed_mua,
            "on channel",
            min_removed_mua_chan,
            "segment",
            min_removed_mua_seg,
        )
        print(
            "Maximum SNR removed was",
            max_removed_snr,
            "on channel",
            max_removed_snr_chan,
            "segment",
            max_removed_snr_seg,
        )

    def remove_segment_binary_pursuit_duplicates(self):
        """Removes overlapping spikes that were both found by binary
        pursuit. This can remove double dipping artifacts as binary pursuit attempts
        to minimize residual error. This only applies for data sorted within the
        same channel, segment and belonging to the same unit. Binary pursuit
        shouldn't screw up by more than the half clip width."""
        for chan in range(0, self.n_chans):
            for seg in range(0, self.n_segments):
                if len(self.sort_data[chan][seg][0]) == 0:
                    # No data in this segment
                    continue
                keep_bool = np.ones(self.sort_data[chan][seg][0].shape[0], dtype="bool")
                for l in np.unique(self.sort_data[chan][seg][1]):
                    unit_select = self.sort_data[chan][seg][1] == l
                    keep_bool[unit_select] = remove_binary_pursuit_duplicates(
                        self.sort_data[chan][seg][0][unit_select],
                        self.sort_data[chan][seg][2][unit_select, :],
                        np.mean(self.sort_data[chan][seg][2][unit_select, :], axis=0),
                        self.sort_data[chan][seg][3][unit_select],
                        2 * self.half_clip_inds,
                    )
                # NOTE: If all items are deleted, data will become empty numpy array
                # rather than the empty list style of original input
                self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][keep_bool]
                self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][keep_bool]
                self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][
                    keep_bool, :
                ]
                self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][keep_bool]

    def get_snr(self, chan, seg, full_template):
        """Get SNR on the main channel relative to 3 STD of background noise."""
        background_noise_std = (
            self.work_items[chan][seg]["thresholds"][
                self.work_items[chan][seg]["chan_neighbor_ind"]
            ]
            / self.sort_info["sigma"]
        )
        main_win = [
            self.sort_info["n_samples_per_chan"]
            * self.work_items[chan][seg]["chan_neighbor_ind"],
            self.sort_info["n_samples_per_chan"]
            * (self.work_items[chan][seg]["chan_neighbor_ind"] + 1),
        ]
        main_template = full_template[main_win[0] : main_win[1]]
        temp_range = np.amax(main_template) - np.amin(main_template)
        return temp_range / (3 * background_noise_std)

    def get_isi_violation_rate(self, chan, seg, label):
        """Compute the spiking activity that occurs during the ISI violation
        period, absolute_refractory_period, relative to the total number of
        spikes within the given segment, 'seg' and unit label 'label'."""
        select_unit = self.sort_data[chan][seg][1] == label
        if ~np.any(select_unit):
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        unit_spikes = self.sort_data[chan][seg][0][select_unit]
        main_win = [
            self.sort_info["n_samples_per_chan"]
            * self.work_items[chan][seg]["chan_neighbor_ind"],
            self.sort_info["n_samples_per_chan"]
            * (self.work_items[chan][seg]["chan_neighbor_ind"] + 1),
        ]
        # Within channel alignment shouldn't be off by more than about half spike width
        duplicate_tol_inds = (
            analyze_spike_timing.calc_spike_half_width(
                np.mean(
                    self.sort_data[chan][seg][2][select_unit][
                        :, main_win[0] : main_win[1]
                    ],
                    axis=0,
                )
            )
            + 1
        )
        refractory_adjustment = duplicate_tol_inds / self.sort_info["sampling_rate"]
        if (self.absolute_refractory_period - refractory_adjustment) <= 0:
            if self.print_duplicate_tol_fail:
                print(
                    "Duplicate_tol_inds encompasses absolute_refractory_period. MUA can't be calculated for this unit."
                )
                self.print_duplicate_tol_fail = False
            return np.nan
        index_isi = np.diff(unit_spikes)
        num_isi_violations = np.count_nonzero(
            index_isi / self.sort_info["sampling_rate"]
            < self.absolute_refractory_period
        )
        n_duplicates = np.count_nonzero(index_isi <= duplicate_tol_inds)
        # Remove duplicate spikes from this computation and adjust the number
        # of spikes and time window accordingly
        num_isi_violations -= n_duplicates
        isi_violation_rate = (
            num_isi_violations
            * (1.0 / (self.absolute_refractory_period - refractory_adjustment))
            / (np.count_nonzero(select_unit) - n_duplicates)
        )
        return isi_violation_rate

    def get_mean_firing_rate(self, chan, seg, label):
        """Compute mean firing rate for the input channel, segment, and label."""
        select_unit = self.sort_data[chan][seg][1] == label
        first_index = next(
            (idx[0] for idx, val in np.ndenumerate(select_unit) if val), [None]
        )
        if first_index is None:
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        last_index = next(
            (
                select_unit.shape[0] - idx[0] - 1
                for idx, val in np.ndenumerate(select_unit[::-1])
                if val
            ),
            None,
        )
        if (
            self.sort_data[chan][seg][0][first_index]
            == self.sort_data[chan][seg][0][last_index]
        ):
            return 0.0  # Only one spike
        mean_rate = (
            self.sort_info["sampling_rate"]
            * np.count_nonzero(select_unit)
            / (
                self.sort_data[chan][seg][0][last_index]
                - self.sort_data[chan][seg][0][first_index]
            )
        )
        return mean_rate

    def get_fraction_mua(self, chan, seg, label):
        """Estimate the fraction of noise/multi-unit activity (MUA) using analysis
        of ISI violations. We do this by looking at the spiking activity that
        occurs during the ISI violation period."""
        isi_violation_rate = self.get_isi_violation_rate(chan, seg, label)
        mean_rate = self.get_mean_firing_rate(chan, seg, label)
        if mean_rate == 0.0:
            return 0.0
        else:
            return isi_violation_rate / mean_rate

    def get_fraction_mua_to_peak(
        self,
        chan,
        seg,
        l,
        template,
        check_window=0.5,
    ):
        """ """
        main_win = [
            self.sort_info["n_samples_per_chan"]
            * self.work_items[chan][seg]["chan_neighbor_ind"],
            self.sort_info["n_samples_per_chan"]
            * (self.work_items[chan][seg]["chan_neighbor_ind"] + 1),
        ]
        # Within channel alignment shouldn't be off by more than about half spike width
        duplicate_tol_inds = (
            analyze_spike_timing.calc_spike_half_width(
                template[main_win[0] : main_win[1]]
            )
            + 1
        )
        unit_spikes = self.sort_data[chan][seg][0][self.sort_data[chan][seg][1] == l]
        all_isis = np.diff(unit_spikes)
        refractory_inds = int(
            round(self.absolute_refractory_period * self.sort_info["sampling_rate"])
        )
        bin_width = refractory_inds - duplicate_tol_inds
        if bin_width <= 0:
            if self.print_duplicate_tol_fail:
                print(
                    "Duplicate_tol_inds encompasses absolute_refractory_period so fraction MUA cannot be computed."
                )
                self.print_duplicate_tol_fail = False
            return np.nan
        check_inds = int(round(check_window * self.sort_info["sampling_rate"]))
        bin_edges = np.arange(duplicate_tol_inds, check_inds + bin_width, bin_width)
        counts, xval = np.histogram(all_isis, bin_edges)
        isi_peak = np.amax(counts)
        num_isi_violations = counts[0]
        if num_isi_violations < 0:
            num_isi_violations = 0
        if isi_peak == 0:
            isi_peak = max(num_isi_violations, 1.0)
        fraction_mua_to_peak = num_isi_violations / isi_peak

        return fraction_mua_to_peak

    def merge_test_two_units(
        self,
        clips_1,
        clips_2,
        p_cut,
        method="template_pca",
        split_only=False,
        merge_only=False,
        use_weights=True,
        curr_chan_inds=None,
    ):
        if self.sort_info["add_peak_valley"] and curr_chan_inds is None:
            raise ValueError("Must give curr_chan_inds if using peak valley.")
        clips = np.vstack((clips_1, clips_2))
        neuron_labels = np.ones(clips.shape[0], dtype=np.int64)
        neuron_labels[clips_1.shape[0] :] = 2
        if method.lower() == "pca":
            scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca(
                clips,
                self.sort_info["check_components"],
                self.sort_info["max_components"],
                add_peak_valley=self.sort_info["add_peak_valley"],
                curr_chan_inds=curr_chan_inds,
            )
        elif method.lower() == "template_pca":
            scores = spikesorting_fullpursuit.dim_reduce.pca.compute_template_pca(
                clips,
                neuron_labels,
                curr_chan_inds,
                self.sort_info["check_components"],
                self.sort_info["max_components"],
                add_peak_valley=self.sort_info["add_peak_valley"],
                use_weights=use_weights,
            )
        elif method.lower() == "channel_template_pca":
            scores = (
                spikesorting_fullpursuit.dim_reduce.pca.compute_template_pca_by_channel(
                    clips,
                    neuron_labels,
                    curr_chan_inds,
                    self.sort_info["check_components"],
                    self.sort_info["max_components"],
                    add_peak_valley=self.sort_info["add_peak_valley"],
                    use_weights=use_weights,
                )
            )
        elif method.lower() == "projection":
            # Projection onto templates, weighted by number of spikes
            t1 = np.mean(clips_1, axis=0)
            t2 = np.mean(clips_2, axis=0)
            if use_weights:
                t1 *= clips_1.shape[0] / clips.shape[0]
                t2 *= clips_2.shape[0] / clips.shape[0]
            scores = clips @ np.vstack((t1, t2)).T
        else:
            raise ValueError(
                "Unknown method", method, "for scores. Must use 'pca' or 'projection'."
            )
        scores = np.float64(scores)
        neuron_labels = merge_clusters(
            scores,
            neuron_labels,
            split_only=split_only,
            merge_only=merge_only,
            p_value_cut_thresh=p_cut,
            match_cluster_size=self.sort_info["match_cluster_size"],
            check_splits=self.sort_info["check_splits"],
        )

        label_is_1 = neuron_labels == 1
        label_is_2 = neuron_labels == 2
        if np.all(label_is_1) or np.all(label_is_2):
            clips_merged = True
        else:
            clips_merged = False
        neuron_labels_1 = neuron_labels[0 : clips_1.shape[0]]
        neuron_labels_2 = neuron_labels[clips_1.shape[0] :]
        return clips_merged, neuron_labels_1, neuron_labels_2

    def find_nearest_shifted_pair(
        self,
        chan,
        seg1,
        seg2,
        labels1,
        labels2,
        l2_workspace,
        curr_chan_inds,
        previously_compared_pairs,
        max_shift_inds=np.inf,
    ):
        """Alternative to sort_cython.identify_clusters_to_compare that simply
        chooses the nearest template after shifting to optimal alignment.
        Intended as helper function so that neurons do not fail to stitch in the
        event their alignment changes between segments."""
        best_distance = np.inf
        for l1 in labels1:
            # Choose seg1 template
            l1_select = self.sort_data[chan][seg1][1] == l1
            if self.voltage is None:
                clips_1 = self.sort_data[chan][seg1][2][l1_select, :]
            else:
                clips_1, _ = get_clips(
                    self.clip_dict,
                    self.voltage,
                    self.work_items[chan][seg1]["neighbors"],
                    self.sort_data[chan][seg1][0][l1_select],
                    self.sort_info["clip_width"],
                    check_valid=False,
                )
            # Only compare clips within specified overlap, if these are in a
            # different segment. If same segment use all clips
            if seg1 != seg2:
                if self.stitch_overlap_only:
                    start_edge = (
                        self.neuron_summary_seg_inds[seg1][1] - self.overlap_indices
                    )
                else:
                    start_edge = self.neuron_summary_seg_inds[seg1][0]
                c1_start = next(
                    (
                        idx[0]
                        for idx, val in np.ndenumerate(
                            self.sort_data[chan][seg1][0][l1_select]
                        )
                        if val >= start_edge
                    ),
                    None,
                )
                if c1_start is None:
                    continue
                if c1_start == clips_1.shape[0]:
                    continue
                clips_1 = clips_1[c1_start:, :]
            l1_template = np.mean(clips_1, axis=0)
            for l2 in labels2:
                if [l1, l2] in previously_compared_pairs:
                    continue
                l2_select = l2_workspace == l2
                if self.voltage is None:
                    clips_2 = self.sort_data[chan][seg2][2][l2_select, :]
                else:
                    clips_2, _ = get_clips(
                        self.clip_dict,
                        self.voltage,
                        self.work_items[chan][seg2]["neighbors"],
                        self.sort_data[chan][seg2][0][l2_select],
                        self.sort_info["clip_width"],
                        check_valid=False,
                    )
                if seg1 != seg2:
                    if self.stitch_overlap_only:
                        # Stop at end of seg1 as this is end of overlap!
                        stop_edge = self.neuron_summary_seg_inds[seg1][1]
                    else:
                        stop_edge = self.neuron_summary_seg_inds[seg2][1]
                    c2_start = next(
                        (
                            idx[0]
                            for idx, val in np.ndenumerate(
                                self.sort_data[chan][seg2][0][l2_select]
                            )
                            if val >= start_edge
                        ),
                        None,
                    )
                    if c2_start is None:
                        continue
                    c2_stop = next(
                        (
                            idx[0]
                            for idx, val in np.ndenumerate(
                                self.sort_data[chan][seg2][0][l2_select]
                            )
                            if val > stop_edge
                        ),
                        None,
                    )
                    if c2_start == c2_stop:
                        continue
                    clips_2 = clips_2[c2_start:c2_stop, :]
                l2_template = np.mean(clips_2, axis=0)
                cross_corr = np.correlate(
                    l1_template[curr_chan_inds],
                    l2_template[curr_chan_inds],
                    mode="full",
                )
                max_corr_ind = np.argmax(cross_corr)
                curr_shift = max_corr_ind - cross_corr.shape[0] // 2
                if np.abs(curr_shift) > max_shift_inds:
                    continue
                # Align and truncate template and compute distance
                if curr_shift > 0:
                    shiftl1 = l1_template[curr_shift:]
                    shiftl2 = l2_template[: -1 * curr_shift]
                elif curr_shift < 0:
                    shiftl1 = l1_template[:curr_shift]
                    shiftl2 = l2_template[-1 * curr_shift :]
                else:
                    shiftl1 = l1_template
                    shiftl2 = l2_template
                # Must normalize distance per data point else reward big shifts
                curr_distance = np.sum((shiftl1 - shiftl2) ** 2) / shiftl1.shape[0]
                if curr_distance < best_distance:
                    best_distance = curr_distance
                    best_shift = curr_shift
                    best_pair = [l1, l2]
                    best_l1_clips = clips_1
                    best_l2_clips = clips_2
        if np.isinf(best_distance):
            # Never found a match
            best_pair = []
            best_shift = 0
            clips_1 = None
            clips_2 = None
            return best_pair, best_shift, clips_1, clips_2, curr_chan_inds
        if best_shift == 0:
            # No shift so just return as is
            return best_pair, best_shift, best_l1_clips, best_l2_clips, curr_chan_inds
        # Align and truncate clips for best match pair
        shift_samples_per_chan = self.sort_info["n_samples_per_chan"] - np.abs(
            best_shift
        )
        clips_1 = np.zeros(
            (
                best_l1_clips.shape[0],
                shift_samples_per_chan
                * self.work_items[chan][seg1]["neighbors"].shape[0],
            ),
            dtype=best_l1_clips.dtype,
        )
        clips_2 = np.zeros(
            (
                best_l2_clips.shape[0],
                shift_samples_per_chan
                * self.work_items[chan][seg2]["neighbors"].shape[0],
            ),
            dtype=best_l2_clips.dtype,
        )
        # Get clips for each channel, shift them, and assign for output, which
        # will be clips that have each channel individually aligned and
        # truncated
        for chan_ind in range(0, self.work_items[chan][seg1]["neighbors"].shape[0]):
            chan_clips_1 = best_l1_clips[
                :,
                chan_ind
                * self.sort_info["n_samples_per_chan"] : (chan_ind + 1)
                * self.sort_info["n_samples_per_chan"],
            ]
            chan_clips_2 = best_l2_clips[
                :,
                chan_ind
                * self.sort_info["n_samples_per_chan"] : (chan_ind + 1)
                * self.sort_info["n_samples_per_chan"],
            ]
            if best_shift > 0:
                clips_1[
                    :,
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan,
                ] = chan_clips_1[:, best_shift:]
                clips_2[
                    :,
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan,
                ] = chan_clips_2[:, : -1 * best_shift]
            elif best_shift < 0:
                clips_1[
                    :,
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan,
                ] = chan_clips_1[:, :best_shift]
                clips_2[
                    :,
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan,
                ] = chan_clips_2[:, -1 * best_shift :]
            if self.work_items[chan][seg1]["neighbors"][chan_ind] == chan:
                curr_chan_inds = np.arange(
                    chan_ind * shift_samples_per_chan,
                    (chan_ind + 1) * shift_samples_per_chan,
                    dtype=np.int64,
                )
        return best_pair, best_shift, clips_1, clips_2, curr_chan_inds

    def sharpen_segments(self):
        """Not currently called ever because it could be a bad idea."""
        for chan in range(0, self.n_chans):
            # Main win is fixed for a given channel
            main_win = [
                self.sort_info["n_samples_per_chan"]
                * self.work_items[chan][0]["chan_neighbor_ind"],
                self.sort_info["n_samples_per_chan"]
                * (self.work_items[chan][0]["chan_neighbor_ind"] + 1),
            ]
            for seg in range(0, self.n_segments):
                if len(self.sort_data[chan][seg][0]) == 0:
                    # No data in this segment
                    continue
                seg_labels = np.unique(self.sort_data[chan][seg][1]).tolist()
                # Do not compare same unit to itself
                previously_compared_pairs = [[sl, sl] for sl in seg_labels]
                while len(seg_labels) > 1:
                    # Must be reset each iteration
                    curr_chan_inds = np.arange(main_win[0], main_win[1], dtype=np.int64)
                    (
                        best_pair,
                        best_shift,
                        clips_1,
                        clips_2,
                        curr_chan_inds,
                    ) = self.find_nearest_shifted_pair(
                        chan,
                        seg,
                        seg,
                        seg_labels,
                        seg_labels,
                        self.sort_data[chan][seg][1],
                        curr_chan_inds,
                        previously_compared_pairs,
                        self.half_clip_inds,
                    )
                    if len(best_pair) == 0:
                        break
                    if clips_1.shape[0] == 1 or clips_2.shape[0] == 1:
                        # Don't mess around with only 1 spike, if they are
                        # nearest each other they can merge
                        is_merged = True
                    else:
                        is_merged, _, _ = self.merge_test_two_units(
                            clips_1,
                            clips_2,
                            self.sort_info["p_value_cut_thresh"],
                            method="template_pca",
                            merge_only=True,
                            curr_chan_inds=curr_chan_inds,
                            use_weights=True,
                        )
                    if is_merged:
                        select_1 = self.sort_data[chan][seg][1] == best_pair[0]
                        select_2 = self.sort_data[chan][seg][1] == best_pair[1]
                        union_spikes = np.hstack(
                            (
                                self.sort_data[chan][seg][0][select_1],
                                self.sort_data[chan][seg][0][select_2],
                            )
                        )
                        union_clips = np.vstack((clips_1, clips_2))
                        union_binary_pursuit_bool = np.hstack(
                            (
                                self.sort_data[chan][seg][3][select_1],
                                self.sort_data[chan][seg][3][select_2],
                            )
                        )
                        spike_order = np.argsort(union_spikes, kind="stable")
                        union_spikes = union_spikes[spike_order]
                        union_clips = union_clips[spike_order, :]
                        union_binary_pursuit_bool = union_binary_pursuit_bool[
                            spike_order
                        ]

                        # Remove any identical index duplicates (either from error or
                        # from combining overlapping segments), preferentially keeping
                        # the waveform best aligned to the template
                        union_template = np.mean(union_clips, axis=0)
                        # We are unioning spikes that may need sharpened due
                        # to alignment problem so use full spike width tol inds
                        spike_half_width = (
                            analyze_spike_timing.calc_spike_half_width(
                                np.mean(union_clips[:, curr_chan_inds], axis=0)
                            )
                            + 1
                        )
                        keep_bool = remove_spike_event_duplicates(
                            union_spikes,
                            union_clips,
                            union_template,
                            tol_inds=2 * spike_half_width,
                        )
                        union_spikes = union_spikes[keep_bool]
                        union_binary_pursuit_bool = union_binary_pursuit_bool[keep_bool]
                        union_clips = union_clips[keep_bool, :]

                        union_fraction_mua_rate = (
                            analyze_spike_timing.calc_fraction_mua(
                                union_spikes,
                                self.sort_info["sampling_rate"],
                                2 * spike_half_width,
                                self.absolute_refractory_period,
                            )
                        )
                        # Need to get fraction MUA by rate, rather than peak,
                        # for comparison here
                        fraction_mua_rate_1 = analyze_spike_timing.calc_fraction_mua(
                            self.sort_data[chan][seg][0][select_1],
                            self.sort_info["sampling_rate"],
                            2 * spike_half_width,
                            self.absolute_refractory_period,
                        )
                        fraction_mua_rate_2 = analyze_spike_timing.calc_fraction_mua(
                            self.sort_data[chan][seg][0][select_2],
                            self.sort_info["sampling_rate"],
                            2 * spike_half_width,
                            self.absolute_refractory_period,
                        )
                        # Use the biggest bin over bin change observed in the
                        # CCG as an error tolerance
                        _, _, max_delta = analyze_spike_timing.calc_ccg_overlap_ratios(
                            self.sort_data[chan][seg][0][select_1],
                            self.sort_data[chan][seg][0][select_2],
                            self.absolute_refractory_period,
                            self.sort_info["sampling_rate"],
                        )
                        if (
                            union_fraction_mua_rate
                            > min(fraction_mua_rate_1, fraction_mua_rate_2) + max_delta
                        ):
                            is_merged = False

                    if self.verbose:
                        print(
                            "Item",
                            self.work_items[chan][seg]["ID"],
                            "on chan",
                            chan,
                            "seg",
                            seg,
                            "merged",
                            is_merged,
                            "for labels",
                            best_pair,
                        )

                    if is_merged:
                        # Combine these into whichever label has more spikes
                        if np.count_nonzero(select_1) > np.count_nonzero(select_2):
                            self.sort_data[chan][seg][1][select_2] = best_pair[0]
                            seg_labels.remove(best_pair[1])
                            self.sort_data[chan][seg][0][select_2] += -1 * best_shift
                        else:
                            self.sort_data[chan][seg][1][select_1] = best_pair[1]
                            seg_labels.remove(best_pair[0])
                            self.sort_data[chan][seg][0][select_1] += best_shift
                    else:
                        # These mutually closest failed so do not repeat either
                        seg_labels.remove(best_pair[0])
                        seg_labels.remove(best_pair[1])
                    previously_compared_pairs.append(best_pair)

    def find_nearest_joint_pair(
        self,
        templates,
        labels,
        previously_compared_pairs,
    ):
        """
        This does NOT currently consider shifts. Using shifts here gets
        complicated and poses additional risk for errors."""
        best_distance = np.inf
        best_pair = []
        for i in range(0, len(labels)):
            for j in range(i + 1, len(labels)):
                if [labels[i], labels[j]] in previously_compared_pairs:
                    continue
                curr_distance = (
                    np.sum((templates[i] - templates[j]) ** 2) / templates[i].shape[0]
                )
                if curr_distance < best_distance:
                    best_pair = [labels[i], labels[j]]
        return best_pair

    def stitch_segments(self):
        """
        Returns
        -------
        Returns None but reassigns class attributes as follows:
        self.sort_data[1] : np.int64
            Reassigns the labels across segments within each channel so that
            they have the same meaning.
        """
        if self.is_stitched:
            print(
                "Neurons are already stitched. Repeated stitching can change results."
            )
            print("Skipped stitching")
            return

        update_new2orig_seg_labels = [[] for _ in range(0, len(self.sort_data[0]))]
        # Stitch each channel separately
        for chan in range(0, self.n_chans):
            print("Start stitching channel", chan)
            jump_to_end = False
            if len(self.sort_data[chan]) <= 1:
                # Need at least 2 segments to stitch
                jump_to_end = True
            # Start with the current labeling scheme in first segment,
            # which is assumed to be ordered from 0-N (as output by sorter)
            # Find first segment with labels and start there
            start_seg = 0
            while start_seg < len(self.sort_data[chan]):
                if len(self.sort_data[chan][start_seg][1]) == 0:
                    start_seg += 1
                    continue
                real_labels = np.unique(self.sort_data[chan][start_seg][1]).tolist()
                next_real_label = max(real_labels) + 1
                if len(real_labels) > 0:
                    break
                start_seg += 1
            start_new_seg = False
            if start_seg >= len(self.sort_data[chan]) - 1:
                # Need at least 2 remaining segments to stitch
                jump_to_end = True
            else:
                # Main win is fixed for a given channel
                main_win = [
                    self.sort_info["n_samples_per_chan"]
                    * self.work_items[chan][start_seg]["chan_neighbor_ind"],
                    self.sort_info["n_samples_per_chan"]
                    * (self.work_items[chan][start_seg]["chan_neighbor_ind"] + 1),
                ]
            # Go through each segment as the "current segment" and set the labels
            # in the next segment according to the scheme in current
            for curr_seg in range(start_seg, len(self.sort_data[chan]) - 1):
                if jump_to_end:
                    break
                next_seg = curr_seg + 1
                if len(self.sort_data[chan][curr_seg][1]) == 0:
                    # curr_seg is now failed next seg of last iteration
                    start_new_seg = True
                    if self.verbose:
                        print("skipping seg", curr_seg, "with no spikes")
                    continue
                if start_new_seg:
                    # Update new labels dictionary FOR THIS CHANNEL ONLY!
                    old_n_labels = np.unique(self.sort_data[chan][curr_seg][1])
                    for n_label_key in old_n_labels:
                        update_new2orig_seg_labels[curr_seg].append(
                            [
                                n_label_key + next_real_label,
                                self.new2orig_seg_labels[curr_seg][n_label_key],
                            ]
                        )
                        # self.new2orig_seg_labels[curr_seg][n_label_key + next_real_label] = self.new2orig_seg_labels[curr_seg][n_label_key]
                        # del self.new2orig_seg_labels[curr_seg][n_label_key]
                    # Map all units in this segment to new real labels
                    self.sort_data[chan][curr_seg][
                        1
                    ] += next_real_label  # Keep out of range
                    old_n_labels += next_real_label
                    for nl in old_n_labels:
                        # Add these new labels to real_labels for tracking
                        real_labels.append(nl)
                        if self.verbose:
                            print(
                                "In NEXT SEG NEW (531) added real label",
                                nl,
                                chan,
                                curr_seg,
                            )
                        next_real_label += 1
                if len(self.sort_data[chan][next_seg][1]) == 0:
                    # No units sorted in NEXT segment so start fresh next segment
                    start_new_seg = True
                    if self.verbose:
                        print(
                            "skipping_seg", curr_seg, "because NEXT seg has no spikes"
                        )
                    continue

                # Make 'fake_labels' for next segment that do not overlap with
                # the current segment and make a work space for the next
                # segment labels so we can compare curr and next without
                # losing track of original labels
                fake_labels = np.copy(np.unique(self.sort_data[chan][next_seg][1]))
                fake_labels += (
                    next_real_label  # Keep these out of range of the real labels
                )
                fake_labels = fake_labels.tolist()
                next_label_workspace = np.copy(self.sort_data[chan][next_seg][1])
                next_label_workspace += next_real_label

                # Merge test all mutually closest clusters and track any labels
                # in the next segment (fake_labels) that do not find a match.
                # These are assigned a new real label.
                leftover_labels = [x for x in fake_labels]
                main_labels = [x for x in real_labels]
                previously_compared_pairs = []
                while len(main_labels) > 0 and len(leftover_labels) > 0:
                    # Must be reset each iteration
                    curr_chan_inds = np.arange(main_win[0], main_win[1], dtype=np.int64)
                    (
                        best_pair,
                        best_shift,
                        clips_1,
                        clips_2,
                        curr_chan_inds,
                    ) = self.find_nearest_shifted_pair(
                        chan,
                        curr_seg,
                        next_seg,
                        main_labels,
                        leftover_labels,
                        next_label_workspace,
                        curr_chan_inds,
                        previously_compared_pairs,
                        self.half_clip_inds,
                    )
                    if len(best_pair) == 0:
                        break
                    if clips_1.shape[0] == 1 or clips_2.shape[0] == 1:
                        # Don't mess around with only 1 spike, it is not helpful to merge
                        is_merged = False
                    else:
                        is_merged, _, _ = self.merge_test_two_units(
                            clips_1,
                            clips_2,
                            self.sort_info["p_value_cut_thresh"],
                            method="template_pca",
                            merge_only=True,
                            curr_chan_inds=curr_chan_inds,
                            use_weights=True,
                        )
                    if self.verbose:
                        print(
                            "Item",
                            self.work_items[chan][curr_seg]["ID"],
                            "on chan",
                            chan,
                            "seg",
                            curr_seg,
                            "merged",
                            is_merged,
                            "for labels",
                            best_pair,
                        )

                    if is_merged:
                        # Choose next seg spikes based on original fake label workspace
                        fake_select = next_label_workspace == best_pair[1]
                        # Update new labels dictionary
                        old_label = np.unique(
                            self.sort_data[chan][next_seg][1][fake_select]
                        )
                        # Update actual next segment label data with same labels
                        # used in curr_seg
                        self.sort_data[chan][next_seg][1][fake_select] = best_pair[0]
                        if old_label.size > 1:
                            raise RuntimeError(
                                "Found too many labels to update label dictionary!"
                            )
                        else:
                            old_label = old_label[0]
                        update_new2orig_seg_labels[next_seg].append(
                            [
                                best_pair[0],
                                self.new2orig_seg_labels[next_seg][old_label],
                            ]
                        )
                        leftover_labels.remove(best_pair[1])
                        main_labels.remove(best_pair[0])
                    else:
                        # These mutually closest failed so do not repeat
                        main_labels.remove(best_pair[0])
                        previously_compared_pairs.append(best_pair)

                # Assign units in next segment that do not match any in the
                # current segment a new real label
                for ll in leftover_labels:
                    ll_select = next_label_workspace == ll
                    # Update new labels dictionary
                    old_label = np.unique(self.sort_data[chan][next_seg][1][ll_select])
                    self.sort_data[chan][next_seg][1][ll_select] = next_real_label
                    if old_label.size > 1:
                        raise RuntimeError(
                            "Found too many labels to update label dictionary!"
                        )
                    else:
                        old_label = old_label[0]
                    update_new2orig_seg_labels[next_seg].append(
                        [next_real_label, self.new2orig_seg_labels[next_seg][old_label]]
                    )
                    # self.new2orig_seg_labels[next_seg][next_real_label] = self.new2orig_seg_labels[next_seg][old_label]
                    # del self.new2orig_seg_labels[next_seg][old_label]
                    real_labels.append(next_real_label)
                    if self.verbose:
                        print(
                            "In leftover labels (882) added real label",
                            next_real_label,
                            chan,
                            curr_seg,
                        )
                    next_real_label += 1

                # If we made it here then we are not starting a new seg
                start_new_seg = False
                if self.verbose:
                    print(
                        "!!!REAL LABELS ARE !!!",
                        real_labels,
                        np.unique(self.sort_data[chan][curr_seg][1]),
                        np.unique(self.sort_data[chan][next_seg][1]),
                    )
                if curr_seg > 0:
                    if self.verbose:
                        print(
                            "Previous seg...",
                            np.unique(self.sort_data[chan][curr_seg - 1][1]),
                        )

            # It is possible to leave loop without checking last seg in the
            # event it is a new seg
            if start_new_seg and len(self.sort_data[chan][-1][0]) > 0:
                # Update new labels dictionary FOR THIS CHANNEL ONLY!
                old_n_labels = np.unique(self.sort_data[chan][-1][1])
                for n_label_key in old_n_labels:
                    update_new2orig_seg_labels[-1].append(
                        [
                            n_label_key + next_real_label,
                            self.new2orig_seg_labels[-1][n_label_key],
                        ]
                    )
                    # self.new2orig_seg_labels[-1][n_label_key + next_real_label] = self.new2orig_seg_labels[-1][n_label_key]
                    # del self.new2orig_seg_labels[-1][n_label_key]
                # Map all units in this segment to new real labels
                # Seg labels start at zero, so just add next_real_label. This
                # is last segment for this channel so no need to increment
                self.sort_data[chan][-1][1] += next_real_label
                real_labels.extend((old_n_labels + next_real_label).tolist())
                if self.verbose:
                    print(
                        "Last seg is new (747) added real labels",
                        np.unique(self.sort_data[chan][-1][1]) + next_real_label,
                        chan,
                        curr_seg,
                    )
            if self.verbose:
                print("!!!REAL LABELS ARE !!!", real_labels)
            self.is_stitched = True
        for seg_ind in range(0, len(update_new2orig_seg_labels)):
            for key, value in update_new2orig_seg_labels[seg_ind]:
                self.new2orig_seg_labels[seg_ind][key] = value

    def get_shifted_neighborhood_SSE(
        self,
        neuron1,
        neuron2,
        max_shift_inds,
    ):
        """ """
        # Find the shared part of each unit's neighborhood and number of samples
        neighbor_overlap_bool = np.in1d(neuron1["neighbors"], neuron2["neighbors"])
        overlap_chans = neuron1["neighbors"][neighbor_overlap_bool]
        n_samples_per_chan = self.sort_info["n_samples_per_chan"]

        # Build a template for each unit that has the intersection of
        # neighborhood channels
        overlap_template_1 = np.zeros(
            overlap_chans.shape[0] * n_samples_per_chan, dtype=neuron1["clips"].dtype
        )
        overlap_template_2 = np.zeros(
            overlap_chans.shape[0] * n_samples_per_chan, dtype=neuron2["clips"].dtype
        )
        for n_chan in range(0, overlap_chans.shape[0]):
            n1_neighbor_ind = next(
                (
                    idx[0]
                    for idx, val in np.ndenumerate(neuron1["neighbors"])
                    if val == overlap_chans[n_chan]
                )
            )
            overlap_template_1[
                n_chan * n_samples_per_chan : (n_chan + 1) * n_samples_per_chan
            ] = neuron1["template"][
                n1_neighbor_ind
                * n_samples_per_chan : (n1_neighbor_ind + 1)
                * n_samples_per_chan
            ]
            n2_neighbor_ind = next(
                (
                    idx[0]
                    for idx, val in np.ndenumerate(neuron2["neighbors"])
                    if val == overlap_chans[n_chan]
                )
            )
            overlap_template_2[
                n_chan * n_samples_per_chan : (n_chan + 1) * n_samples_per_chan
            ] = neuron2["template"][
                n2_neighbor_ind
                * n_samples_per_chan : (n2_neighbor_ind + 1)
                * n_samples_per_chan
            ]

        # Align templates based on peak cross correlation
        cross_corr = np.correlate(overlap_template_1, overlap_template_2, mode="full")
        max_corr_ind = np.argmax(cross_corr)
        shift = max_corr_ind - cross_corr.shape[0] // 2
        if np.abs(shift) > max_shift_inds:
            # Best alignment is too far
            return np.inf
        if shift == 0:
            # Compute SSE on clips as is
            # Must normalize distance per data point else reward big shifts
            SSE = (
                np.sum((overlap_template_1 - overlap_template_2) ** 2)
                / overlap_template_1.shape[0]
            )
            return SSE

        # Align and truncate templates to compute SSE
        shift_samples_per_chan = self.sort_info["n_samples_per_chan"] - np.abs(shift)
        shift_template_1 = np.zeros(
            shift_samples_per_chan * overlap_chans.shape[0], dtype=chan_temp_1.dtype
        )
        shift_template_2 = np.zeros(
            shift_samples_per_chan * overlap_chans.shape[0], dtype=chan_temp_2.dtype
        )
        # Get clips for each channel, shift them, and assign for output, which
        # will be clips that have each channel individually aligned and
        # truncated
        for chan_ind in range(0, overlap_chans.shape[0]):
            chan_temp_1 = overlap_template_1[
                chan_ind
                * self.sort_info["n_samples_per_chan"] : (chan_ind + 1)
                * self.sort_info["n_samples_per_chan"]
            ]
            chan_temp_2 = overlap_template_2[
                chan_ind
                * self.sort_info["n_samples_per_chan"] : (chan_ind + 1)
                * self.sort_info["n_samples_per_chan"]
            ]
            if shift > 0:
                shift_template_1[
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan
                ] = chan_temp_1[shift:]
                shift_template_2[
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan
                ] = chan_temp_2[: -1 * shift]
            elif shift < 0:
                shift_template_1[
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan
                ] = chan_temp_1[:shift]
                shift_template_2[
                    chan_ind
                    * shift_samples_per_chan : (chan_ind + 1)
                    * shift_samples_per_chan
                ] = chan_temp_2[-1 * shift :]
        # Must normalize distance per data point else reward big shifts
        SSE = (
            np.sum((shift_template_1 - shift_template_2) ** 2)
            / shift_template_1.shape[0]
        )
        return SSE

    def summarize_neurons_by_seg(self):
        """Make a neuron summary for each unit in each segment and add them to
        a new class attribute 'neuron_summary_by_seg'.

        """
        self.neuron_summary_by_seg = [[] for x in range(0, self.n_segments)]
        for seg in range(0, self.n_segments):
            (
                pair_separability_matrix,
                noise_contamination,
                noise_misses,
            ) = neuron_separability.pairwise_separability(
                self.sort_info["separability_metrics"][seg], self.sort_info
            )
            for chan in range(0, self.n_chans):
                cluster_labels = np.unique(self.sort_data[chan][seg][1])
                for neuron_label in cluster_labels:
                    neuron = {}
                    neuron["summary_type"] = "single_segment"
                    neuron["channel"] = self.work_items[chan][seg]["channel"]
                    neuron["neighbors"] = self.work_items[chan][seg]["neighbors"]
                    neuron["chan_neighbor_ind"] = self.work_items[chan][seg][
                        "chan_neighbor_ind"
                    ]
                    neuron["segment"] = self.work_items[chan][seg]["seg_number"]
                    neuron["label"] = neuron_label
                    neuron["main_win"] = [
                        self.sort_info["n_samples_per_chan"]
                        * neuron["chan_neighbor_ind"],
                        self.sort_info["n_samples_per_chan"]
                        * (neuron["chan_neighbor_ind"] + 1),
                    ]
                    assert neuron["segment"] == seg, "Somethings messed up here?"
                    assert neuron["channel"] == chan

                    select_label = self.sort_data[chan][seg][1] == neuron_label
                    neuron["spike_indices"] = self.sort_data[chan][seg][0][select_label]
                    neuron["binary_pursuit_bool"] = self.sort_data[chan][seg][3][
                        select_label
                    ]

                    # for tracking pairwise separability metrics
                    if self.sort_info["output_separability_metrics"]:
                        original_label = self.new2orig_seg_labels[seg][neuron["label"]]
                        neuron["n_max_confusion"] = np.argmax(
                            pair_separability_matrix[original_label, :]
                        )
                        neuron["max_confusion"] = pair_separability_matrix[
                            original_label, neuron["n_max_confusion"]
                        ]
                        neuron["p_noise_added"] = noise_contamination[original_label]
                        neuron["p_noise_miss"] = noise_misses[original_label]
                        # Find the current label of max confusion
                        for n_label in self.new2orig_seg_labels[seg].keys():
                            if (
                                self.new2orig_seg_labels[seg][n_label]
                                == neuron["n_max_confusion"]
                            ):
                                neuron["n_max_confusion"] = n_label
                                break

                    # NOTE: This still needs to be done even though segments
                    # were ordered because of overlap!
                    # Ensure spike times are ordered. Must use 'stable' sort for
                    # output to be repeatable because overlapping segments and
                    # binary pursuit can return slightly different dupliate spikes
                    spike_order = np.argsort(neuron["spike_indices"], kind="stable")
                    neuron["spike_indices"] = neuron["spike_indices"][spike_order]
                    if self.voltage is None:
                        neuron["clips"] = self.sort_data[chan][seg][2][select_label, :]
                        neuron["clips"] = neuron["clips"][spike_order, :]
                    else:
                        neuron["clips"], _ = get_clips(
                            self.clip_dict,
                            self.voltage,
                            neuron["neighbors"],
                            neuron["spike_indices"],
                            self.sort_info["clip_width"],
                            check_valid=False,
                        )
                    neuron["binary_pursuit_bool"] = neuron["binary_pursuit_bool"][
                        spike_order
                    ]

                    # Set duplicate tolerance as half spike width since within
                    # channel summary shouldn't be off by this
                    neuron["duplicate_tol_inds"] = (
                        analyze_spike_timing.calc_spike_half_width(
                            np.mean(
                                neuron["clips"][
                                    :, neuron["main_win"][0] : neuron["main_win"][1]
                                ],
                                axis=0,
                            )
                        )
                        + 1
                    )

                    # Recompute template and store output
                    neuron["template"] = np.mean(neuron["clips"], axis=0).astype(
                        neuron["clips"].dtype
                    )
                    neuron["snr"] = self.get_snr(chan, seg, neuron["template"])
                    neuron[
                        "fraction_mua"
                    ] = analyze_spike_timing.calc_fraction_mua_to_peak(
                        neuron["spike_indices"],
                        self.sort_info["sampling_rate"],
                        neuron["duplicate_tol_inds"],
                        self.absolute_refractory_period,
                    )
                    neuron["threshold"] = self.work_items[chan][seg]["thresholds"][
                        self.work_items[chan][seg]["channel"]
                    ]
                    neuron["quality_score"] = (
                        neuron["snr"]
                        * (1 - neuron["fraction_mua"])
                        * (neuron["spike_indices"].shape[0])
                    )
                    self.neuron_summary_by_seg[seg].append(neuron)

    def get_overlap_ratio(
        self,
        seg1,
        n1_ind,
        seg2,
        n2_ind,
        overlap_time,
    ):
        """ """
        # Overlap time is the time that their spikes coexist
        n1_spikes = self.neuron_summary_by_seg[seg1][n1_ind]["spike_indices"]
        n2_spikes = self.neuron_summary_by_seg[seg2][n2_ind]["spike_indices"]
        max_samples = int(round(overlap_time * self.sort_info["sampling_rate"]))
        overlap_ratio = analyze_spike_timing.calc_overlap_ratio(
            n1_spikes, n2_spikes, max_samples
        )
        return overlap_ratio

    def any_linked(self, neuron):
        """Check if neuron has at least one link. If there is only 1 segment
        returns True."""
        if (
            (neuron["prev_seg_link"] is None)
            and (neuron["next_seg_link"] is None)
            and (self.n_segments > 1)
        ):
            is_any_linked = False
        else:
            # has at least 1 link
            is_any_linked = True
        return is_any_linked

    def all_linked(self, seg, neuron):
        """Check if the neuron is fully linked on both ends. If there is only
        1 segment returns True."""
        is_all_linked = True
        if self.n_segments == 1:
            return is_all_linked
        if seg == self.n_segments - 1:
            # Last segment can't have next link
            if neuron["prev_seg_link"] is None:
                is_all_linked = False
        elif seg == 0:
            # First segment can't have previous link
            if neuron["next_seg_link"] is None:
                is_all_linked = False
        else:
            # Need to check both links
            if neuron["prev_seg_link"] is None or neuron["next_seg_link"] is None:
                is_all_linked = False
        return is_all_linked

    def remove_redundant_neurons(self, seg, overlap_ratio_threshold=5):
        """
        Note that this function does not actually delete anything. It removes
        links between segments for redundant units and it adds a flag under
        the key 'deleted_as_redundant' to indicate that a segment unit should
        be deleted. Deleting units in this function would ruin the indices used
        to link neurons together later and is not worth the book keeping trouble.
        Note: overlap_ratio_threshold == np.inf will not delete anything while
        overlap_ratio_threshold == -np.inf will delete everything except 1 unit.
        """
        rn_verbose = False
        # Since we are comparing across channels, we need to consider potentially
        # large alignment differences in the overlap_time
        overlap_time = self.half_clip_inds / self.sort_info["sampling_rate"]
        neurons = self.neuron_summary_by_seg[seg]
        overlap_ratio = np.zeros((len(neurons), len(neurons)))
        expected_ratio = np.zeros((len(neurons), len(neurons)))
        delta_ratio = np.zeros((len(neurons), len(neurons)))
        quality_scores = [n["quality_score"] for n in neurons]
        violation_partners = [set() for x in range(0, len(neurons))]
        for neuron1_ind, neuron1 in enumerate(neurons):
            violation_partners[neuron1_ind].add(neuron1_ind)
            # Loop through all pairs of units and compute overlap and expected
            for neuron2_ind in range(neuron1_ind + 1, len(neurons)):
                neuron2 = neurons[neuron2_ind]
                if neuron1["channel"] == neuron2["channel"]:
                    continue  # If they are on the same channel, do nothing
                # if neuron1['channel'] not in neuron2['neighbors']:
                #     continue # If they are not in same neighborhood, do nothing
                # overlap_time = max(neuron1['duplicate_tol_inds'], neuron2['duplicate_tol_inds']) / self.sort_info['sampling_rate']
                exp, act, delta = analyze_spike_timing.calc_ccg_overlap_ratios(
                    neuron1["spike_indices"],
                    neuron2["spike_indices"],
                    overlap_time,
                    self.sort_info["sampling_rate"],
                )
                expected_ratio[neuron1_ind, neuron2_ind] = exp
                expected_ratio[neuron2_ind, neuron1_ind] = expected_ratio[
                    neuron1_ind, neuron2_ind
                ]
                overlap_ratio[neuron1_ind, neuron2_ind] = act
                overlap_ratio[neuron2_ind, neuron1_ind] = overlap_ratio[
                    neuron1_ind, neuron2_ind
                ]
                delta_ratio[neuron1_ind, neuron2_ind] = delta
                delta_ratio[neuron2_ind, neuron1_ind] = delta_ratio[
                    neuron1_ind, neuron2_ind
                ]

                if (
                    overlap_ratio[neuron1_ind, neuron2_ind]
                    >= overlap_ratio_threshold * delta_ratio[neuron1_ind, neuron2_ind]
                    + expected_ratio[neuron1_ind, neuron2_ind]
                ):
                    # Overlap is higher than chance and at least one of these will be removed
                    violation_partners[neuron1_ind].add(neuron2_ind)
                    violation_partners[neuron2_ind].add(neuron1_ind)

        neurons_remaining_indices = [x for x in range(0, len(neurons))]
        max_accepted = 0.0
        max_expected = 0.0
        while True:
            # Look for our next best pair
            best_ratio = -np.inf
            best_pair = []
            for i in range(0, len(neurons_remaining_indices)):
                for j in range(i + 1, len(neurons_remaining_indices)):
                    neuron_1_index = neurons_remaining_indices[i]
                    neuron_2_index = neurons_remaining_indices[j]
                    if (
                        overlap_ratio[neuron_1_index, neuron_2_index]
                        < overlap_ratio_threshold
                        * delta_ratio[neuron_1_index, neuron_2_index]
                        + expected_ratio[neuron_1_index, neuron_2_index]
                    ):
                        # Overlap not high enough to merit deletion of one
                        # But track our proximity to input threshold
                        if overlap_ratio[neuron_1_index, neuron_2_index] > max_accepted:
                            max_accepted = overlap_ratio[neuron_1_index, neuron_2_index]
                            max_expected = (
                                overlap_ratio_threshold
                                * delta_ratio[neuron_1_index, neuron_2_index]
                                + expected_ratio[neuron_1_index, neuron_2_index]
                            )
                        continue
                    if overlap_ratio[neuron_1_index, neuron_2_index] > best_ratio:
                        best_ratio = overlap_ratio[neuron_1_index, neuron_2_index]
                        best_pair = [neuron_1_index, neuron_2_index]
            if len(best_pair) == 0 or best_ratio == 0:
                # No more pairs exceed ratio threshold
                print(
                    "Maximum accepted ratio was",
                    max_accepted,
                    "at expected threshold",
                    max_expected,
                )
                break

            # We now need to choose one of the pair to delete.
            neuron_1 = neurons[best_pair[0]]
            neuron_2 = neurons[best_pair[1]]
            delete_1 = False
            delete_2 = False

            # We will also consider how good each neuron is relative to the
            # other neurons that it overlaps with. Basically, if one unit is
            # a best remaining copy while the other has better units it overlaps
            # with, we want to preferentially keep the best remaining copy
            # This trimming should work because we choose the most overlapping
            # pairs for each iteration
            combined_violations = violation_partners[best_pair[0]].union(
                violation_partners[best_pair[1]]
            )
            max_other_n1 = neuron_1["quality_score"]
            other_n1 = combined_violations - violation_partners[best_pair[1]]
            for v_ind in other_n1:
                if quality_scores[v_ind] > max_other_n1:
                    max_other_n1 = quality_scores[v_ind]
            max_other_n2 = neuron_2["quality_score"]
            other_n2 = combined_violations - violation_partners[best_pair[0]]
            for v_ind in other_n2:
                if quality_scores[v_ind] > max_other_n2:
                    max_other_n2 = quality_scores[v_ind]
            # Rate each unit on the difference between its quality and the
            # quality of its best remaining violation partner
            # NOTE: diff_score = 0 means this unit is the best remaining
            diff_score_1 = max_other_n1 - neuron_1["quality_score"]
            diff_score_2 = max_other_n2 - neuron_2["quality_score"]

            # Check if both or either had a failed MUA calculation
            if np.isnan(neuron_1["fraction_mua"]) and np.isnan(
                neuron_2["fraction_mua"]
            ):
                # MUA calculation was invalid so just use SNR
                if (
                    neuron_1["snr"] * neuron_1["spike_indices"].shape[0]
                    > neuron_2["snr"] * neuron_2["spike_indices"].shape[0]
                ):
                    delete_2 = True
                else:
                    delete_1 = True
            elif np.isnan(neuron_1["fraction_mua"]) or np.isnan(
                neuron_2["fraction_mua"]
            ):
                # MUA calculation was invalid for one unit so pick the other
                if np.isnan(neuron_1["fraction_mua"]):
                    delete_1 = True
                else:
                    delete_2 = True
            elif diff_score_1 > diff_score_2:
                # Neuron 1 has a better copy somewhere so delete it
                delete_1 = True
                delete_2 = False
            elif diff_score_2 > diff_score_1:
                # Neuron 2 has a better copy somewhere so delete it
                delete_1 = False
                delete_2 = True
            # elif self.all_linked(seg, neuron_1) and not self.any_linked(neuron_2):
            #     # Neuron 1 is fully linked while neuron 2 has no links
            #     # so defer to segment stitching and delete neuron 2
            #     delete_2 = True
            # elif self.all_linked(seg, neuron_2) and not self.any_linked(neuron_1):
            #     # Neuron 2 is fully linked while neuron 1 has no links
            #     # so defer to segment stitching and delete neuron 1
            #     delete_1 = True
            else:
                # Both diff scores == 0 and both are linked so we have to pick one
                if diff_score_1 != 0 and diff_score_2 != 0:
                    raise RuntimeError(
                        "DIFF SCORES IN REDUNDANT ARE NOT BOTH EQUAL TO ZERO BUT I THOUGHT THEY SHOULD BE!"
                    )
                # First defer to choosing highest quality score
                if neuron_1["quality_score"] > neuron_2["quality_score"]:
                    delete_1 = False
                    delete_2 = True
                else:
                    delete_1 = True
                    delete_2 = False

                # Check if quality score is primarily driven by number of spikes rather than SNR and MUA
                # Spike number is primarily valuable in the case that one unit
                # is truly a subset of another. If one unit is a mixture, we
                # need to avoid relying on spike count
                if (
                    delete_2
                    and (1 - neuron_2["fraction_mua"]) * neuron_2["snr"]
                    > (1 - neuron_1["fraction_mua"]) * neuron_1["snr"]
                ) or (
                    delete_1
                    and (1 - neuron_1["fraction_mua"]) * neuron_1["snr"]
                    > (1 - neuron_2["fraction_mua"]) * neuron_2["snr"]
                ):
                    if rn_verbose:
                        print("Checking for mixture due to lopsided spike counts")
                    if len(violation_partners[best_pair[0]]) < len(
                        violation_partners[best_pair[1]]
                    ):
                        if rn_verbose:
                            print(
                                "Neuron 1 has fewer violation partners. Set default delete neuron 2."
                            )
                        delete_1 = False
                        delete_2 = True
                    elif len(violation_partners[best_pair[1]]) < len(
                        violation_partners[best_pair[0]]
                    ):
                        if rn_verbose:
                            print(
                                "Neuron 2 has fewer violation partners. Set default delete neuron 1."
                            )
                        delete_1 = True
                        delete_2 = False
                    else:
                        if rn_verbose:
                            print("Both have equal violation partners")
                    # We will now check if one unit appears to be a subset of the other
                    # If these units are truly redundant subsets, then the MUA of
                    # their union will be <= max(mua1, mua2)
                    # If instead one unit is largely a mixture containing the
                    # other, then the MUA of their union should greatly increase
                    # Note that the if statement above typically only passes
                    # in the event that one unit has considerably more spikes or
                    # both units are extremely similar. Because rates can vary,
                    # we do not use peak MUA here but rather the rate based MUA
                    # Need to union with compliment so spikes are not double
                    # counted, which will reduce the rate based MUA
                    neuron_1_compliment = (
                        ~analyze_spike_timing.find_overlapping_spike_bool(
                            neuron_1["spike_indices"],
                            neuron_2["spike_indices"],
                            self.half_clip_inds,
                        )
                    )
                    union_spikes = np.hstack(
                        (
                            neuron_1["spike_indices"][neuron_1_compliment],
                            neuron_2["spike_indices"],
                        )
                    )
                    union_spikes.sort()
                    # union_duplicate_tol = self.half_clip_inds
                    union_duplicate_tol = max(
                        neuron_1["duplicate_tol_inds"], neuron_2["duplicate_tol_inds"]
                    )
                    union_fraction_mua_rate = analyze_spike_timing.calc_fraction_mua(
                        union_spikes,
                        self.sort_info["sampling_rate"],
                        union_duplicate_tol,
                        self.absolute_refractory_period,
                    )
                    # Need to get fraction MUA by rate, rather than peak,
                    # for comparison here
                    fraction_mua_rate_1 = analyze_spike_timing.calc_fraction_mua(
                        neuron_1["spike_indices"],
                        self.sort_info["sampling_rate"],
                        union_duplicate_tol,
                        self.absolute_refractory_period,
                    )
                    fraction_mua_rate_2 = analyze_spike_timing.calc_fraction_mua(
                        neuron_2["spike_indices"],
                        self.sort_info["sampling_rate"],
                        union_duplicate_tol,
                        self.absolute_refractory_period,
                    )
                    # We will decide not to consider spike count if this looks like
                    # one unit could be a large mixture. This usually means that
                    # the union MUA goes up substantially. To accomodate noise,
                    # require that it exceeds both the minimum MUA plus the MUA
                    # expected if the units were totally independent, and the
                    # MUA of either unit alone.
                    if union_fraction_mua_rate > min(
                        fraction_mua_rate_1, fraction_mua_rate_2
                    ) + delta_ratio[
                        best_pair[0], best_pair[1]
                    ] and union_fraction_mua_rate > max(
                        fraction_mua_rate_1, fraction_mua_rate_2
                    ):
                        # This is a red flag that one unit is likely a large mixture
                        # and we should ignore spike count
                        if rn_verbose:
                            print("This flagged as a large mixture")
                        if (1 - neuron_2["fraction_mua"]) * neuron_2["snr"] > (
                            1 - neuron_1["fraction_mua"]
                        ) * neuron_1["snr"]:
                            # Neuron 2 has better MUA and SNR so pick it
                            delete_1 = True
                            delete_2 = False
                        else:
                            # Neuron 1 has better MUA and SNR so pick it
                            delete_1 = False
                            delete_2 = True

            if delete_1:
                if rn_verbose:
                    print(
                        "Choosing from neurons with channels",
                        neuron_1["channel"],
                        neuron_2["channel"],
                        "in segment",
                        seg,
                    )
                if rn_verbose:
                    print(
                        "Deleting neuron 1 with violators",
                        violation_partners[best_pair[0]],
                        "MUA",
                        neuron_1["fraction_mua"],
                        "snr",
                        neuron_1["snr"],
                        "n spikes",
                        neuron_1["spike_indices"].shape[0],
                    )
                if rn_verbose:
                    print(
                        "Keeping neuron 2 with violators",
                        violation_partners[best_pair[1]],
                        "MUA",
                        neuron_2["fraction_mua"],
                        "snr",
                        neuron_2["snr"],
                        "n spikes",
                        neuron_2["spike_indices"].shape[0],
                    )
                neurons_remaining_indices.remove(best_pair[0])
                for vp in violation_partners:
                    vp.discard(best_pair[0])
                if seg > 0:
                    # Reassign any unit in previous segment with a next link
                    # to this unit to None
                    for p_ind, prev_n in enumerate(self.neuron_summary_by_seg[seg - 1]):
                        if prev_n["next_seg_link"] is None:
                            continue
                        if prev_n["next_seg_link"] == best_pair[0]:
                            prev_n["next_seg_link"] = None
                if seg < self.n_segments - 1:
                    # Reassign any unit in next segment with a previous link
                    # to this unit to None
                    for next_n in self.neuron_summary_by_seg[seg + 1]:
                        if next_n["prev_seg_link"] is None:
                            continue
                        if next_n["prev_seg_link"] == best_pair[0]:
                            next_n["prev_seg_link"] = None
                # Assign current neuron not to anything since
                # it is designated as trash fro deletion
                neurons[best_pair[0]]["prev_seg_link"] = None
                neurons[best_pair[0]]["next_seg_link"] = None
                neurons[best_pair[0]]["deleted_as_redundant"] = True
            if delete_2:
                if rn_verbose:
                    print(
                        "Choosing from neurons with channels",
                        neuron_1["channel"],
                        neuron_2["channel"],
                        "in segment",
                        seg,
                    )
                if rn_verbose:
                    print(
                        "Keeping neuron 1 with violators",
                        violation_partners[best_pair[0]],
                        "MUA",
                        neuron_1["fraction_mua"],
                        "snr",
                        neuron_1["snr"],
                        "n spikes",
                        neuron_1["spike_indices"].shape[0],
                    )
                if rn_verbose:
                    print(
                        "Deleting neuron 2 with violators",
                        violation_partners[best_pair[1]],
                        "MUA",
                        neuron_2["fraction_mua"],
                        "snr",
                        neuron_2["snr"],
                        "n spikes",
                        neuron_2["spike_indices"].shape[0],
                    )
                neurons_remaining_indices.remove(best_pair[1])
                for vp in violation_partners:
                    vp.discard(best_pair[1])
                if seg > 0:
                    # Reassign any unit in previous segment with a next link
                    # to this unit to None
                    for p_ind, prev_n in enumerate(self.neuron_summary_by_seg[seg - 1]):
                        if prev_n["next_seg_link"] is None:
                            continue
                        if prev_n["next_seg_link"] == best_pair[1]:
                            prev_n["next_seg_link"] = None
                if seg < self.n_segments - 1:
                    # Reassign any unit in next segment with a previous link
                    # to this unit to None
                    for next_n in self.neuron_summary_by_seg[seg + 1]:
                        if next_n["prev_seg_link"] is None:
                            continue
                        if next_n["prev_seg_link"] == best_pair[1]:
                            next_n["prev_seg_link"] = None
                # Assign current neuron not to anything since
                # it is designated as trash fro deletion
                neurons[best_pair[1]]["prev_seg_link"] = None
                neurons[best_pair[1]]["next_seg_link"] = None
                neurons[best_pair[1]]["deleted_as_redundant"] = True
        return neurons

    def remove_redundant_neurons_by_seg(self, overlap_ratio_threshold=5):
        """Calls remove_redundant_neurons for each segment."""
        if overlap_ratio_threshold >= self.last_overlap_ratio_threshold:
            print(
                "Redundant neurons already removed at threshold >=",
                overlap_ratio_threshold,
                "Further attempts will have no effect.",
            )
            print("Skipping remove_redundant_neurons_by_seg.")
            return
        else:
            self.last_overlap_ratio_threshold = overlap_ratio_threshold
        for seg in range(0, self.n_segments):
            self.neuron_summary_by_seg[seg] = self.remove_redundant_neurons(
                seg, overlap_ratio_threshold
            )

    def make_overlapping_links(self, ol_verbose=False):
        """ """
        # Now looking for overlaps not only between channels, but between segments
        # so use the largest reasonable overlap time window
        overlap_time = self.sort_info["clip_width"][1] - self.sort_info["clip_width"][0]
        for seg in range(0, self.n_segments - 1):
            n1_remaining = [
                x
                for x in range(0, len(self.neuron_summary_by_seg[seg]))
                if self.neuron_summary_by_seg[seg][x]["next_seg_link"] is None
            ]
            bad_n1 = []
            for n_ind in n1_remaining:
                if self.neuron_summary_by_seg[seg][n_ind]["deleted_as_redundant"]:
                    bad_n1.append(n_ind)
            for dn in bad_n1:
                n1_remaining.remove(dn)
            while len(n1_remaining) > 0:
                max_overlap_ratio = -np.inf
                max_overlap_pair = []
                # Choose the n1/n2 pair with most overlap and link if over threshold
                for n1_ind in n1_remaining:
                    n1 = self.neuron_summary_by_seg[seg][n1_ind]
                    if ol_verbose:
                        print(
                            "CHECKING for matches for n1 on channel",
                            n1["channel"],
                            "quality",
                            n1["quality_score"],
                            "n spikes",
                            n1["spike_indices"].shape[0],
                        )
                    for n2_ind, n2 in enumerate(self.neuron_summary_by_seg[seg + 1]):
                        if ol_verbose:
                            print(
                                "AGAINST n2 on channel",
                                n2["channel"],
                                "quality",
                                n2["quality_score"],
                                "n spikes",
                                n2["spike_indices"].shape[0],
                            )
                        # Only choose from neighborhood
                        if n1["channel"] not in n2["neighbors"]:
                            if ol_verbose:
                                print("RESULT: not in neighbors")
                            continue
                        # Only choose from n2 with no previous link and not redundant
                        if (
                            n2["prev_seg_link"] is not None
                            or n2["deleted_as_redundant"]
                        ):
                            if ol_verbose:
                                if n2["prev_seg_link"] is not None:
                                    print("RESULT: n2 has previous link")
                                if n2["deleted_as_redundant"]:
                                    print("RESULT: n2 was deleted as redundant")
                            continue
                        curr_overlap = self.get_overlap_ratio(
                            seg, n1_ind, seg + 1, n2_ind, overlap_time
                        )
                        if curr_overlap < self.min_overlapping_spikes:
                            if ol_verbose:
                                print(
                                    "RESULT: overlap of",
                                    curr_overlap,
                                    "is less than min_overlapping spikes",
                                    self.min_overlapping_spikes,
                                )
                            continue
                        if curr_overlap >= max_overlap_ratio:
                            if ol_verbose:
                                print(
                                    "RESULT: overlap of",
                                    curr_overlap,
                                    "is set to max for linking",
                                )
                            max_overlap_ratio = curr_overlap
                            max_overlap_pair = [n1_ind, n2_ind]
                if max_overlap_ratio > 0:
                    # Found a match that satisfies thresholds so link them
                    if ol_verbose:
                        print(
                            "Maximum overlap is",
                            max_overlap_ratio,
                            "which satisfies thresholds so linking them.",
                        )
                    self.neuron_summary_by_seg[seg][max_overlap_pair[0]][
                        "next_seg_link"
                    ] = max_overlap_pair[1]
                    self.neuron_summary_by_seg[seg + 1][max_overlap_pair[1]][
                        "prev_seg_link"
                    ] = max_overlap_pair[0]
                    n1_remaining.remove(max_overlap_pair[0])
                else:
                    # Maximum overlap is less than minimum required so done
                    if ol_verbose:
                        print(
                            "Maximum overlap is less than minimum required. Finished."
                        )
                    break

    def stitch_neurons(self):
        start_seg = 0
        while start_seg < self.n_segments:
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                start_seg += 1
                continue
            neurons = [[x] for x in self.neuron_summary_by_seg[start_seg]]
            break
        if start_seg == self.n_segments:
            # No segments with data found
            return [{}]
        elif start_seg == self.n_segments - 1:
            # Need at least 2 remaining segments to stitch.
            # With this being the only segment, we are done. Each neuron is a
            # list with data only for the one segment
            neurons = [[n] for n in self.neuron_summary_by_seg[start_seg]]
            return neurons

        for next_seg in range(start_seg + 1, self.n_segments):
            if len(self.neuron_summary_by_seg[next_seg]) == 0:
                # Nothing to link
                continue
            # For each currently known neuron, remove the neuron it connects to
            # (if any) in the next segment and add it to its own list
            # This is done by following the link of the last unit in the list
            next_seg_inds = [
                x for x in range(0, len(self.neuron_summary_by_seg[next_seg]))
            ]
            for n_list in neurons:
                link_index = n_list[-1]["next_seg_link"]
                if link_index is None:
                    continue
                n_list.append(self.neuron_summary_by_seg[next_seg][link_index])
                next_seg_inds.remove(link_index)
            for new_neuron in next_seg_inds:
                # Start a new list in neurons for anything that didn't link above
                neurons.append([self.neuron_summary_by_seg[next_seg][new_neuron]])
        return neurons

    def join_neuron_dicts(self, unit_dicts_list, n_id=None):
        """ """
        if len(unit_dicts_list) == 0:
            print("Joine neuron dicst is returning empty")
            return {}
        combined_neuron = {}
        combined_neuron["sort_info"] = self.sort_info
        combined_neuron["sort_info"][
            "absolute_refractory_period"
        ] = self.absolute_refractory_period
        # Since a neuron can exist over multiple channels, we need to discover
        # all the channels that are present and track which channel data
        # correspond to
        combined_neuron["channel"] = []
        combined_neuron["neighbors"] = {}
        combined_neuron["chan_neighbor_ind"] = {}
        combined_neuron["main_windows"] = {}
        combined_neuron["duplicate_tol_inds"] = 0
        combined_neuron["labels"] = []
        combined_neuron["id"] = n_id
        combined_neuron["max_confusion"] = {}
        combined_neuron["p_noise_added"] = []
        combined_neuron["p_noise_miss"] = []

        chan_align_peak = {}
        n_total_spikes = 0
        n_peak = 0
        max_clip_samples = 0
        for x in unit_dicts_list:
            n_total_spikes += x["spike_indices"].shape[0]
            # Duplicates across segments and channels can be very different, at least
            # up to a full spike width. So choose to use the largest estimated spike
            # width from any of the composite neuron summaries. Input units use
            # half spike width as duplicate tol inds so double it.
            if 2 * x["duplicate_tol_inds"] > combined_neuron["duplicate_tol_inds"]:
                combined_neuron["duplicate_tol_inds"] = 2 * x["duplicate_tol_inds"]
            if x["channel"] not in combined_neuron["channel"]:
                combined_neuron["channel"].append(x["channel"])
                combined_neuron["neighbors"][x["channel"]] = x["neighbors"]
                combined_neuron["chan_neighbor_ind"][x["channel"]] = x[
                    "chan_neighbor_ind"
                ]
                combined_neuron["main_windows"][x["channel"]] = x["main_win"]
                chan_align_peak[x["channel"]] = [0, 0]  # [Peak votes, total]
                if x["clips"].shape[1] > max_clip_samples:
                    max_clip_samples = x["clips"].shape[1]
            if np.amax(x["template"][x["main_win"][0] : x["main_win"][1]]) > np.amin(
                x["template"][x["main_win"][0] : x["main_win"][1]]
            ):
                chan_align_peak[x["channel"]][0] += 1
            chan_align_peak[x["channel"]][1] += 1

            combined_neuron["labels"].append(x["label"])
            # for tracking pairwise separability metrics
            if "max_confusion" in x:
                if x["n_max_confusion"] in combined_neuron["max_confusion"]:
                    combined_neuron["max_confusion"][x["n_max_confusion"]].append(
                        x["max_confusion"]
                    )
                else:
                    combined_neuron["max_confusion"][x["n_max_confusion"]] = [
                        x["max_confusion"]
                    ]
                combined_neuron["p_noise_added"].append(x["p_noise_added"])
                combined_neuron["p_noise_miss"].append(x["p_noise_miss"])

        if len(combined_neuron["p_noise_added"]) > 0:
            combined_neuron["p_noise_added"] = np.nanmean(
                combined_neuron["p_noise_added"]
            )
        if len(combined_neuron["p_noise_miss"]) > 0:
            combined_neuron["p_noise_miss"] = np.nanmean(
                combined_neuron["p_noise_miss"]
            )

        waveform_clip_center = (
            int(
                round(
                    np.abs(
                        self.sort_info["clip_width"][0]
                        * self.sort_info["sampling_rate"]
                    )
                )
            )
            + 1
        )
        channel_selector = []
        indices_by_unit = []
        clips_by_unit = []
        bp_spike_bool_by_unit = []
        threshold_by_unit = []
        segment_by_unit = []
        snr_by_unit = []
        for unit in unit_dicts_list:
            n_unit_events = unit["spike_indices"].shape[0]
            if n_unit_events == 0:
                continue
            # First adjust all spike indices to where they would have been if
            # aligned to the specified peak or valley for units on the same channel
            # NOTE: I can't think of a good way to do this reliably across channels
            # if chan_align_peak[unit['channel']][0] / chan_align_peak[unit['channel']][1] > 0.5:
            #     # Most templates on this channel have greater peak so align peak
            #     shift = np.argmax(unit['template'][unit['main_win'][0]:unit['main_win'][1]]) - waveform_clip_center
            # else:
            #     shift = np.argmin(unit['template'][unit['main_win'][0]:unit['main_win'][1]]) - waveform_clip_center
            # shift = np.argmax(np.abs(unit['template'][unit['main_win'][0]:unit['main_win'][1]])) - waveform_clip_center
            shift = 0

            indices_by_unit.append(unit["spike_indices"] + shift)
            clips_by_unit.append(unit["clips"])
            bp_spike_bool_by_unit.append(unit["binary_pursuit_bool"])
            # Make and append a bunch of book keeping numpy arrays
            channel_selector.append(np.full(n_unit_events, unit["channel"]))
            threshold_by_unit.append(np.full(n_unit_events, unit["threshold"]))
            segment_by_unit.append(np.full(n_unit_events, unit["segment"]))
            # NOTE: redundantly piling this up makes it easy to track and gives
            # a weighted SNR as its final result
            snr_by_unit.append(np.full(n_unit_events, unit["snr"]))

        # Now combine everything into one
        channel_selector = np.hstack(channel_selector)
        threshold_by_unit = np.hstack(threshold_by_unit)
        segment_by_unit = np.hstack(segment_by_unit)
        snr_by_unit = np.hstack(snr_by_unit)
        combined_neuron["spike_indices"] = np.hstack(indices_by_unit)
        # Need to account for fact that different channels can have different
        # neighborhood sizes. So make all clips start from beginning, and
        # remainder zeroed out if it has no data
        combined_neuron["clips"] = np.zeros(
            (combined_neuron["spike_indices"].shape[0], max_clip_samples),
            dtype=clips_by_unit[0].dtype,
        )
        clip_start_ind = 0
        for clips in clips_by_unit:
            combined_neuron["clips"][
                clip_start_ind : clips.shape[0] + clip_start_ind, 0 : clips.shape[1]
            ] = clips
            clip_start_ind += clips.shape[0]
        combined_neuron["binary_pursuit_bool"] = np.hstack(bp_spike_bool_by_unit)

        # Cleanup some memory that wasn't overwritten during concatenation
        del indices_by_unit
        del clips_by_unit
        del bp_spike_bool_by_unit

        # NOTE: This still needs to be done even though segments
        # were ordered because of overlap!
        # Ensure everything is ordered. Must use 'stable' sort for
        # output to be repeatable because overlapping segments and
        # binary pursuit can return slightly different dupliate spikes
        spike_order = np.argsort(combined_neuron["spike_indices"], kind="stable")
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][spike_order]
        combined_neuron["clips"] = combined_neuron["clips"][spike_order, :]
        combined_neuron["binary_pursuit_bool"] = combined_neuron["binary_pursuit_bool"][
            spike_order
        ]
        channel_selector = channel_selector[spike_order]
        threshold_by_unit = threshold_by_unit[spike_order]
        segment_by_unit = segment_by_unit[spike_order]
        snr_by_unit = snr_by_unit[spike_order]

        # Get each spike's channel of origin and the clips on main channel
        combined_neuron["channel_selector"] = {}
        combined_neuron["template"] = {}
        for chan in combined_neuron["channel"]:
            chan_select = channel_selector == chan
            combined_neuron["channel_selector"][chan] = chan_select
            combined_neuron["template"][chan] = np.mean(
                combined_neuron["clips"][chan_select, :], axis=0
            ).astype(combined_neuron["clips"].dtype)

        # Find the main channel with the largest spike width and use that as
        # the duplicate tolerance for this unit. Even with ZCA and across
        # channels binary pursuit event indices should be more accurate than this
        combined_neuron["duplicate_tol_inds"] = 0
        for c in combined_neuron["channel"]:
            c_main_win = combined_neuron["main_windows"][c]
            curr_spike_width = (
                analyze_spike_timing.calc_spike_half_width(
                    np.mean(
                        combined_neuron["clips"][:, c_main_win[0] : c_main_win[1]],
                        axis=0,
                    )
                )
                + 1
            )
            if curr_spike_width > combined_neuron["duplicate_tol_inds"]:
                combined_neuron["duplicate_tol_inds"] = curr_spike_width

        # Remove any identical index duplicates (either from error or
        # from combining overlapping segments), preferentially keeping
        # the waveform most similar to its channel's template
        keep_bool = analyze_spike_timing.remove_spike_event_duplicates_across_chans(
            combined_neuron
        )
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][keep_bool]
        combined_neuron["binary_pursuit_bool"] = combined_neuron["binary_pursuit_bool"][
            keep_bool
        ]
        combined_neuron["clips"] = combined_neuron["clips"][keep_bool, :]
        for chan in combined_neuron["channel"]:
            combined_neuron["channel_selector"][chan] = combined_neuron[
                "channel_selector"
            ][chan][keep_bool]
        channel_selector = channel_selector[keep_bool]
        threshold_by_unit = threshold_by_unit[
            keep_bool
        ]  # NOTE: not currently output...
        segment_by_unit = segment_by_unit[keep_bool]
        snr_by_unit = snr_by_unit[keep_bool]

        # Recompute things of interest like SNR and templates by channel as the
        # average over all data from that channel and store for output
        combined_neuron["template"] = {}
        combined_neuron["snr"] = {}
        chans_to_remove = []
        for chan in combined_neuron["channel"]:
            if snr_by_unit[combined_neuron["channel_selector"][chan]].size == 0:
                # Spikes contributing from this channel have been removed so
                # remove all its data below
                chans_to_remove.append(chan)
            else:
                combined_neuron["template"][chan] = np.mean(
                    combined_neuron["clips"][
                        combined_neuron["channel_selector"][chan], :
                    ],
                    axis=0,
                ).astype(combined_neuron["clips"].dtype)
                combined_neuron["snr"][chan] = np.mean(
                    snr_by_unit[combined_neuron["channel_selector"][chan]]
                )
        for chan_ind in reversed(range(0, len(chans_to_remove))):
            chan_num = chans_to_remove[chan_ind]
            del combined_neuron["channel"][chan_ind]  # A list so use index
            del combined_neuron["neighbors"][chan_num]  # Rest of these are all
            del combined_neuron["chan_neighbor_ind"][chan_num]  # dictionaries so
            del combined_neuron["channel_selector"][chan_num]  #  use value
            del combined_neuron["main_windows"][chan_num]
        # Weighted average SNR over all segments
        combined_neuron["snr"]["average"] = np.mean(snr_by_unit)
        combined_neuron["duration_s"] = (
            combined_neuron["spike_indices"][-1] - combined_neuron["spike_indices"][0]
        ) / (combined_neuron["sort_info"]["sampling_rate"])
        if combined_neuron["duration_s"] == 0:
            combined_neuron["firing_rate"] = 0.0
        else:
            combined_neuron["firing_rate"] = (
                combined_neuron["spike_indices"].shape[0]
                / combined_neuron["duration_s"]
            )
        combined_neuron[
            "fraction_mua"
        ] = analyze_spike_timing.calc_fraction_mua_to_peak(
            combined_neuron["spike_indices"],
            self.sort_info["sampling_rate"],
            combined_neuron["duplicate_tol_inds"],
            self.absolute_refractory_period,
        )

        return combined_neuron

    def finalize_confusion(self, neuron_summary):
        """Updates confusion metrics if output to reflect the indices of the
        final output neurons for pairwise max confusion averages and alters
        the dictionaries from neuron_summary in place."""

        if self.sort_info["output_separability_metrics"]:
            # Update confusioin indices to match summary outputs
            all_new_confusion = []
            for n1 in neuron_summary:
                new_confusion = {}
                for n2 in neuron_summary:
                    if n1 is n2:
                        continue
                    for conf1_label in n1["max_confusion"].keys():
                        for conf2_label in n2["labels"]:
                            if conf1_label == conf2_label:
                                # some early formatting might be messed up sometimes so enforce proper list format here
                                if n2["id"] in new_confusion:
                                    if isinstance(
                                        n1["max_confusion"][conf1_label], list
                                    ):
                                        new_confusion[n2["id"]].extend(
                                            n1["max_confusion"][conf1_label]
                                        )
                                    else:
                                        new_confusion[n2["id"]].append(
                                            n1["max_confusion"][conf1_label]
                                        )
                                else:
                                    new_confusion[n2["id"]] = n1["max_confusion"][
                                        conf1_label
                                    ]
                                    if not isinstance(new_confusion[n2["id"]], list):
                                        new_confusion[n2["id"]] = list(
                                            new_confusion[n2["id"]]
                                        )
                # Now go through and take mean of new confusions
                for nc in new_confusion.keys():
                    try:
                        new_confusion[nc] = np.nanmean(new_confusion[nc])
                    except:
                        print(new_confusion)
                        print(nc)
                        print(n2["id"])
                        print(n1["max_confusion"])
                        raise
                all_new_confusion.append(new_confusion)
            for n, c in zip(neuron_summary, all_new_confusion):
                n["max_confusion"] = c

        return None

    def summarize_neurons_across_channels(
        self,
        overlap_ratio_threshold=5,
        min_segs_per_unit=1,
        remove_clips=False,
        ol_verbose=False,
    ):
        """Creates output neurons list by combining segment-wise neurons across
        segments and across channels based on stitch_segments and then using
        identical spikes found during the overlapping portions of consecutive
        segments. Requires that there be
        overlap between segments, and enough overlap to be useful.
        The overlap time is the window used to consider spikes the same"""
        if self.overlap_indices == 0 and self.n_segments > 1:
            raise RuntimeError(
                "Cannot do across channel summary without overlap between segments"
            )
        elif (
            self.overlap_indices < self.sort_info["sampling_rate"]
            and self.n_segments > 1
        ):
            summary_message = (
                "Summarizing neurons for multiple data segments"
                "with less than 1 second of overlapping data."
            )
            warnings.warn(summary_message, RuntimeWarning, stacklevel=2)
        # if not self.is_stitched and self.n_segments > 1:
        #     summary_message = "Summarizing neurons for multiple data segments " \
        #                         "without first stitching will result in " \
        #                         "duplicate units discontinuous throughout the " \
        #                         "sorting time period. Call 'stitch_segments()' " \
        #                         "first to combine data across time segments."
        #     warnings.warn(summary_message, RuntimeWarning, stacklevel=2)
        # neuron_summary = []
        # for n in work_summary.neuron_summary_by_seg:
        #     neuron_summary.extend(n)
        # return neuron_summary

        # Start with neurons as those in the first segment with data
        start_seg = 0
        while start_seg < self.n_segments:
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                start_seg += 1
                continue
            # All units in first seg previous link to None
            for n in self.neuron_summary_by_seg[start_seg]:
                n["prev_seg_link"] = None
            break

        start_new_seg = False
        for seg in range(start_seg, self.n_segments - 1):
            if len(self.neuron_summary_by_seg[seg]) == 0:
                start_new_seg = True
                continue
            if start_new_seg:
                for n in self.neuron_summary_by_seg[seg]:
                    n["prev_seg_link"] = None
                start_new_seg = False
            # For the current seg, we will discover their next seg link
            for n in self.neuron_summary_by_seg[seg]:
                n["next_seg_link"] = None
                # Take this opportunity to set this to default
                n["deleted_as_redundant"] = False
            next_seg = seg + 1
            if len(self.neuron_summary_by_seg[next_seg]) == 0:
                # Current seg neurons can't link with neurons in next seg
                continue
            # For the next seg, we will discover their previous seg link.
            for n in self.neuron_summary_by_seg[next_seg]:
                n["prev_seg_link"] = None
            # Use stitching labels to link segments within channel
            for prev_ind, n in enumerate(self.neuron_summary_by_seg[seg]):
                for next_ind, next_n in enumerate(self.neuron_summary_by_seg[next_seg]):
                    if (
                        n["label"] == next_n["label"]
                        and n["channel"] == next_n["channel"]
                    ):
                        n["next_seg_link"] = next_ind
                        next_n["prev_seg_link"] = prev_ind

        # All units in last seg next link to None
        for n in self.neuron_summary_by_seg[self.n_segments - 1]:
            n["next_seg_link"] = None
            n["deleted_as_redundant"] = False

        # Remove redundant items across channels
        self.remove_redundant_neurons_by_seg(overlap_ratio_threshold)
        # NOTE: This MUST run AFTER remove redundant by seg or else you can
        # end up linking a redundant mixture to a good unit with broken link!
        if self.is_stitched:
            self.make_overlapping_links(ol_verbose)

        neurons = self.stitch_neurons()
        # Delete any redundant segs. These shouldn't really be in here anyways
        for n_list in neurons:
            inds_to_delete = []
            for n_seg_ind, n_seg in enumerate(n_list):
                if n_seg["deleted_as_redundant"]:
                    inds_to_delete.append(n_seg_ind)
            for d_ind in reversed(inds_to_delete):
                del n_list[d_ind]
        inds_to_delete = []
        for n_ind, n_list in enumerate(neurons):
            if len(n_list) < min_segs_per_unit:
                inds_to_delete.append(n_ind)
        for d_ind in reversed(inds_to_delete):
            del neurons[d_ind]
        # Use the links to join eveyrthing for final output
        neuron_summary = []
        for n_ind, n in enumerate(neurons):
            neuron_summary.append(self.join_neuron_dicts(n, n_id=n_ind))
            # Indicate origin of summary for each neuron
            neuron_summary[-1]["summary_type"] = "across_channels"
        if remove_clips:
            neuron_summary = delete_spike_clips(neuron_summary)
        self.finalize_confusion(neuron_summary)

        return neuron_summary

    def summarize_neurons_within_channel(
        self,
        min_segs_per_unit=1,
        remove_clips=False,
    ):
        """Creates output neurons list by combining segment-wise neurons across
        segments within each channel using stitch_segments. Requires that there
        be overlap between segments, and enough overlap to be useful."""
        if not self.is_stitched and self.n_segments > 1:
            summary_message = (
                "Summarizing neurons for multiple data segments"
                "without first stitching will result in"
                "duplicate units discontinuous throughout the"
                "sorting time period. Call 'stitch_segments()' "
                "first to combine data across time segments."
            )
            warnings.warn(summary_message, RuntimeWarning, stacklevel=2)
            # neuron_summary = []
            # for n in work_summary.neuron_summary_by_seg:
            #     neuron_summary.extend(n)
            # return neuron_summary

        # Start with neurons as those in the first segment with data
        start_seg = 0
        while start_seg < self.n_segments:
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                start_seg += 1
                continue
            # All units in first seg previous link to None
            for n in self.neuron_summary_by_seg[start_seg]:
                n["prev_seg_link"] = None
            break

        start_new_seg = False
        for seg in range(start_seg, self.n_segments - 1):
            if len(self.neuron_summary_by_seg[seg]) == 0:
                start_new_seg = True
                continue
            if start_new_seg:
                for n in self.neuron_summary_by_seg[seg]:
                    n["prev_seg_link"] = None
                start_new_seg = False
            # For the current seg, we will discover their next seg link
            for n in self.neuron_summary_by_seg[seg]:
                n["next_seg_link"] = None
                # Take this opportunity to set this to default
                n["deleted_as_redundant"] = False
            next_seg = seg + 1
            if len(self.neuron_summary_by_seg[next_seg]) == 0:
                # Current seg neurons can't link with neurons in next seg
                continue
            # For the next seg, we will discover their previous seg link.
            for n in self.neuron_summary_by_seg[next_seg]:
                n["prev_seg_link"] = None
            # Use stitching labels to link segments within channel
            for prev_ind, n in enumerate(self.neuron_summary_by_seg[seg]):
                for next_ind, next_n in enumerate(self.neuron_summary_by_seg[next_seg]):
                    if (
                        n["label"] == next_n["label"]
                        and n["channel"] == next_n["channel"]
                    ):
                        n["next_seg_link"] = next_ind
                        next_n["prev_seg_link"] = prev_ind

        # All units in last seg next link to None
        for n in self.neuron_summary_by_seg[self.n_segments - 1]:
            n["next_seg_link"] = None
            n["deleted_as_redundant"] = False

        neurons = self.stitch_neurons()
        # Delete any redundant segs. These shouldn't really be in here anyways
        for n_list in neurons:
            inds_to_delete = []
            for n_seg_ind, n_seg in enumerate(n_list):
                if n_seg["deleted_as_redundant"]:
                    inds_to_delete.append(n_seg_ind)
            for d_ind in reversed(inds_to_delete):
                del n_list[d_ind]
        inds_to_delete = []
        for n_ind, n_list in enumerate(neurons):
            if len(n_list) < min_segs_per_unit:
                inds_to_delete.append(n_ind)
        for d_ind in reversed(inds_to_delete):
            del neurons[d_ind]
        # Use the links to join eveyrthing for final output
        neuron_summary = []
        for n in neurons:
            neuron_summary.append(self.join_neuron_dicts(n))
            # Indicate origin of summary for each neuron
            neuron_summary[-1]["summary_type"] = "within_channel"
        if remove_clips:
            neuron_summary = delete_spike_clips(neuron_summary)
        return neuron_summary

    def remove_redundant_within_channel_summaries(
        self,
        neurons,
        overlap_ratio_threshold=5,
    ):
        """
        Removes redundant complete summaries as in 'remove_redundant_neurons' but
        for summaries created by 'summarize_neurons_within_channel'.
        """
        # Since we are comparing across channels, we need to consider potentially
        # large alignment differences in the overlap_time
        overlap_time = self.half_clip_inds / self.sort_info["sampling_rate"]
        overlap_ratio = np.zeros((len(neurons), len(neurons)))
        expected_ratio = np.zeros((len(neurons), len(neurons)))
        delta_ratio = np.zeros((len(neurons), len(neurons)))
        quality_scores = np.zeros(len(neurons))
        violation_partners = [set() for x in range(0, len(neurons))]
        for neuron1_ind, neuron1 in enumerate(neurons):
            violation_partners[neuron1_ind].add(neuron1_ind)
            # Initialize some items to neuron dictionary that we will use then remove
            neuron1["deleted_as_redundant"] = False
            neuron1["quality_score"] = (
                neuron1["snr"][neuron1["channel"][0]]
                * (1 - neuron1["fraction_mua"])
                * (neuron1["spike_indices"].shape[0])
            )
            # Loop through all pairs of units and compute overlap and expected
            for neuron2_ind in range(neuron1_ind + 1, len(neurons)):
                neuron2 = neurons[neuron2_ind]
                if neuron1["channel"] == neuron2["channel"]:
                    continue  # If they are on the same channel, do nothing
                # if neuron1['channel'] not in neuron2['neighbors']:
                #     continue # If they are not in same neighborhood, do nothing
                # overlap_time = max(neuron1['duplicate_tol_inds'], neuron2['duplicate_tol_inds']) / self.sort_info['sampling_rate']
                exp, act, delta = analyze_spike_timing.calc_ccg_overlap_ratios(
                    neuron1["spike_indices"],
                    neuron2["spike_indices"],
                    overlap_time,
                    self.sort_info["sampling_rate"],
                )
                expected_ratio[neuron1_ind, neuron2_ind] = exp
                expected_ratio[neuron2_ind, neuron1_ind] = expected_ratio[
                    neuron1_ind, neuron2_ind
                ]
                overlap_ratio[neuron1_ind, neuron2_ind] = act
                overlap_ratio[neuron2_ind, neuron1_ind] = overlap_ratio[
                    neuron1_ind, neuron2_ind
                ]
                delta_ratio[neuron1_ind, neuron2_ind] = delta
                delta_ratio[neuron2_ind, neuron1_ind] = delta_ratio[
                    neuron1_ind, neuron2_ind
                ]

                if (
                    overlap_ratio[neuron1_ind, neuron2_ind]
                    >= overlap_ratio_threshold * delta_ratio[neuron1_ind, neuron2_ind]
                    + expected_ratio[neuron1_ind, neuron2_ind]
                ):
                    # Overlap is higher than chance and at least one of these will be removed
                    violation_partners[neuron1_ind].add(neuron2_ind)
                    violation_partners[neuron2_ind].add(neuron1_ind)

        neurons_remaining_indices = [x for x in range(0, len(neurons))]
        max_accepted = 0.0
        max_expected = 0.0
        while True:
            # Look for our next best pair
            best_ratio = -np.inf
            best_pair = []
            for i in range(0, len(neurons_remaining_indices)):
                for j in range(i + 1, len(neurons_remaining_indices)):
                    neuron_1_index = neurons_remaining_indices[i]
                    neuron_2_index = neurons_remaining_indices[j]
                    if (
                        overlap_ratio[neuron_1_index, neuron_2_index]
                        < overlap_ratio_threshold
                        * delta_ratio[neuron_1_index, neuron_2_index]
                        + expected_ratio[neuron_1_index, neuron_2_index]
                    ):
                        # Overlap not high enough to merit deletion of one
                        # But track our proximity to input threshold
                        if overlap_ratio[neuron_1_index, neuron_2_index] > max_accepted:
                            max_accepted = overlap_ratio[neuron_1_index, neuron_2_index]
                            max_expected = (
                                overlap_ratio_threshold
                                * delta_ratio[neuron_1_index, neuron_2_index]
                                + expected_ratio[neuron_1_index, neuron_2_index]
                            )
                        continue
                    if overlap_ratio[neuron_1_index, neuron_2_index] > best_ratio:
                        best_ratio = overlap_ratio[neuron_1_index, neuron_2_index]
                        best_pair = [neuron_1_index, neuron_2_index]
            if len(best_pair) == 0 or best_ratio == 0:
                # No more pairs exceed ratio threshold
                print(
                    "Maximum accepted ratio was",
                    max_accepted,
                    "at expected threshold",
                    max_expected,
                )
                break

            # We now need to choose one of the pair to delete.
            neuron_1 = neurons[best_pair[0]]
            neuron_2 = neurons[best_pair[1]]
            # Make SNR indexing simpler since these are all single channel
            chan1 = neuron_1["channel"][0]
            chan2 = neuron_2["channel"][0]
            delete_1 = False
            delete_2 = False

            # We will also consider how good each neuron is relative to the
            # other neurons that it overlaps with. Basically, if one unit is
            # a best remaining copy while the other has better units it overlaps
            # with, we want to preferentially keep the best remaining copy
            # This trimming should work because we choose the most overlapping
            # pairs for each iteration
            combined_violations = violation_partners[best_pair[0]].union(
                violation_partners[best_pair[1]]
            )
            max_other_n1 = neuron_1["quality_score"]
            other_n1 = combined_violations - violation_partners[best_pair[1]]
            for v_ind in other_n1:
                if quality_scores[v_ind] > max_other_n1:
                    max_other_n1 = quality_scores[v_ind]
            max_other_n2 = neuron_2["quality_score"]
            other_n2 = combined_violations - violation_partners[best_pair[0]]
            for v_ind in other_n2:
                if quality_scores[v_ind] > max_other_n2:
                    max_other_n2 = quality_scores[v_ind]
            # Rate each unit on the difference between its quality and the
            # quality of its best remaining violation partner
            # NOTE: diff_score = 0 means this unit is the best remaining
            diff_score_1 = max_other_n1 - neuron_1["quality_score"]
            diff_score_2 = max_other_n2 - neuron_2["quality_score"]

            # Check if both or either had a failed MUA calculation
            if np.isnan(neuron_1["fraction_mua"]) and np.isnan(
                neuron_2["fraction_mua"]
            ):
                # MUA calculation was invalid so just use SNR
                if (
                    neuron_1["snr"][chan1] * neuron_1["spike_indices"].shape[0]
                    > neuron_2["snr"][chan2] * neuron_2["spike_indices"].shape[0]
                ):
                    delete_2 = True
                else:
                    delete_1 = True
            elif np.isnan(neuron_1["fraction_mua"]) or np.isnan(
                neuron_2["fraction_mua"]
            ):
                # MUA calculation was invalid for one unit so pick the other
                if np.isnan(neuron_1["fraction_mua"]):
                    delete_1 = True
                else:
                    delete_2 = True
            elif diff_score_1 > diff_score_2:
                # Neuron 1 has a better copy somewhere so delete it
                delete_1 = True
                delete_2 = False
            elif diff_score_2 > diff_score_1:
                # Neuron 2 has a better copy somewhere so delete it
                delete_1 = False
                delete_2 = True
            else:
                # Both diff scores == 0 so we have to pick one
                if diff_score_1 != 0 and diff_score_2 != 0:
                    raise RuntimeError(
                        "DIFF SCORES IN REDUNDANT ARE NOT BOTH EQUAL TO ZERO BUT I THOUGHT THEY SHOULD BE!"
                    )
                # First defer to choosing highest quality score
                if neuron_1["quality_score"] > neuron_2["quality_score"]:
                    delete_2 = True
                else:
                    delete_1 = True

                # Check if quality score is primarily driven by number of spikes rather than SNR and MUA
                # Spike number is primarily valuable in the case that one unit
                # is truly a subset of another. If one unit is a mixture, we
                # need to avoid relying on spike count
                if (
                    delete_2
                    and (1 - neuron_2["fraction_mua"]) * neuron_2["snr"][chan2]
                    > (1 - neuron_1["fraction_mua"]) * neuron_1["snr"][chan1]
                ) or (
                    delete_1
                    and (1 - neuron_1["fraction_mua"]) * neuron_1["snr"][chan1]
                    > (1 - neuron_2["fraction_mua"]) * neuron_2["snr"][chan2]
                ):
                    # We will now check if one unit appears to be a subset of the other
                    # If these units are truly redundant subsets, then the MUA of
                    # their union will be <= max(mua1, mua2)
                    # If instead one unit is largely a mixture containing the
                    # other, then the MUA of their union should greatly increase
                    # Note that the if statement above typically only passes
                    # in the event that one unit has considerably more spikes or
                    # both units are extremely similar. Because rates can vary,
                    # we do not use peak MUA here but rather the rate based MUA
                    # Need to union with compliment so spikes are not double
                    # counted, which will reduce the rate based MUA
                    neuron_1_compliment = (
                        ~analyze_spike_timing.find_overlapping_spike_bool(
                            neuron_1["spike_indices"],
                            neuron_2["spike_indices"],
                            self.half_clip_inds,
                        )
                    )
                    union_spikes = np.hstack(
                        (
                            neuron_1["spike_indices"][neuron_1_compliment],
                            neuron_2["spike_indices"],
                        )
                    )
                    union_spikes.sort()
                    union_fraction_mua_rate = analyze_spike_timing.calc_fraction_mua(
                        union_spikes,
                        self.sort_info["sampling_rate"],
                        self.half_clip_inds,
                        self.absolute_refractory_period,
                    )
                    # Need to get fraction MUA by rate, rather than peak,
                    # for comparison here
                    fraction_mua_rate_1 = analyze_spike_timing.calc_fraction_mua(
                        neuron_1["spike_indices"],
                        self.sort_info["sampling_rate"],
                        self.half_clip_inds,
                        self.absolute_refractory_period,
                    )
                    fraction_mua_rate_2 = analyze_spike_timing.calc_fraction_mua(
                        neuron_2["spike_indices"],
                        self.sort_info["sampling_rate"],
                        self.half_clip_inds,
                        self.absolute_refractory_period,
                    )
                    # We will decide not to consider spike count if this looks like
                    # one unit could be a large mixture. This usually means that
                    # the union MUA goes up substantially. To accomodate noise,
                    # require that it exceeds both the minimum MUA plus the MUA
                    # expected if the units were totally independent, and the
                    # MUA of either unit alone.
                    if union_fraction_mua_rate > min(
                        fraction_mua_rate_1, fraction_mua_rate_2
                    ) + delta_ratio[
                        best_pair[0], best_pair[1]
                    ] and union_fraction_mua_rate > max(
                        fraction_mua_rate_1, fraction_mua_rate_2
                    ):
                        # This is a red flag that one unit is likely a large mixture
                        # and we should ignore spike count
                        if (1 - neuron_2["fraction_mua"]) * neuron_2["snr"][chan2] > (
                            1 - neuron_1["fraction_mua"]
                        ) * neuron_1["snr"][chan1]:
                            # Neuron 2 has better MUA and SNR so pick it
                            delete_1 = True
                            delete_2 = False
                        else:
                            # Neuron 1 has better MUA and SNR so pick it
                            delete_1 = False
                            delete_2 = True

            if delete_1:
                neurons_remaining_indices.remove(best_pair[0])
                for vp in violation_partners:
                    vp.discard(best_pair[0])
                # Do not delete this yet since we want to keep remaining indices order
                neurons[best_pair[0]]["deleted_as_redundant"] = True
            if delete_2:
                neurons_remaining_indices.remove(best_pair[1])
                for vp in violation_partners:
                    vp.discard(best_pair[1])
                # Do not delete this yet since we want to keep remaining indices order
                neurons[best_pair[1]]["deleted_as_redundant"] = True

        for check_n in reversed(range(0, len(neurons))):
            if neurons[check_n]["deleted_as_redundant"]:
                del neurons[check_n]
            else:
                del neurons[check_n]["deleted_as_redundant"]
                del neurons[check_n]["quality_score"]
        return neurons
