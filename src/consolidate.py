import numpy as np
from spikesorting_python.src import segment
from spikesorting_python.src.sort import merge_clusters
from spikesorting_python.src import preprocessing
from spikesorting_python.src.c_cython import sort_cython
from scipy import stats
from scipy.optimize import nnls, lsq_linear
import copy
import warnings


import matplotlib.pyplot as plt



def zero_symmetric_ccg(spikes_1, spikes_2, samples_window=40, d_samples=40, return_trains=False):
    """ Returns the cross correlogram of
        spikes_2 relative to spikes_1 at each lag from -samples_window to
        +samples_window in steps of d_samples.  Bins of the CCG are centered
        at these steps, and so will include a bin centered on lag zero if d_samples
        divides evently into samples_window.

        This is a more basic version of other CCG functions in that it is made
        to quickly compute a simple, zero-centered CCG with integer window
        value and integer step size where the window and step are in UNITS OF
        SAMPLES!  Remember that CCG is computed over a BINARY spike train,
        so using excessively large bins will actually cause the overlaps
        to decrease. """
    samples_window = int(samples_window)
    d_samples = int(d_samples)
    samples_axis = np.arange(-1 * samples_window, samples_window+d_samples, d_samples).astype(np.int64)
    counts = np.zeros(samples_axis.size, dtype=np.int64)

    # Convert the spike indices to units of d_samples
    spikes_1 = (np.floor((spikes_1 + d_samples/2) / d_samples)).astype(np.int64)
    spikes_2 = (np.floor((spikes_2 + d_samples/2) / d_samples)).astype(np.int64)
    samples_axis = (np.floor((samples_axis + d_samples/2) / d_samples)).astype(np.int64)

    # Convert to spike trains
    train_len = int(max(spikes_1[-1], spikes_2[-1])) + 1 # This is samples, so need to add 1
    spike_trains_1 = np.zeros(train_len, dtype=np.bool)
    spike_trains_1[spikes_1] = True
    spike_trains_2 = np.zeros(train_len, dtype=np.bool)
    spike_trains_2[spikes_2] = True

    for lag_ind, lag in enumerate(samples_axis):
        # Construct a timeseries for neuron 2 based on the current spike index as the
        # zero point
        if lag > 0:
            counts[lag_ind] = np.count_nonzero(np.logical_and(spike_trains_1[0:-1*lag],
                                            spike_trains_2[lag:]))
        elif lag < 0:
            counts[lag_ind] = np.count_nonzero(np.logical_and(spike_trains_2[0:lag],
                                            spike_trains_1[-1*lag:]))
        else:
            counts[lag_ind] = np.count_nonzero(np.logical_and(spike_trains_1, spike_trains_2))

    if return_trains:
        return counts, samples_axis, spike_trains_1, spike_trains_2
    else:
        return counts, samples_axis


def find_overlapping_spike_bool(spikes1, spikes2, max_samples=20, except_equal=False):
    """ Finds an index into spikes2 that indicates whether a spike index of spike2
        occurs within +/- max_samples of a spike index in spikes1.  Spikes1 and 2
        are numpy arrays of spike indices in units of samples. Input spikes1 and
        spikes2 will be SORTED IN PLACE because this function won't work if they
        are not ordered. If spikes1 == spikes2 and except_equal is True, the first
        spike in the pair of overlapping spikes is flagged as True in the output
        overlapping_spike_bool. """
    max_samples = np.ceil(max_samples).astype(np.int64)
    overlapping_spike_bool = np.zeros(spikes2.size, dtype=np.bool)
    spike_1_index = 0
    n_spike_1 = 0
    for spike_2_index in range(0, spikes2.size):
        for spike_1_index in range(n_spike_1, spikes1.size):
            if except_equal and spikes1[spike_1_index] == spikes2[spike_2_index]:
                continue
            if ((spikes1[spike_1_index] >= spikes2[spike_2_index] - max_samples) and
               (spikes1[spike_1_index] <= spikes2[spike_2_index] + max_samples)):
                overlapping_spike_bool[spike_2_index] = True
                break
            elif spikes1[spike_1_index] > spikes2[spike_2_index] + max_samples:
                break
        n_spike_1 = spike_1_index

    return overlapping_spike_bool


def keep_binary_pursuit_duplicates(event_indices, new_spike_bool, tol_inds):
    """ Preferentially KEEPS spikes found in binary pursuit.
    """
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            if new_spike_bool[curr_index] and ~new_spike_bool[next_index]:
                keep_bool[next_index] = False
                # Move next index, keep current index
                next_index += 1
            elif ~new_spike_bool[curr_index] and new_spike_bool[next_index]:
                keep_bool[curr_index] = False
                # Move current index to next, move next index
                curr_index = next_index
                next_index += 1
            else:
                # Two spikes with same index, neither from binary pursuit.
                #  Should be chosen based on templates or some other means.
                # Move both indices to next pair
                curr_index = next_index
                next_index += 1
        else:
            # Move both indices to next pair
            curr_index = next_index
            next_index += 1

    return keep_bool


def remove_binary_pursuit_duplicates(event_indices, clips, unit_template,
                                     new_spike_bool, tol_inds):
    """ Preferentially removes overlapping spikes that were both found by binary
    pursuit. This can remove double dipping artifacts as binary pursuit attempts
    to minimize residual error. This only applies for data sorted within the
    same segment and belonging to the same unit, so inputs should reflect that. """
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    temp_sse = np.zeros(2)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            # Violate window
            if new_spike_bool[curr_index] and new_spike_bool[next_index]:
                # AND both found by binary pursuit
                temp_sse[0] = np.sum((clips[curr_index, :] - unit_template) ** 2)
                temp_sse[1] = np.sum((clips[next_index, :] - unit_template) ** 2)
                if temp_sse[0] <= temp_sse[1]:
                    # current spike is better or equal
                    keep_bool[next_index] = False
                    # Move next index, keep current index
                    next_index += 1
                else:
                    # next spike is better
                    keep_bool[curr_index] = False
                    # Move current index to next, move next index
                    curr_index = next_index
                    next_index += 1
            elif new_spike_bool[curr_index] and ~new_spike_bool[next_index]:
                # Move next index, keep current index
                next_index += 1
            else:
                # Move current index to next, move next index
                curr_index = next_index
                next_index += 1
        else:
            # Move both indices to next pair
            curr_index = next_index
            next_index += 1

    return keep_bool


def remove_spike_event_duplicates(event_indices, clips, unit_template, tol_inds):
    """
    """
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    temp_sse = np.zeros(2)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= tol_inds:
            temp_sse[0] = np.sum((clips[curr_index, :] - unit_template) ** 2)
            temp_sse[1] = np.sum((clips[next_index, :] - unit_template) ** 2)
            if temp_sse[0] <= temp_sse[1]:
                # current spike is better or equal
                keep_bool[next_index] = False
                # Move next index, keep current index
                next_index += 1
            else:
                # next spike is better
                keep_bool[curr_index] = False
                # Move current index to next, move next index
                curr_index = next_index
                next_index += 1
        else:
            # Move both indices to next pair
            curr_index = next_index
            next_index += 1

    return keep_bool


def remove_spike_event_duplicates_across_chans(combined_neuron):
    """
    """
    event_indices = combined_neuron['spike_indices']
    keep_bool = np.ones(event_indices.size, dtype=np.bool)
    temp_sse = np.zeros(2)
    curr_index = 0
    next_index = 1
    while next_index < event_indices.size:
        if event_indices[next_index] - event_indices[curr_index] <= combined_neuron['duplicate_tol_inds']:
            # Compare each spike to the template for its own channel and choose
            # the one that matches its own assigned template/channel best
            curr_chan, next_chan = None, None
            for chan in combined_neuron['channel_selector'].keys():
                if combined_neuron['channel_selector'][chan][curr_index]:
                    curr_chan = chan
                if combined_neuron['channel_selector'][chan][next_index]:
                    next_chan = chan
            temp_sse[0] = np.sum((combined_neuron['waveforms'][curr_index, :] - combined_neuron['template'][curr_chan]) ** 2)
            temp_sse[1] = np.sum((combined_neuron['waveforms'][next_index, :] - combined_neuron['template'][next_chan]) ** 2)
            if temp_sse[0] <= temp_sse[1]:
                # current spike is better or equal
                keep_bool[next_index] = False
                next_index += 1
            else:
                # next spike is better
                keep_bool[curr_index] = False
                curr_index = next_index
                next_index += 1
        else:
            curr_index = next_index
            next_index += 1

    return keep_bool


def compute_spike_trains(spike_indices_list, bin_width_samples, min_max_samples):

    if type(spike_indices_list) is not list:
        spike_indices_list = [spike_indices_list]
    if type(min_max_samples) is not list:
        min_max_samples = [0, min_max_samples]
    bin_width_samples = int(bin_width_samples)
    train_len = int(np.ceil((min_max_samples[1] - min_max_samples[0] + bin_width_samples/2) / bin_width_samples) + 1) # Add one because it's a time/samples slice
    spike_trains_list = [[] for x in range(0, len(spike_indices_list))]

    for ind, unit in enumerate(spike_indices_list):
        unit_select = np.logical_and(unit >= min_max_samples[0], unit <= min_max_samples[1])
        # Convert the spike indices to units of bin_width_samples
        spikes = (np.floor((unit[unit_select] + bin_width_samples/2 - min_max_samples[0]) / bin_width_samples)).astype(np.int64)
        spike_trains_list[ind] = np.zeros(train_len, dtype=np.bool)
        spike_trains_list[ind][spikes] = True

    return spike_trains_list if len(spike_trains_list) > 1 else spike_trains_list[0]


def calculate_expected_overlap(n1, n2, overlap_time, sampling_rate):
    """ Returns the expected number of overlapping spikes between neuron 1 and
    neuron 2 within a time window 'overlap_time' assuming independent spiking.
    As usual, spike_indices within each neuron must be sorted. """
    first_index = max(n1['spike_indices'][0], n2['spike_indices'][0])
    last_index = min(n1['spike_indices'][-1], n2['spike_indices'][-1])
    num_ms = int(np.ceil((last_index - first_index) / (sampling_rate / 1000)))
    if num_ms <= 0:
        # Neurons never fire at the same time, so expected overlap is 0
        return 0.

    # Find spike indices from each neuron that fall within the same time window
    # Should be impossible to return None because of num_ms check above
    n1_start = next((idx[0] for idx, val in np.ndenumerate(n1['spike_indices']) if val >= first_index), None)
    n1_stop = next((idx[0] for idx, val in np.ndenumerate(n1['spike_indices'][::-1]) if val <= last_index), None)
    n1_stop = n1['spike_indices'].shape[0] - n1_stop - 1 # Not used as slice so -1
    n2_start = next((idx[0] for idx, val in np.ndenumerate(n2['spike_indices']) if val >= first_index), None)
    n2_stop = next((idx[0] for idx, val in np.ndenumerate(n2['spike_indices'][::-1]) if val <= last_index), None)
    n2_stop = n2['spike_indices'].shape[0] - n2_stop - 1 # Not used as slice so -1

    # Infer number of spikes in overlapping window based on first and last index
    n1_count = n1_stop - n1_start
    n2_count = n2_stop - n2_start
    expected_overlap = (overlap_time * 1000 * n1_count * n2_count) / num_ms

    return expected_overlap


def calc_spike_half_width(clips, clip_width, sampling_rate):
    """ Computes half width of spike as number of indices between peak and valley. """
    template = np.mean(clips, axis=0)
    peak_ind = np.argmax(template)
    valley_ind = np.argmin(template)
    if peak_ind >= valley_ind:
        # peak is after valley
        spike_width = peak_ind - valley_ind
    else:
        spike_width = valley_ind - peak_ind

    return spike_width


def calc_fraction_mua_to_peak(spike_indices, sampling_rate, duplicate_tol_inds,
                         absolute_refractory_period, check_window=0.5):
    """ Estimates the fraction of multiunit activity/noise contained in the
    input spike indices relative to the peak firing rate.

    The ISI distribution is created by binning spike indices from duplicate
    tolerance indices to the end of check window with a bin width. The ISI
    violations are computed as the number of events between duplicate tolerance
    and the absolute refractory period. The returned fraction MUA is this
    number divided by the maximum ISI bin count. """
    all_isis = np.diff(spike_indices)
    refractory_inds = int(round(absolute_refractory_period * sampling_rate))
    bin_width = refractory_inds - duplicate_tol_inds
    if bin_width <= 0:
        print("duplicate_tol_inds encompasses absolute_refractory_period so fraction MUA cannot be computed.")
        return -1.
    check_inds = int(round(check_window * sampling_rate))
    bin_edges = np.arange(duplicate_tol_inds+1, check_inds + bin_width, bin_width)
    counts, xval = np.histogram(all_isis, bin_edges)
    isi_peak = np.amax(counts)
    num_isi_violations = counts[0]
    if num_isi_violations < 0:
        num_isi_violations = 0
    if isi_peak == 0:
        isi_peak = max(num_isi_violations, 1.)
    fraction_mua_to_peak = num_isi_violations / isi_peak

    return fraction_mua_to_peak


def calc_isi_violation_rate(spike_indices, sampling_rate,
        absolute_refractory_period, duplicate_tol_inds):
    """ Estimates the fraction of multiunit activity/noise contained in the
    input spike indices relative to the average firing rate.

    The ISI violations are counted in the window from duplicate tolerance
    indices to the absolute refractory period. The units firing rate in this
    window is then divided by the average unit firing rate to compute the
    returned ISI violation firing rate. """
    all_isis = np.diff(spike_indices)
    duplicate_time = duplicate_tol_inds / sampling_rate
    if absolute_refractory_period - duplicate_time <= 0:
        print("duplicate_tol_inds encompasses absolute_refractory_period so fraction MUA cannot be computed.")
        return -1.
    num_isi_violations = np.count_nonzero(all_isis / sampling_rate < absolute_refractory_period)
    n_duplicates = np.count_nonzero(all_isis <= duplicate_tol_inds)
    # Remove duplicate spikes from this computation and adjust the number
    # of spikes and time window accordingly
    num_isi_violations -= n_duplicates
    isi_violation_rate = num_isi_violations \
                         * (1.0 / (absolute_refractory_period - duplicate_time))\
                         / (spike_indices.shape[0] - n_duplicates)
    return isi_violation_rate


def mean_firing_rate(spike_indices, sampling_rate):
    """ Compute mean firing rate for input spike indices. Spike indices must be
    in sorted order for this to work. """
    if spike_indices[0] == spike_indices[-1]:
        return 0. # Only one spike (could be repeated)
    mean_rate = sampling_rate * spike_indices.size / (spike_indices[-1] - spike_indices[0])
    return mean_rate


def calc_fraction_mua(spike_indices, sampling_rate, duplicate_tol_inds,
                 absolute_refractory_period):
    """ Estimate the fraction of noise/multi-unit activity (MUA) using analysis
    of ISI violations. We do this by looking at the spiking activity that
    occurs during the ISI violation period. """
    isi_violation_rate = calc_isi_violation_rate(spike_indices, sampling_rate,
                                absolute_refractory_period, duplicate_tol_inds)
    mean_rate = mean_firing_rate(spike_indices, sampling_rate)
    if mean_rate == 0.:
        return 0.
    else:
        return (isi_violation_rate / mean_rate)


class WorkItemSummary(object):
    """ Main class that handles and organizes output of spike_sort function.

    duplicate_tol_inds
        NOTE: this will be added to half spike width, which is roughly the max
        spike sorter shift error
    """
    def __init__(self, sort_data, work_items, sort_info,
                 duplicate_tol_inds=1, absolute_refractory_period=10e-4,
                 max_mua_ratio=0.05, min_snr=1.5, n_max_merge_test_clips=None,
                 merge_test_overlap_indices=None, min_overlapping_spikes=.5,
                 binary_pursuit_skip_time='clip_width', m_overlap_k_neighbors=50,
                 verbose=False):
        self.check_input_data(sort_data, work_items)
        self.sort_info = sort_info
        # These are used so frequently they are aliased for convenience
        self.n_chans = self.sort_info['n_channels']
        self.n_segments = self.sort_info['n_segments']
        self.duplicate_tol_inds = duplicate_tol_inds # Added to spike half width for duplicates
        self.absolute_refractory_period = absolute_refractory_period
        self.max_mua_ratio = max_mua_ratio
        self.min_snr = min_snr
        self.n_max_merge_test_clips = n_max_merge_test_clips
        if n_max_merge_test_clips is None:
            # Setting to np.inf uses all clips in overlap indices
            self.n_max_merge_test_clips = np.inf
        if merge_test_overlap_indices is None:
            # Use full overlap window by default
            merge_test_overlap_indices = work_items[0]['overlap']
        self.merge_test_overlap_indices = merge_test_overlap_indices
        self.min_overlapping_spikes = min_overlapping_spikes
        self.is_stitched = False # Repeated stitching can change results so track
        self.last_overlap_ratio_threshold = np.inf # Track how much overlap deleted units
        self.m_overlap_k_neighbors = m_overlap_k_neighbors
        self.verbose = verbose
        # Organize sort_data to be arranged for stitching and summarizing
        self.organize_sort_data()
        self.temporal_order_sort_data()
        if binary_pursuit_skip_time is None:
            self.binary_pursuit_skip_time = 0
        elif binary_pursuit_skip_time == 'clip_width':
            self.binary_pursuit_skip_time = sort_info['clip_width'][1] - sort_info['clip_width'][0]
        else:
            self.binary_pursuit_skip_time = binary_pursuit_skip_time
        if self.binary_pursuit_skip_time > 0:
            self.remove_segment_binary_pursuit_duplicates()
        self.delete_bad_mua_snr_units()

    def check_input_data(self, sort_data, work_items):
        """ Quick check to see if everything looks right with sort_data and
        work_items before proceeding. """
        if len(sort_data) != len(work_items):
            raise ValueError("Sort data and work items must be lists of the same length")
        # Make sure sort_data and work items are ordered one to one
        # by ordering their IDs
        sort_data.sort(key=lambda x: x[4])
        work_items.sort(key=lambda x: x['ID'])
        for s_data, w_item in zip(sort_data, work_items):
            if len(s_data) != 5:
                raise ValueError("sort_data for item", s_data[4], "is not the right size")
            if s_data[4] != w_item['ID']:
                raise ValueError("Could not match work order of sort data and work items. First failure on sort_data", s_data[4], "work item ID", w_item['ID'])
            empty_count = 0
            n_spikes = len(s_data[0])
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
                    sort_data_message = ("One element of sort data, but not " \
                                        "all, indicated no spikes for work " \
                                        "item {0} on channel {1}, segment " \
                                        "{2}. All data for this item are " \
                                        "being removed.".format(w_item['ID'], w_item['channel'], w_item['seg_number']))
                    warnings.warn(sort_data_message, RuntimeWarning, stacklevel=2)
        self.sort_data = sort_data
        self.work_items = work_items

    def organize_sort_data(self):
        """Takes the output of the spike sorting algorithms and organizes is by
        channel in segment order.
        Parameters
        ----------
        sort_data: list
            Each element contains a list of the 4 outputs of spike_sort_segment,
            and the ID of the work item that created it, split up by
            individual channels into 'work_items' (or spike_sort_parallel output)
            "crossings, labels, waveforms, new_waveforms", in this order, for a
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
            chan_items, chan_data = zip(*[[x, y] for x, y in sorted(
                                        zip(self.work_items, self.sort_data), key=lambda pair:
                                        pair[0]['seg_number'])
                                        if x['channel'] == chan])
            organized_data[chan].extend(chan_data)
            organized_items[chan].extend(chan_items)
        self.sort_data = organized_data
        self.work_items = organized_items

    def temporal_order_sort_data(self):
        """ Places all data within each segment in temporal order according to
        the spike event times. Use 'stable' sort for output to be repeatable
        because overlapping segments and binary pursuit can return identical
        dupliate spikes that become sorted in different orders. """
        self.neuron_summary_seg_inds = []
        for chan in range(0, self.n_chans):
            for seg in range(0, self.n_segments):
                if chan == 0:
                    # This is the same for every channel so only do once
                    self.neuron_summary_seg_inds.append(self.work_items[0][seg]['index_window'])
                    if self.neuron_summary_seg_inds[-1][1] - self.neuron_summary_seg_inds[-1][0] <= self.merge_test_overlap_indices:
                        raise ValueError("Number of merge test overlap indices must be less than the total segment size!")
                if len(self.sort_data[chan][seg][0]) == 0:
                    continue # No spikes in this segment
                spike_order = np.argsort(self.sort_data[chan][seg][0], kind='stable')
                self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][spike_order]
                self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][spike_order]
                self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][spike_order, :]
                self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][spike_order]

    def delete_label(self, chan, seg, label):
        """ Remove the unit corresponding to label from input segment and channel. """
        # NOTE: If all items are deleted, data will become empty numpy array
        # rather than the empty list style of original input
        keep_indices = self.sort_data[chan][seg][1] != label
        self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][keep_indices]
        self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][keep_indices]
        self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][keep_indices, :]
        self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][keep_indices]
        return keep_indices

    def delete_bad_mua_snr_units(self):
        """ Goes through each sorted item and removes any units with an SNR less
        than input min_snr or fraction MUA greater than max_mua_ratio.

        Function tracks and prints the lowest fraction_mua that was removed and
        the maximum SNR that was removed. This allows the user to understand
        whether any units of interest had segments that were 'just barely'
        removed under the current criteria. """
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
                    mua_ratio = self.get_fraction_mua_to_peak(chan, seg, l)
                    if mua_ratio > self.max_mua_ratio:
                        self.delete_label(chan, seg, l)
                        if mua_ratio < min_removed_mua:
                            min_removed_mua = mua_ratio
                            min_removed_mua_chan = chan
                            min_removed_mua_seg = seg
                        # Can skip computing SNR, but will not track max
                        # removed accurately
                        continue
                    select = self.sort_data[chan][seg][1] == l
                    snr = self.get_snr(chan, seg, np.mean(self.sort_data[chan][seg][2][select, :], axis=0))
                    if snr < self.min_snr:
                        self.delete_label(chan, seg, l)
                        if snr > max_removed_snr:
                            max_removed_snr = snr
                            max_removed_snr_chan = chan
                            max_removed_snr_seg = seg
        print("Least MUA removed was", min_removed_mua, "on channel", min_removed_mua_chan, "segment", min_removed_mua_seg)
        print("Maximum SNR removed was", max_removed_snr, "on channel", max_removed_snr_chan, "segment", max_removed_snr_seg)

    def remove_segment_binary_pursuit_duplicates(self):
        """ Removes overlapping spikes that were both found by binary
        pursuit. This can remove double dipping artifacts as binary pursuit attempts
        to minimize residual error. This only applies for data sorted within the
        same segment and belonging to the same unit. """
        skip_indices = int(round(self.binary_pursuit_skip_time * self.sort_info['sampling_rate']))
        for chan in range(0, self.n_chans):
            for seg in range(0, self.n_segments):
                if len(self.sort_data[chan][seg][0]) == 0:
                    # No data in this segment
                    continue
                keep_bool = np.ones(self.sort_data[chan][seg][0].shape[0], dtype=np.bool)
                for l in np.unique(self.sort_data[chan][seg][1]):
                    unit_select = self.sort_data[chan][seg][1] == l
                    keep_bool[unit_select] = remove_binary_pursuit_duplicates(
                            self.sort_data[chan][seg][0][unit_select],
                            self.sort_data[chan][seg][2][unit_select, :],
                            np.mean(self.sort_data[chan][seg][2][unit_select, :], axis=0),
                            self.sort_data[chan][seg][3][unit_select],
                            skip_indices)
                # NOTE: If all items are deleted, data will become empty numpy array
                # rather than the empty list style of original input
                self.sort_data[chan][seg][0] = self.sort_data[chan][seg][0][keep_bool]
                self.sort_data[chan][seg][1] = self.sort_data[chan][seg][1][keep_bool]
                self.sort_data[chan][seg][2] = self.sort_data[chan][seg][2][keep_bool, :]
                self.sort_data[chan][seg][3] = self.sort_data[chan][seg][3][keep_bool]

    def get_snr(self, chan, seg, full_template):
        """ Get SNR on the main channel relative to 3 STD of background noise. """
        background_noise_std = self.work_items[chan][seg]['thresholds'][self.work_items[chan][seg]['chan_neighbor_ind']] / self.sort_info['sigma']
        main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][seg]['chan_neighbor_ind'],
                    self.sort_info['n_samples_per_chan'] * (self.work_items[chan][seg]['chan_neighbor_ind'] + 1)]
        main_template = full_template[main_win[0]:main_win[1]]
        temp_range = np.amax(main_template) - np.amin(main_template)
        return temp_range / (3 * background_noise_std)

    def get_isi_violation_rate(self, chan, seg, label):
        """ Compute the spiking activity that occurs during the ISI violation
        period, absolute_refractory_period, relative to the total number of
        spikes within the given segment, 'seg' and unit label 'label'. """
        select_unit = self.sort_data[chan][seg][1] == label
        if ~np.any(select_unit):
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        unit_spikes = self.sort_data[chan][seg][0][select_unit]
        main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][seg]['chan_neighbor_ind'],
                    self.sort_info['n_samples_per_chan'] * (self.work_items[chan][seg]['chan_neighbor_ind'] + 1)]
        duplicate_tol_inds = calc_spike_half_width(
                                self.sort_data[chan][seg][2][select_unit][:, main_win[0]:main_win[1]],
                                self.sort_info['clip_width'], self.sort_info['sampling_rate'])
        duplicate_tol_inds += self.duplicate_tol_inds
        refractory_adjustment = duplicate_tol_inds / self.sort_info['sampling_rate']
        if self.absolute_refractory_period - refractory_adjustment <= 0:
            print("duplicate_tol_inds encompasses absolute_refractory_period. duplicate tolerence enforced at", self.duplicate_tol_inds)
            duplicate_tol_inds = self.duplicate_tol_inds
            refractory_adjustment = 0
            return -1.
        index_isi = np.diff(unit_spikes)
        num_isi_violations = np.count_nonzero(
            index_isi / self.sort_info['sampling_rate']
            < self.absolute_refractory_period)
        n_duplicates = np.count_nonzero(index_isi <= duplicate_tol_inds)
        # Remove duplicate spikes from this computation and adjust the number
        # of spikes and time window accordingly
        num_isi_violations -= n_duplicates
        isi_violation_rate = num_isi_violations \
                             * (1.0 / (self.absolute_refractory_period - refractory_adjustment))\
                             / (np.count_nonzero(select_unit) - n_duplicates)
        return isi_violation_rate

    def get_mean_firing_rate(self, chan, seg, label):
        """ Compute mean firing rate for the input channel, segment, and label. """
        select_unit = self.sort_data[chan][seg][1] == label
        first_index = next((idx[0] for idx, val in np.ndenumerate(select_unit) if val), [None])
        if first_index is None:
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        last_index = next((select_unit.shape[0] - idx[0] - 1 for idx, val in np.ndenumerate(select_unit[::-1]) if val), None)
        if self.sort_data[chan][seg][0][first_index] == self.sort_data[chan][seg][0][last_index]:
            return 0. # Only one spike
        mean_rate = self.sort_info['sampling_rate'] * np.count_nonzero(select_unit)\
                    / (self.sort_data[chan][seg][0][last_index] - self.sort_data[chan][seg][0][first_index])
        return mean_rate

    def get_fraction_mua(self, chan, seg, label):
        """ Estimate the fraction of noise/multi-unit activity (MUA) using analysis
        of ISI violations. We do this by looking at the spiking activity that
        occurs during the ISI violation period. """
        isi_violation_rate = self.get_isi_violation_rate(chan, seg, label)
        mean_rate = self.get_mean_firing_rate(chan, seg, label)
        if mean_rate == 0.:
            return 0.
        else:
            return (isi_violation_rate / mean_rate)

    def get_fraction_mua_to_peak(self, chan, seg, label, check_window=0.5):
        """
        """
        select_unit = self.sort_data[chan][seg][1] == label
        if ~np.any(select_unit):
            # There are no spikes that match this label
            raise ValueError("There are no spikes for label", label, ".")
        unit_spikes = self.sort_data[chan][seg][0][select_unit]
        main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][seg]['chan_neighbor_ind'],
                    self.sort_info['n_samples_per_chan'] * (self.work_items[chan][seg]['chan_neighbor_ind'] + 1)]
        duplicate_tol_inds = calc_spike_half_width(
                                self.sort_data[chan][seg][2][select_unit][:, main_win[0]:main_win[1]],
                                self.sort_info['clip_width'], self.sort_info['sampling_rate'])
        duplicate_tol_inds += self.duplicate_tol_inds
        all_isis = np.diff(unit_spikes)
        refractory_inds = int(round(self.absolute_refractory_period * self.sort_info['sampling_rate']))
        bin_width = refractory_inds - duplicate_tol_inds
        if bin_width <= 0:
            print("duplicate_tol_inds encompasses absolute_refractory_period so fraction MUA cannot be computed.")
            return -1.
        check_inds = int(round(check_window * self.sort_info['sampling_rate']))
        bin_edges = np.arange(duplicate_tol_inds+1, check_inds + bin_width, bin_width)
        counts, xval = np.histogram(all_isis, bin_edges)
        isi_peak = np.amax(counts)
        num_isi_violations = counts[0]
        if num_isi_violations < 0:
            num_isi_violations = 0
        if isi_peak == 0:
            isi_peak = max(num_isi_violations, 1.)
        fraction_mua_to_peak = num_isi_violations / isi_peak

        return fraction_mua_to_peak

    def merge_test_two_units(self, clips_1, clips_2, p_cut, method='template_pca',
                             split_only=False, merge_only=False, curr_chan_inds=None):
        if self.sort_info['add_peak_valley'] and curr_chan_inds is None:
            raise ValueError("Must give curr_chan_inds if using peak valley.")
        clips = np.vstack((clips_1, clips_2))
        neuron_labels = np.ones(clips.shape[0], dtype=np.int64)
        neuron_labels[clips_1.shape[0]:] = 2
        if method.lower() == 'pca':
            scores = preprocessing.compute_pca(clips,
                        self.sort_info['check_components'],
                        self.sort_info['max_components'],
                        add_peak_valley=self.sort_info['add_peak_valley'],
                        curr_chan_inds=curr_chan_inds)
        elif method.lower() == 'template_pca':
            scores = preprocessing.compute_template_pca(clips, neuron_labels,
                        curr_chan_inds, self.sort_info['check_components'],
                        self.sort_info['max_components'],
                        add_peak_valley=self.sort_info['add_peak_valley'])
        elif method.lower() == 'projection':
            # Projection onto templates, weighted by number of spikes
            t1 = np.mean(clips_1, axis=0) * (clips_1.shape[0] / clips.shape[0])
            t2 = np.mean(clips_2, axis=0) * (clips_2.shape[0] / clips.shape[0])
            scores = clips @ np.vstack((t1, t2)).T
        else:
            raise ValueError("Unknown method", method, "for scores. Must use 'pca' or 'projection'.")
        neuron_labels = merge_clusters(scores, neuron_labels,
                            split_only=split_only, merge_only=merge_only,
                            p_value_cut_thresh=p_cut, flip_labels=False)

        label_is_1 = neuron_labels == 1
        label_is_2 = neuron_labels == 2
        if np.all(label_is_1) or np.all(label_is_2):
            clips_merged = True
        else:
            clips_merged = False
        neuron_labels_1 = neuron_labels[0:clips_1.shape[0]]
        neuron_labels_2 = neuron_labels[clips_1.shape[0]:]
        return clips_merged, neuron_labels_1, neuron_labels_2

    def find_nearest_shifted_pair(self, chan, seg1, seg2, labels1, labels2,
                                  l2_workspace, curr_chan_inds,
                                  previously_compared_pairs):
        """ Alternative to sort_cython.identify_clusters_to_compare that simply
        chooses the nearest template after shifting to optimal alignment.
        Intended as helper function so that neurons do not fail to stitch in the
        event their alignment changes between segments. """
        best_distance = np.inf
        for l1 in labels1:
            # Choose seg1 template
            l1_select = self.sort_data[chan][seg1][1] == l1
            clips_1 = self.sort_data[chan][seg1][2][l1_select, :]
            # Only compare clips within specified overlap, if these are in a
            # different segment. If same segment use all clips
            if seg1 != seg2:
                start_edge = self.neuron_summary_seg_inds[seg1][1] - self.merge_test_overlap_indices
                c1_start = next((idx[0] for idx, val in np.ndenumerate(
                            self.sort_data[chan][seg1][0][l1_select])
                            if val >= start_edge), None)
                if c1_start is None:
                    continue
                if c1_start == clips_1.shape[0]:
                    continue
                clips_1 = clips_1[c1_start:, :]
                clips_1 = clips_1[max(clips_1.shape[0]-self.n_max_merge_test_clips, 0):, :]
            l1_template = np.mean(clips_1, axis=0)
            for l2 in labels2:
                if [l1, l2] in previously_compared_pairs:
                    continue
                l2_select = l2_workspace == l2
                clips_2 = self.sort_data[chan][seg2][2][l2_select, :]
                if seg1 != seg2:
                    # Stop at end of seg1 as this is end of overlap!
                    stop_edge = self.neuron_summary_seg_inds[seg1][1]
                    c2_start = next((idx[0] for idx, val in np.ndenumerate(
                                self.sort_data[chan][seg2][0][l2_select])
                                if val >= start_edge), None)
                    if c2_start is None:
                        continue
                    c2_stop = next((idx[0] for idx, val in np.ndenumerate(
                                self.sort_data[chan][seg2][0][l2_select])
                                if val > stop_edge), None)
                    if c2_start == c2_stop:
                        continue
                    clips_2 = clips_2[c2_start:c2_stop, :]
                    clips_2 = clips_2[:min(self.n_max_merge_test_clips, clips_2.shape[0]), :]
                l2_template = np.mean(clips_2, axis=0)
                cross_corr = np.correlate(l1_template[curr_chan_inds],
                                          l2_template[curr_chan_inds],
                                          mode='full')
                max_corr_ind = np.argmax(cross_corr)
                curr_shift = max_corr_ind - cross_corr.shape[0]//2
                # Align and truncate template and compute distance
                if curr_shift > 0:
                    shiftl1 = l1_template[curr_shift:]
                    shiftl2 = l2_template[:-1*curr_shift]
                elif curr_shift < 0:
                    shiftl1 = l1_template[:curr_shift]
                    shiftl2 = l2_template[-1*curr_shift:]
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
            best_l1_clips = None
            best_l2_clips = None
        # Align and truncate clips for best match pair
        if best_shift > 0:
            best_l1_clips = best_l1_clips[:, best_shift:]
            best_l2_clips = best_l2_clips[:, :-1*best_shift]
        elif best_shift < 0:
            best_l1_clips = best_l1_clips[:, :best_shift]
            best_l2_clips = best_l2_clips[:, -1*best_shift:]
        else:
            # No need to shift (or didn't find any pairs)
            pass
        return best_pair, best_shift, best_l1_clips, best_l2_clips

    def find_nearest_joint_pair(self, templates, labels, curr_chan_inds,
                                previously_compared_pairs):
        """
        """
        best_distance = np.inf
        best_pair = []
        best_shift = 0
        for i in range(0, len(labels)):
            for j in range(i+1, len(labels)):
                if [labels[i], labels[j]] in previously_compared_pairs:
                    continue
                cross_corr = np.correlate(templates[i][curr_chan_inds],
                                          templates[j][curr_chan_inds],
                                          mode='full')
                max_corr_ind = np.argmax(cross_corr)
                curr_shift = max_corr_ind - cross_corr.shape[0]//2
                # Align and truncate template and compute distance
                if curr_shift > 0:
                    shift_i = templates[i][curr_shift:]
                    shift_j = templates[j][:-1*curr_shift]
                elif curr_shift < 0:
                    shift_i = templates[i][:curr_shift]
                    shift_j = templates[j][-1*curr_shift:]
                else:
                    shift_i = templates[i]
                    shift_j = templates[j]
                curr_distance = np.sum((shift_i - shift_j) ** 2) / shift_i.shape[0]
                if curr_distance < best_distance:
                    best_shift = curr_shift
                    best_pair = [labels[i], labels[j]]
        return best_pair, best_shift

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
            print("Neurons are already stitched. Repeated stitching can change results.")
            print("Skipped stitching")
            return
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
            if start_seg >= len(self.sort_data[chan])-1:
                # Need at least 2 remaining segments to stitch
                jump_to_end = True
            else:
                main_win = [self.sort_info['n_samples_per_chan'] * self.work_items[chan][start_seg]['chan_neighbor_ind'],
                            self.sort_info['n_samples_per_chan'] * (self.work_items[chan][start_seg]['chan_neighbor_ind'] + 1)]
                curr_chan_inds = np.arange(main_win[0], main_win[1], dtype=np.int64)
                split_memory_dicts = [{} for x in range(0, self.n_segments)]
            # Go through each segment as the "current segment" and set the labels
            # in the next segment according to the scheme in current
            for curr_seg in range(start_seg, len(self.sort_data[chan])-1):
                if jump_to_end:
                    break
                next_seg = curr_seg + 1
                if len(self.sort_data[chan][curr_seg][1]) == 0:
                    # curr_seg is now failed next seg of last iteration
                    start_new_seg = True
                    if self.verbose: print("skipping seg", curr_seg, "with no spikes")
                    continue
                if start_new_seg:
                    # Map all units in this segment to new real labels
                    self.sort_data[chan][curr_seg][1] += next_real_label # Keep out of range
                    for nl in np.unique(self.sort_data[chan][curr_seg][1]):
                        # Add these new labels to real_labels for tracking
                        real_labels.append(nl)
                        if self.verbose: print("In NEXT SEG NEW (531) added real label", nl, chan, curr_seg)
                        next_real_label += 1
                if len(self.sort_data[chan][next_seg][1]) == 0:
                    # No units sorted in NEXT segment so start fresh next segment
                    start_new_seg = True
                    if self.verbose: print("skipping_seg", curr_seg, "because NEXT seg has no spikes")
                    continue

                # Make 'fake_labels' for next segment that do not overlap with
                # the current segment and make a work space for the next
                # segment labels so we can compare curr and next without
                # losing track of original labels
                fake_labels = np.copy(np.unique(self.sort_data[chan][next_seg][1]))
                fake_labels += next_real_label # Keep these out of range of the real labels
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
                    best_pair, best_shift, clips_1, clips_2 = self.find_nearest_shifted_pair(
                                    chan, curr_seg, next_seg, main_labels,
                                    leftover_labels, next_label_workspace,
                                    curr_chan_inds, previously_compared_pairs)
                    if len(best_pair) == 0:
                        break
                    if clips_1.shape[0] == 1 or clips_2.shape[0] == 1:
                        # Don't mess around with only 1 spike, if they are
                        # nearest each other they can merge
                        ismerged = True
                    else:
                        is_merged, _, _ = self.merge_test_two_units(
                                clips_1, clips_2, self.sort_info['p_value_cut_thresh'],
                                method='template_pca', merge_only=True,
                                curr_chan_inds=curr_chan_inds + best_shift)
                    if self.verbose: print("Item", self.work_items[chan][curr_seg]['ID'], "on chan", chan, "seg", curr_seg, "merged", is_merged, "for labels", best_pair)

                    if is_merged:
                        # Choose next seg spikes based on original fake label workspace
                        fake_select = next_label_workspace == best_pair[1]
                        # Update actual next segment label data with same labels
                        # used in curr_seg
                        self.sort_data[chan][next_seg][1][fake_select] = best_pair[0]
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
                    self.sort_data[chan][next_seg][1][ll_select] = next_real_label
                    real_labels.append(next_real_label)
                    if self.verbose: print("In leftover labels (882) added real label", next_real_label, chan, curr_seg)
                    next_real_label += 1

                # Make sure newly stitched segment labels are separate from
                # their nearest template match. This should help in instances
                # where one of the stitched units is mixed in one segment, but
                # in the other segment the units are separable (due to, for
                # instance, drift). This can help maintain segment continuity if
                # a unit exists in both segments but only has enough spikes/
                # separation to be sorted in one of the segments.
                # Do this by considering curr and next seg combined
                # NOTE: This could also be done by considering ALL previous
                # segments combined or once at the end over all data combined
                # joint_clips = np.vstack((self.sort_data[chan][curr_seg][2],
                #                          self.sort_data[chan][next_seg][2]))
                # joint_labels = np.hstack((self.sort_data[chan][curr_seg][1],
                #                           self.sort_data[chan][next_seg][1]))
                # joint_templates, temp_labels = segment.calculate_templates(
                #                         joint_clips, joint_labels)
                #
                # # Find all pairs of templates that are mutually closest
                # tmp_reassign = np.zeros_like(joint_labels)
                # temp_labels = temp_labels.tolist()
                # previously_compared_pairs = []
                # while len(temp_labels) > 0:
                #     best_pair, best_shift = self.find_nearest_joint_pair(
                #                     joint_templates, temp_labels,
                #                     curr_chan_inds, previously_compared_pairs)
                #     if len(best_pair) == 0:
                #         break
                #     previously_compared_pairs.append(best_pair)
                #     # Perform a split only between all minimum distance pairs
                #     c1, c2 = best_pair[0], best_pair[1]
                #     c1_select = joint_labels == c1
                #     clips_1 = joint_clips[c1_select, :]
                #     c2_select = joint_labels == c2
                #     clips_2 = joint_clips[c2_select, :]
                #
                #     # Align and truncate clips for best match pair
                #     if best_shift > 0:
                #         clips_1 = clips_1[:, best_shift:]
                #         clips_2 = clips_2[:, :-1*best_shift]
                #     elif best_shift < 0:
                #         clips_1 = clips_1[:, :best_shift]
                #         clips_2 = clips_2[:, -1*best_shift:]
                #     else:
                #         pass
                #     if clips_1.shape[0] == 1 or clips_2.shape[0] == 2:
                #         ismerged = True
                #     else:
                #         ismerged, labels_1, labels_2 = self.merge_test_two_units(
                #                 clips_1, clips_2, self.sort_info['p_value_cut_thresh'],
                #                 method='template_pca', split_only=True,
                #                 curr_chan_inds=curr_chan_inds)
                #     if ismerged:
                #         # This can happen if the split cutpoint forces
                #         # a merge so check and skip
                #         # Remove label with fewest spikes
                #         if clips_1.shape[0] >= clips_2.shape[0]:
                #             remove_l = best_pair[1]
                #         else:
                #             remove_l = best_pair[0]
                #         for x in reversed(range(0, len(temp_labels))):
                #             if temp_labels[x] == remove_l:
                #                 del temp_labels[x]
                #                 del joint_templates[x]
                #                 break
                #         continue
                #
                #     # Compute the neuron quality scores for each unit being
                #     # compared BEFORE reassigning based on split
                #     unit_1_score_pre = 0
                #     unit_2_score_pre = 0
                #     for curr_l in [c1, c2]:
                #         if curr_l in self.sort_data[chan][curr_seg][1]:
                #             mua_ratio = self.get_fraction_mua_to_peak(chan, curr_seg, curr_l)
                #             select = self.sort_data[chan][curr_seg][1] == curr_l
                #             curr_snr = self.get_snr(chan, curr_seg, np.mean(self.sort_data[chan][curr_seg][2][select, :], axis=0))
                #             if curr_l == c1:
                #                 unit_1_score_pre += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #             else:
                #                 unit_2_score_pre += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #         if curr_l in self.sort_data[chan][next_seg][1]:
                #             mua_ratio = self.get_fraction_mua_to_peak(chan, next_seg, curr_l)
                #             select = self.sort_data[chan][next_seg][1] == curr_l
                #             curr_snr = self.get_snr(chan, next_seg, np.mean(self.sort_data[chan][next_seg][2][select, :], axis=0))
                #             if curr_l == c1:
                #                 unit_1_score_pre += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #             else:
                #                 unit_2_score_pre += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #
                #     # Reassign spikes in c1 that split into c2
                #     # The merge test was done on joint clips and labels, so
                #     # we have to figure out where their indices all came from
                #     tmp_reassign[:] = 0
                #     tmp_reassign[c1_select] = labels_1
                #     tmp_reassign[c2_select] = labels_2
                #     curr_reassign_index_to_c1 = tmp_reassign[0:self.sort_data[chan][curr_seg][1].size] == 1
                #     curr_original_index_to_c1 = self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c1]
                #     self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c1] = c1
                #     curr_reassign_index_to_c2 = tmp_reassign[0:self.sort_data[chan][curr_seg][1].size] == 2
                #     curr_original_index_to_c2 = self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c2]
                #     self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c2] = c2
                #
                #     # Repeat for assignments in next_seg
                #     next_reassign_index_to_c1 = tmp_reassign[self.sort_data[chan][curr_seg][1].size:] == 1
                #     next_original_index_to_c1 = self.sort_data[chan][next_seg][1][next_reassign_index_to_c1]
                #     self.sort_data[chan][next_seg][1][next_reassign_index_to_c1] = c1
                #     next_reassign_index_to_c2 = tmp_reassign[self.sort_data[chan][curr_seg][1].size:] == 2
                #     next_original_index_to_c2 = self.sort_data[chan][next_seg][1][next_reassign_index_to_c2]
                #     self.sort_data[chan][next_seg][1][next_reassign_index_to_c2] = c2
                #
                #     # Check if split was a good idea and undo it if not. Basically,
                #     # if at least one of the units saw a 10% or greater improvement
                #     # in their quality score due to the split, we will stick with
                #     # the split. Otherwise, it probably didn't help or hurt, and
                #     # we should stick with the original sorter output.
                #     unit_1_score_post = 0
                #     unit_2_score_post = 0
                #     for curr_l in [c1, c2]:
                #         if curr_l in self.sort_data[chan][curr_seg][1]:
                #             mua_ratio = self.get_fraction_mua_to_peak(chan, curr_seg, curr_l)
                #             select = self.sort_data[chan][curr_seg][1] == curr_l
                #             curr_snr = self.get_snr(chan, curr_seg, np.mean(self.sort_data[chan][curr_seg][2][select, :], axis=0))
                #             if curr_l == c1:
                #                 unit_1_score_post += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #             else:
                #                 unit_2_score_post += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #         if curr_l in self.sort_data[chan][next_seg][1]:
                #             mua_ratio = self.get_fraction_mua_to_peak(chan, next_seg, curr_l)
                #             select = self.sort_data[chan][next_seg][1] == curr_l
                #             curr_snr = self.get_snr(chan, next_seg, np.mean(self.sort_data[chan][next_seg][2][select, :], axis=0))
                #             if curr_l == c1:
                #                 unit_1_score_post += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #             else:
                #                 unit_2_score_post += curr_snr * (1 - mua_ratio) * np.count_nonzero(select)
                #
                #     if (unit_1_score_post > 1.1*unit_1_score_pre) or (unit_2_score_post > 1.1*unit_2_score_pre):
                #         undo_split = False
                #     else:
                #         undo_split = True
                #     if undo_split:
                #         if self.verbose: print("undoing split between", c1, c2)
                #         if 2 in labels_1:
                #             self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c2] = curr_original_index_to_c2
                #             self.sort_data[chan][next_seg][1][next_reassign_index_to_c2] = next_original_index_to_c2
                #         if 1 in labels_2:
                #             self.sort_data[chan][curr_seg][1][curr_reassign_index_to_c1] = curr_original_index_to_c1
                #             self.sort_data[chan][next_seg][1][next_reassign_index_to_c1] = next_original_index_to_c1
                #     else:
                #         split_memory_dicts[curr_seg][c1] = [curr_reassign_index_to_c1, curr_original_index_to_c1]
                #         split_memory_dicts[curr_seg][c2] = [curr_reassign_index_to_c2, curr_original_index_to_c2]
                #     # NOTE: Not sure if this should depend on whether we split or not?
                #     # Remove label with fewest spikes
                #     if clips_1.shape[0] >= clips_2.shape[0]:
                #         remove_l = best_pair[1]
                #     else:
                #         remove_l = best_pair[0]
                #     for x in reversed(range(0, len(temp_labels))):
                #         if temp_labels[x] == remove_l:
                #             del temp_labels[x]
                #             del joint_templates[x]
                #             break

                # If we made it here then we are not starting a new seg
                start_new_seg = False
                if self.verbose: print("!!!REAL LABELS ARE !!!", real_labels, np.unique(self.sort_data[chan][curr_seg][1]), np.unique(self.sort_data[chan][next_seg][1]))
                if curr_seg > 0:
                    if self.verbose: print("Previous seg...", np.unique(self.sort_data[chan][curr_seg-1][1]))

            # It is possible to leave loop without checking last seg in the
            # event it is a new seg
            if start_new_seg and len(self.sort_data[chan][-1][0]) > 0:
                # Map all units in this segment to new real labels
                # Seg labels start at zero, so just add next_real_label. This
                # is last segment for this channel so no need to increment
                self.sort_data[chan][-1][1] += next_real_label
                real_labels.extend((np.unique(self.sort_data[chan][-1][1]) + next_real_label).tolist())
                if self.verbose: print("Last seg is new (747) added real labels", np.unique(self.sort_data[chan][-1][1]) + next_real_label, chan, curr_seg)
            if self.verbose: print("!!!REAL LABELS ARE !!!", real_labels)
            self.is_stitched = True

    def get_shifted_neighborhood_SSE(self, neuron1, neuron2):
        """
        """
        # Find the shared part of each unit's neighborhood and number of samples
        neighbor_overlap_bool = np.in1d(neuron1['neighbors'], neuron2['neighbors'])
        overlap_chans = neuron1['neighbors'][neighbor_overlap_bool]
        n_samples_per_chan = self.sort_info['n_samples_per_chan']

        # Build a template for each unit that has the intersection of
        # neighborhood channels
        overlap_template_1 = np.zeros(overlap_chans.shape[0] * n_samples_per_chan)
        overlap_template_2 = np.zeros(overlap_chans.shape[0] * n_samples_per_chan)
        for n_chan in range(0, overlap_chans.shape[0]):
            n1_neighbor_ind = next((idx[0] for idx, val in np.ndenumerate(neuron1['neighbors']) if val == overlap_chans[n_chan]))
            overlap_template_1[n_chan*n_samples_per_chan:(n_chan+1)*n_samples_per_chan] = \
                        neuron1['template'][n1_neighbor_ind*n_samples_per_chan:(n1_neighbor_ind+1)*n_samples_per_chan]
            n2_neighbor_ind = next((idx[0] for idx, val in np.ndenumerate(neuron2['neighbors']) if val == overlap_chans[n_chan]))
            overlap_template_2[n_chan*n_samples_per_chan:(n_chan+1)*n_samples_per_chan] = \
                        neuron2['template'][n2_neighbor_ind*n_samples_per_chan:(n2_neighbor_ind+1)*n_samples_per_chan]

        # Align templates based on peak cross correlation
        cross_corr = np.correlate(overlap_template_1, overlap_template_2, mode='full')
        max_corr_ind = np.argmax(cross_corr)
        shift = max_corr_ind - cross_corr.shape[0]//2
        # Align and truncate templates and compute SSE
        if shift > 0:
            overlap_template_1 = overlap_template_1[shift:]
            overlap_template_2 = overlap_template_2[:-1*shift]
        elif shift < 0:
            overlap_template_1 = overlap_template_1[:shift]
            overlap_template_2 = overlap_template_2[-1*shift:]
        else:
            # Already aligned
            pass
        # Must normalize distance per data point else reward big shifts
        SSE = np.sum((overlap_template_1 - overlap_template_2) ** 2) / overlap_template_1.shape[0]
        return SSE

    def summarize_neurons_by_seg(self):
        """ Make a neuron summary for each unit in each segment and add them to
        a new class attribute 'neuron_summary_by_seg'.

        """
        self.neuron_summary_by_seg = [[] for x in range(0, self.n_segments)]
        for seg in range(0, self.n_segments):
            for chan in range(0, self.n_chans):
                cluster_labels = np.unique(self.sort_data[chan][seg][1])
                for neuron_label in cluster_labels:
                    neuron = {}
                    neuron['summary_type'] = 'single_segment'
                    neuron["channel"] = self.work_items[chan][seg]['channel']
                    neuron['neighbors'] = self.work_items[chan][seg]['neighbors']
                    neuron['chan_neighbor_ind'] = self.work_items[chan][seg]['chan_neighbor_ind']
                    neuron['segment'] = self.work_items[chan][seg]['seg_number']
                    neuron['label'] = neuron_label
                    neuron['main_win'] = [self.sort_info['n_samples_per_chan'] * neuron['chan_neighbor_ind'],
                                          self.sort_info['n_samples_per_chan'] * (neuron['chan_neighbor_ind'] + 1)]
                    assert neuron['segment'] == seg, "Somethings messed up here?"
                    assert neuron['channel'] == chan

                    select_label = self.sort_data[chan][seg][1] == neuron_label
                    neuron["spike_indices"] = self.sort_data[chan][seg][0][select_label]
                    neuron['waveforms'] = self.sort_data[chan][seg][2][select_label, :]
                    neuron["new_spike_bool"] = self.sort_data[chan][seg][3][select_label]

                    # NOTE: This still needs to be done even though segments
                    # were ordered because of overlap!
                    # Ensure spike times are ordered. Must use 'stable' sort for
                    # output to be repeatable because overlapping segments and
                    # binary pursuit can return slightly different dupliate spikes
                    spike_order = np.argsort(neuron["spike_indices"], kind='stable')
                    neuron["spike_indices"] = neuron["spike_indices"][spike_order]
                    neuron['waveforms'] = neuron['waveforms'][spike_order, :]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][spike_order]

                    # Set duplicate tolerance as half spike width + duplicate_tol_inds
                    duplicate_tol_inds = calc_spike_half_width(
                        neuron['waveforms'][:, neuron['main_win'][0]:neuron['main_win'][1]],
                        self.sort_info['clip_width'], self.sort_info['sampling_rate'])
                    duplicate_tol_inds += self.duplicate_tol_inds
                    neuron['duplicate_tol_inds'] = duplicate_tol_inds
                    # Keep duplicates found in binary pursuit since it can reject
                    # false positives
                    keep_bool = keep_binary_pursuit_duplicates(neuron["spike_indices"],
                                    neuron["new_spike_bool"],
                                    tol_inds=duplicate_tol_inds)
                    neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                    neuron['waveforms'] = neuron['waveforms'][keep_bool, :]

                    # Remove any identical index duplicates (either from error or
                    # from combining overlapping segments), preferentially keeping
                    # the waveform best aligned to the template
                    neuron["template"] = np.mean(neuron['waveforms'], axis=0)
                    keep_bool = remove_spike_event_duplicates(neuron["spike_indices"],
                                    neuron['waveforms'], neuron["template"],
                                    tol_inds=duplicate_tol_inds)
                    neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                    neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                    neuron['waveforms'] = neuron['waveforms'][keep_bool, :]

                    # Recompute template and store output
                    neuron["template"] = np.mean(neuron['waveforms'], axis=0)
                    neuron['snr'] = self.get_snr(chan, seg, neuron["template"])
                    neuron['fraction_mua'] = calc_fraction_mua_to_peak(
                                                neuron["spike_indices"],
                                                self.sort_info['sampling_rate'],
                                                neuron['duplicate_tol_inds'],
                                                self.absolute_refractory_period)
                    neuron['threshold'] = self.work_items[chan][seg]['thresholds'][self.work_items[chan][seg]['channel']]
                    neuron['quality_score'] = neuron['snr'] * (1-neuron['fraction_mua']) \
                                                    * (neuron['spike_indices'].shape[0])
                    self.neuron_summary_by_seg[seg].append(neuron)

    def get_overlap_ratio(self, seg1, n1_ind, seg2, n2_ind, overlap_time=2.5e-4):
        """
        """
        # From next seg2 start to seg1 end
        overlap_win = [self.neuron_summary_seg_inds[seg2][0], self.neuron_summary_seg_inds[seg1][1]]
        max_samples = int(round(overlap_time * self.sort_info['sampling_rate']))
        n_total_samples = overlap_win[1] - overlap_win[0]
        if n_total_samples < 0:
            raise ValueError("Input seg2 must follow seg1")
        if n_total_samples < 10 * self.sort_info['sampling_rate']:
            summarize_message = ("Consecutive segment overlap is less than "\
                                 "10 seconds which may not provide enough "\
                                 "data to yield good across channel neuron summary")
            warnings.warn(sort_data_message, RuntimeWarning, stacklevel=2)

        n1_spikes = self.neuron_summary_by_seg[seg1][n1_ind]['spike_indices']
        n2_spikes = self.neuron_summary_by_seg[seg2][n2_ind]['spike_indices']
        if seg1 == seg2:
            n1_start = 0
        else:
            n1_start = next((idx[0] for idx, val in np.ndenumerate(n1_spikes)
                             if val >= overlap_win[0]), None)
        if n1_start is None:
            # neuron 1 has no spikes in the overlap
            return 0.
        if seg1 == seg2:
            n2_stop = n2_spikes.shape[0]
        else:
            n2_stop = next((idx[0] for idx, val in np.ndenumerate(n2_spikes)
                            if val >= overlap_win[1]), None)
        if n2_stop is None or n2_stop == 0:
            # neuron 2 has no spikes in the overlap
            return 0.
        n1_spike_train = compute_spike_trains(n1_spikes[n1_start:],
                                              max_samples, overlap_win)
        n_n1_spikes = np.count_nonzero(n1_spike_train)
        n2_spike_train = compute_spike_trains(n2_spikes[:n2_stop],
                                              max_samples, overlap_win)
        n_n2_spikes = np.count_nonzero(n2_spike_train)
        num_hits = np.count_nonzero(np.logical_and(n1_spike_train, n2_spike_train))
        n1_misses = np.count_nonzero(np.logical_and(n1_spike_train, ~n2_spike_train))
        n2_misses = np.count_nonzero(np.logical_and(n2_spike_train, ~n1_spike_train))
        overlap_ratio = max(num_hits / (num_hits + n1_misses),
                            num_hits / (num_hits + n2_misses))
        return overlap_ratio

    def remove_redundant_neurons(self, seg, overlap_time=2.5e-4, overlap_ratio_threshold=2):
        """
        Note that this function does not actually delete anything. It removes
        links between segments for redundant units and it adds a flag under
        the key 'deleted_as_redundant' to indicate that a segment unit should
        be deleted. Deleting units in this function would ruin the indices used
        to link neurons together later and is not worth the book keeping trouble.
        """
        neurons = self.neuron_summary_by_seg[seg]
        max_samples = int(round(overlap_time * self.sort_info['sampling_rate']))

        overlap_ratio = np.zeros((len(neurons), len(neurons)))
        expected_ratio = np.zeros((len(neurons), len(neurons)))
        quality_scores = np.zeros(len(neurons))
        violation_partners = [set() for x in range(0, len(neurons))]
        for neuron1_ind, neuron1 in enumerate(neurons):
            violation_partners[neuron1_ind].add(neuron1_ind)
            # Loop through all pairs of units and compute overlap and expected
            for neuron2_ind in range(neuron1_ind+1, len(neurons)):
                neuron2 = neurons[neuron2_ind]
                if neuron1['channel'] == neuron2['channel']:
                    continue # If they are on the same channel, do nothing
                # if neuron1['channel'] not in neuron2['neighbors']:
                #     continue # If they are not in same neighborhood, do nothing
                overlap_ratio[neuron1_ind, neuron2_ind] = self.get_overlap_ratio(
                                        seg, neuron1_ind, seg, neuron2_ind, overlap_time)
                overlap_ratio[neuron2_ind, neuron1_ind] = overlap_ratio[neuron1_ind, neuron2_ind]
                expected_hits = calculate_expected_overlap(neuron1, neuron2,
                                    overlap_time, self.sort_info['sampling_rate'])
                # NOTE: Should this be expected hits minus remaining number of spikes?
                expected_misses = min(neuron1['spike_indices'].shape[0], neuron2['spike_indices'].shape[0]) - expected_hits
                expected_ratio[neuron1_ind, neuron2_ind] = expected_hits / (expected_hits + expected_misses)
                expected_ratio[neuron2_ind, neuron1_ind] = expected_ratio[neuron1_ind, neuron2_ind]
                if (overlap_ratio[neuron1_ind, neuron2_ind] >=
                        overlap_ratio_threshold * expected_ratio[neuron1_ind, neuron2_ind]):
                    # Overlap is higher than chance and at least one of these will be removed
                    violation_partners[neuron1_ind].add(neuron2_ind)
                    violation_partners[neuron2_ind].add(neuron1_ind)

        neurons_remaining_indices = [x for x in range(0, len(neurons))]
        max_accepted = 0.
        max_expected = 0.
        while True:
            # Look for our next best pair
            best_ratio = -np.inf
            best_expected = -np.inf
            best_pair = []
            for i in range(0, len(neurons_remaining_indices)):
                for j in range(i+1, len(neurons_remaining_indices)):
                    neuron_1_index = neurons_remaining_indices[i]
                    neuron_2_index = neurons_remaining_indices[j]
                    if (overlap_ratio[neuron_1_index, neuron_2_index] <
                        overlap_ratio_threshold * expected_ratio[neuron_1_index, neuron_2_index]):
                        # Overlap not high enough to merit deletion of one
                        # But track our proximity to input threshold
                        if overlap_ratio[neuron_1_index, neuron_2_index] > max_accepted:
                            max_accepted = overlap_ratio[neuron_1_index, neuron_2_index]
                            max_expected = overlap_ratio_threshold * expected_ratio[neuron_1_index, neuron_2_index]
                        continue
                    if overlap_ratio[neuron_1_index, neuron_2_index] > best_ratio:
                        best_ratio = overlap_ratio[neuron_1_index, neuron_2_index]
                        best_pair = [neuron_1_index, neuron_2_index]
                        best_expected = expected_ratio[neuron_1_index, neuron_2_index]

            if len(best_pair) == 0 or best_ratio == 0:
                # No more pairs exceed ratio threshold
                print("Maximum accepted ratio was", max_accepted, "at expected threshold", max_expected)
                break



            # We now need to choose one of the pair to delete.
            neuron_1 = neurons[best_pair[0]]
            neuron_2 = neurons[best_pair[1]]
            delete_1 = False
            delete_2 = False
            """First doing the MUA and spike number checks because at this point
            the stitch segments function has deleted anything with MUA over the
            input max_mua_ratio. We can then fall back to SNR, since SNR does
            not always correspond to isolation quality, specifically in the case
            where other neurons are present on the same channel. Conversely,
            low MUA can indicate good isolation, or perhaps that the unit has a
            very small number of spikes. So we first consider MUA and spike
            count jointly before deferring to SNR. """
            if neuron_1['fraction_mua'] < 0 and neuron_2['fraction_mua'] < 0:
                print("Both units had BAD MUA")
                # MUA calculation was invalid so just use SNR
                if (neuron_1['snr']*neuron_1['spike_indices'].shape[0] > neuron_2['snr']*neuron_2['spike_indices'].shape[0]):
                    delete_2 = True
                else:
                    delete_1 = True
            elif neuron_1['fraction_mua'] < 0 or neuron_2['fraction_mua'] < 0:
                # MUA calculation was invalid for one unit so pick the other
                print("One unit had BAD MUA")
                if neuron_1['fraction_mua'] < 0:
                    delete_1 = True
                else:
                    delete_2 = True
            elif neuron_1['quality_score'] > neuron_2['quality_score']:
                delete_2 = True
            elif neuron_1['quality_score'] < neuron_2['quality_score']:
                delete_1 = True

            combined_violations = violation_partners[best_pair[0]].union(violation_partners[best_pair[1]])
            max_other_n1 = neuron_1['quality_score']
            other_n1 = combined_violations - violation_partners[best_pair[1]]
            for v_ind in other_n1:
                if quality_scores[v_ind] > max_other_n1:
                    max_other_n1 = quality_scores[v_ind]
            max_other_n2 = neuron_2['quality_score']
            other_n2 = combined_violations - violation_partners[best_pair[0]]
            for v_ind in other_n2:
                if quality_scores[v_ind] > max_other_n2:
                    max_other_n2 = quality_scores[v_ind]
            diff_score_1 = max_other_n1 - neuron_1['quality_score']
            diff_score_2 = max_other_n2 - neuron_2['quality_score']

            if len(combined_violations) == 2 or (diff_score_1 == 0 and diff_score_2 == 0):
                # This implies that these two units only violate with each other
                # or are their best remaining copies
                adjusted_n1_score = (1-neuron_1['fraction_mua']) * neuron_1['snr'] * (neuron_1['spike_indices'].shape[0] + (neuron_2['fraction_mua'] - neuron_1['fraction_mua'])*neuron_2['spike_indices'].shape[0])
                adjusted_n2_score = (1-neuron_2['fraction_mua']) * neuron_2['snr'] * (neuron_2['spike_indices'].shape[0] + (neuron_1['fraction_mua'] - neuron_2['fraction_mua'])*neuron_1['spike_indices'].shape[0])
                if (delete_2 and adjusted_n1_score < adjusted_n2_score) or \
                    (delete_1 and adjusted_n2_score < adjusted_n1_score):
                    print("Adjusted scores n1", adjusted_n1_score, "n2", adjusted_n2_score)
                    print("original scores n1", neuron_1['quality_score'], "n2", neuron_2['quality_score'])

                    # Need to union with compliment so spikes are not double
                    # counted, which will reduce the rate based MUA
                    max_duplicate_tol_inds = max(neuron_1['duplicate_tol_inds'], neuron_2['duplicate_tol_inds'])
                    neuron_2_compliment = np.in1d(neuron_1['spike_indices'], neuron_2['spike_indices'], invert=True)
                    union_spikes = np.hstack((neuron_1['spike_indices'][neuron_2_compliment], neuron_2['spike_indices']))
                    union_spikes.sort()
                    union_fraction_mua_rate = calc_fraction_mua(
                                                     union_spikes,
                                                     self.sort_info['sampling_rate'],
                                                     max_duplicate_tol_inds,
                                                     self.absolute_refractory_period)
                    # Need to get fraction MUA by rate, rather than peak,
                    # for comparison here
                    fraction_mua_rate_1 = calc_fraction_mua(
                                             neuron_1['spike_indices'],
                                             self.sort_info['sampling_rate'],
                                             max_duplicate_tol_inds,
                                             self.absolute_refractory_period)
                    fraction_mua_rate_2 = calc_fraction_mua(
                                             neuron_2['spike_indices'],
                                             self.sort_info['sampling_rate'],
                                             max_duplicate_tol_inds,
                                             self.absolute_refractory_period)

                    print("Union MUA", union_fraction_mua_rate, "n1 mua", fraction_mua_rate_1, "n2 mua", fraction_mua_rate_2)
                    if union_fraction_mua_rate > overlap_ratio_threshold * min(fraction_mua_rate_1, fraction_mua_rate_2) \
                        and union_fraction_mua_rate > max(fraction_mua_rate_1, fraction_mua_rate_2):
                        print("!!! OVERRIDING !!!")
                        print("!!! OVERRIDING !!!")
                        # The unit with more spikes and MUA is likely a bad mixture
                        # Choose unit with fewer spikes (also less MUA due to
                        # above IF statement)
                        if neuron_1['spike_indices'].shape[0] > neuron_2['spike_indices'].shape[0]:
                            delete_1 = True
                            delete_2 = False
                        else:
                            delete_1 = False
                            delete_2 = True

                """THIS IS COMPLETELY MAKING DECISIONS UNITL WE GET TO END. GUESS WE SHOULD
                PUT AN ELSE STATEMENT ABOVE FOR USING QUALITY SCORES"""
            elif diff_score_1 > diff_score_2:
                # Neuron 1 has a better copy somewhere so delete it
                if not delete_1:
                        print("!!! OVERRIDING !!!")
                        print("!!! OVERRIDING !!!")
                delete_1 = True
                delete_2 = False
            else:
                if not delete_2:
                        print("!!! OVERRIDING !!!")
                        print("!!! OVERRIDING !!!")
                delete_1 = False
                delete_2 = True

            # if neurons[best_pair[0]]['prev_seg_link'] is None and \
            #     neurons[best_pair[0]]['next_seg_link'] is None:
            #     if neurons[best_pair[1]]['prev_seg_link'] is not None or \
            #         neurons[best_pair[1]]['next_seg_link'] is not None:
            #         # Neuron 1 links to nothing, but neuron 2 has a link, so
            #         # keep neuron 2, delete 1
            #         print("!!! NEURON 1 LINKS TO NONE !!!")
            # if neurons[best_pair[1]]['prev_seg_link'] is None and \
            #     neurons[best_pair[1]]['next_seg_link'] is None:
            #     if neurons[best_pair[0]]['prev_seg_link'] is not None or \
            #         neurons[best_pair[0]]['next_seg_link'] is not None:
            #         # Neuron 2 links to nothing, but neuron 1 has a link, so
            #         # keep neuron 1, delete 2
            #         print("!!! NEURON 2 LINKS TO NONE !!!")

            print("Choosing from neurons with channels", neuron_1['channel'], neuron_2['channel'], "in segment", seg)
            if delete_1:
                print("Deleting neuron 1 with violators", violation_partners[best_pair[0]], "MUA", neuron_1['fraction_mua'], 'snr', neuron_1['snr'], "n spikes", neuron_1['spike_indices'].shape[0])
                print("Keeping neuron 2 with violators", violation_partners[best_pair[1]], "MUA", neuron_2['fraction_mua'], 'snr', neuron_2['snr'], "n spikes", neuron_2['spike_indices'].shape[0])
                neurons_remaining_indices.remove(best_pair[0])
                for vp in violation_partners:
                    vp.discard(best_pair[0])
                if seg > 0:
                    # Reassign any unit in previous segment with a next link
                    # to this unit to None
                    for p_ind, prev_n in enumerate(self.neuron_summary_by_seg[seg-1]):
                        if prev_n['next_seg_link'] is None:
                            continue
                        if prev_n['next_seg_link'] == best_pair[0]:
                                prev_n['next_seg_link'] = None
                if seg < self.n_segments - 1:
                    # Reassign any unit in next segment with a previous link
                    # to this unit to None
                    for next_n in self.neuron_summary_by_seg[seg+1]:
                        if next_n['prev_seg_link'] is None:
                            continue
                        if next_n['prev_seg_link'] == best_pair[0]:
                            next_n['prev_seg_link'] = None
                # Assign current neuron not to anything since
                # it is designated as trash fro deletion
                neurons[best_pair[0]]['prev_seg_link'] = None
                neurons[best_pair[0]]['next_seg_link'] = None
                neurons[best_pair[0]]['deleted_as_redundant'] = True
            if delete_2:
                print("Keeping neuron 1 with violators", violation_partners[best_pair[0]], "MUA", neuron_1['fraction_mua'], 'snr', neuron_1['snr'], "n spikes", neuron_1['spike_indices'].shape[0])
                print("Deleting neuron 2 with violators", violation_partners[best_pair[1]], "MUA", neuron_2['fraction_mua'], 'snr', neuron_2['snr'], "n spikes", neuron_2['spike_indices'].shape[0])
                neurons_remaining_indices.remove(best_pair[1])
                for vp in violation_partners:
                    vp.discard(best_pair[1])
                if seg > 0:
                    # Reassign any unit in previous segment with a next link
                    # to this unit to None
                    for p_ind, prev_n in enumerate(self.neuron_summary_by_seg[seg-1]):
                        if prev_n['next_seg_link'] is None:
                            continue
                        if prev_n['next_seg_link'] == best_pair[1]:
                                prev_n['next_seg_link'] = None
                if seg < self.n_segments - 1:
                    # Reassign any unit in next segment with a previous link
                    # to this unit to None
                    for next_n in self.neuron_summary_by_seg[seg+1]:
                        if next_n['prev_seg_link'] is None:
                            continue
                        if next_n['prev_seg_link'] == best_pair[1]:
                            next_n['prev_seg_link'] = None
                # Assign current neuron not to anything since
                # it is designated as trash fro deletion
                neurons[best_pair[1]]['prev_seg_link'] = None
                neurons[best_pair[1]]['next_seg_link'] = None
                neurons[best_pair[1]]['deleted_as_redundant'] = True
        return neurons

    def remove_redundant_neurons_by_seg(self, overlap_time=2.5e-4, overlap_ratio_threshold=2):
        """ Calls remove_redundant_neurons for each segment. """
        if overlap_ratio_threshold >= self.last_overlap_ratio_threshold:
            print("Redundant neurons already removed at threshold >=", overlap_ratio_threshold, "Further attempts will have no effect.")
            print("Skipping remove_redundant_neurons_by_seg.")
            return
        else:
            self.last_overlap_ratio_threshold = overlap_ratio_threshold
        for seg in range(0, self.n_segments):
            self.neuron_summary_by_seg[seg] = self.remove_redundant_neurons(
                                    seg, overlap_time, overlap_ratio_threshold)

    def make_overlapping_links(self, overlap_time):
        """
        """
        overlap_inds = int(round(overlap_time * self.sort_info['sampling_rate']))
        for seg in range(0, self.n_segments-1):
            n1_remaining = [x for x in range(0, len(self.neuron_summary_by_seg[seg]))
                            if self.neuron_summary_by_seg[seg][x]['next_seg_link'] is None]
            bad_n1 = []
            for n_ind in n1_remaining:
                if self.neuron_summary_by_seg[seg][n_ind]['deleted_as_redundant']:
                    bad_n1.append(n_ind)
                elif self.neuron_summary_by_seg[seg][n_ind]['fraction_mua'] > self.max_mua_ratio:
                    bad_n1.append(n_ind)
                elif self.neuron_summary_by_seg[seg][n_ind]['snr'] < self.min_snr:
                    bad_n1.append(n_ind)
            for dn in bad_n1:
                n1_remaining.remove(dn)
            while len(n1_remaining) > 0:
                best_n1_score = 0.
                best_n1_ind = 0
                for n1_ind in n1_remaining:
                    # Choose the best n1_remaining
                    if self.neuron_summary_by_seg[seg][n1_ind]['fraction_mua'] == -1:
                        # Invalid MUA so pick on SNR
                        n1_score = self.neuron_summary_by_seg[seg][n1_ind]['snr']
                    else:
                        n1_score = self.neuron_summary_by_seg[seg][n1_ind]['quality_score']
                    if n1_score > best_n1_score:
                        best_n1_score = n1_score
                        best_n1_ind = n1_ind
                n1_ind = best_n1_ind
                n1 = self.neuron_summary_by_seg[seg][n1_ind]

                best_n2_score = 0.
                min_SSE = np.inf
                max_overlap_ratio = -np.inf
                for n2_ind, n2 in enumerate(self.neuron_summary_by_seg[seg+1]):
                    # Choose the best n2 match for n1 that satisfies thresholds
                    if n2['prev_seg_link'] is None and not n2['deleted_as_redundant']:
                        if n1['channel'] not in n2['neighbors']:
                            continue
                        if n2['fraction_mua'] > self.max_mua_ratio:
                            continue
                        if n2['snr'] < self.min_snr:
                            continue
                        curr_overlap = self.get_overlap_ratio(
                                seg, n1_ind, seg+1, n2_ind, overlap_time)
                        if curr_overlap < self.min_overlapping_spikes:
                            continue

                        # # If we made it here, the overlap is sufficient to link
                        # # Now we must choose the best n2 to make it here
                        # if n2['fraction_mua'] == -1 and n1['fraction_mua'] == -1:
                        #     # Both invalid MUA, so pick on SNR
                        #     n2_score = n2['snr']
                        # elif n2['fraction_mua'] == -1 or n1['fraction_mua'] == -1:
                        #     # Only one unit has invalid MUA so don't link
                        #     continue
                        # else:
                        #     n2_score = n2['quality_score']
                        # if n2_score > best_n2_score:
                        #     best_n2_score = n2_score
                        #     max_overlap_pair = [n1_ind, n2_ind]

                        template_SSE = self.get_shifted_neighborhood_SSE(n1, n2)
                        if (curr_overlap > max_overlap_ratio) and (template_SSE < min_SSE):
                            max_overlap_ratio = curr_overlap
                            min_SSE = template_SSE
                            max_overlap_pair = [n1_ind, n2_ind]

                if max_overlap_ratio > 0:
                    # Found a match that satisfies thresholds so link them
                    self.neuron_summary_by_seg[seg][max_overlap_pair[0]]['next_seg_link'] = max_overlap_pair[1]
                    self.neuron_summary_by_seg[seg+1][max_overlap_pair[1]]['prev_seg_link'] = max_overlap_pair[0]
                    n1_remaining.remove(max_overlap_pair[0])
                else:
                    # Best remaining n1 could not find an n2 match so remove it
                    n1_remaining.remove(n1_ind)

    def stitch_neurons_across_channels(self):
        start_seg = 0
        while start_seg < self.n_segments:
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                start_seg += 1
                continue
            neurons = [[x] for x in self.neuron_summary_by_seg[start_seg]]
            break
        if start_seg >= self.n_segments-1:
            # Need at least 2 remaining segments to stitch.
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                # No segments with data found
                return [{}]
            # With this being the only segment, we are done. Each neuron is a
            # list with data only for the one segment
            neurons = [[n] for n in self.neuron_summary_by_seg[start_seg]]
            return neurons

        for next_seg in range(start_seg+1, self.n_segments):
            if len(self.neuron_summary_by_seg[next_seg]) == 0:
                # Nothing to link
                continue
            # For each currently known neuron, remove the neuron it connects to
            # (if any) in the next segment and add it to its own list
            # This is done by following the link of the last unit in the list
            next_seg_inds = [x for x in range(0, len(self.neuron_summary_by_seg[next_seg]))]
            for n_list in neurons:
                link_index = n_list[-1]['next_seg_link']
                if link_index is None:
                    continue
                n_list.append(self.neuron_summary_by_seg[next_seg][link_index])
                next_seg_inds.remove(link_index)
            for new_neuron in next_seg_inds:
                # Start a new list in neurons for anything that didn't link above
                neurons.append([self.neuron_summary_by_seg[next_seg][new_neuron]])
        return neurons

    def join_neuron_dicts(self, unit_dicts_list):
        """
        """
        if len(unit_dicts_list) == 0:
            print("Joine neuron dicst is returning empty")
            return {}
        combined_neuron = {}
        combined_neuron['summary_type'] = 'across_channels'
        combined_neuron['sort_info'] = self.sort_info
        # Since a neuron can exist over multiple channels, we need to discover
        # all the channels that are present and track which channel data
        # correspond to
        combined_neuron['channel'] = []
        combined_neuron['neighbors'] = {}
        combined_neuron['chan_neighbor_ind'] = {}
        combined_neuron['main_windows'] = {}
        combined_neuron['duplicate_tol_inds'] = 0
        chan_align_peak = {}
        n_total_spikes = 0
        n_peak = 0
        max_clip_samples = 0
        for x in unit_dicts_list:
            n_total_spikes += x['spike_indices'].shape[0]
            if x['channel'] not in combined_neuron['channel']:
                combined_neuron["channel"].append(x['channel'])
                combined_neuron['neighbors'][x['channel']] = x['neighbors']
                combined_neuron['chan_neighbor_ind'][x['channel']] = x['chan_neighbor_ind']
                combined_neuron['main_windows'][x['channel']] = x['main_win']
                chan_align_peak[x['channel']] = [0, 0] # [Peak votes, total]
                if x['waveforms'].shape[1] > max_clip_samples:
                    max_clip_samples = x['waveforms'].shape[1]
            if x['duplicate_tol_inds'] > combined_neuron['duplicate_tol_inds']:
                combined_neuron['duplicate_tol_inds'] = x['duplicate_tol_inds']
            if np.amax(x['template'][x['main_win'][0]:x['main_win'][1]]) \
                > np.amin(x['template'][x['main_win'][0]:x['main_win'][1]]):
                chan_align_peak[x['channel']][0] += 1
            chan_align_peak[x['channel']][1] += 1

        waveform_clip_center = int(round(np.abs(self.sort_info['clip_width'][0] * self.sort_info['sampling_rate']))) + 1
        chan_to_ind_map = {}
        for ind, chan in enumerate(combined_neuron['channel']):
            chan_to_ind_map[chan] = ind

        channel_selector = []
        indices_by_unit = []
        waveforms_by_unit = []
        new_spike_bool_by_unit = []
        threshold_by_unit = []
        segment_by_unit = []
        snr_by_unit = []
        for unit in unit_dicts_list:
            n_unit_events = unit['spike_indices'].size
            if n_unit_events == 0:
                continue
            # First adjust all spike indices to where they would have been if
            # aligned to the specified peak or valley for units on the same channel
            # NOTE: I can't think of a good way to do this reliably across channels
            if chan_align_peak[unit['channel']][0] / chan_align_peak[unit['channel']][1] > 0.5:
                # Most templates on this channel have greater peak so align peak
                shift = np.argmax(unit['template'][unit['main_win'][0]:unit['main_win'][1]]) - waveform_clip_center
            else:
                shift = np.argmin(unit['template'][unit['main_win'][0]:unit['main_win'][1]]) - waveform_clip_center
            shift = np.argmax(np.abs(unit['template'][unit['main_win'][0]:unit['main_win'][1]])) - waveform_clip_center

            indices_by_unit.append(unit['spike_indices'] + shift)
            waveforms_by_unit.append(unit['waveforms'])
            new_spike_bool_by_unit.append(unit['new_spike_bool'])
            # Make and append a bunch of book keeping numpy arrays
            channel_selector.append(np.full(n_unit_events, unit['channel']))
            threshold_by_unit.append(np.full(n_unit_events, unit['threshold']))
            segment_by_unit.append(np.full(n_unit_events, unit['segment']))
            snr_by_unit.append(np.full(n_unit_events, unit['snr']))

        # Now combine everything into one
        channel_selector = np.hstack(channel_selector)
        threshold_by_unit = np.hstack(threshold_by_unit)
        segment_by_unit = np.hstack(segment_by_unit)
        snr_by_unit = np.hstack(snr_by_unit)
        combined_neuron["spike_indices"] = np.hstack(indices_by_unit)
        # Need to account for fact that different channels can have different
        # neighborhood sizes. So make all waveforms start from beginning, and
        # remainder zeroed out if it has no data
        combined_neuron['waveforms'] = np.zeros((combined_neuron["spike_indices"].shape[0], max_clip_samples))
        wave_start_ind = 0
        for waves in waveforms_by_unit:
            combined_neuron['waveforms'][wave_start_ind:waves.shape[0]+wave_start_ind, 0:waves.shape[1]] = waves
            wave_start_ind += waves.shape[0]
        combined_neuron["new_spike_bool"] = np.hstack(new_spike_bool_by_unit)

        # Cleanup some memory that wasn't overwritten during concatenation
        del indices_by_unit
        del waveforms_by_unit
        del new_spike_bool_by_unit

        # NOTE: This still needs to be done even though segments
        # were ordered because of overlap!
        # Ensure everything is ordered. Must use 'stable' sort for
        # output to be repeatable because overlapping segments and
        # binary pursuit can return slightly different dupliate spikes
        spike_order = np.argsort(combined_neuron["spike_indices"], kind='stable')
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][spike_order]
        combined_neuron['waveforms'] = combined_neuron['waveforms'][spike_order, :]
        combined_neuron["new_spike_bool"] = combined_neuron["new_spike_bool"][spike_order]
        channel_selector = channel_selector[spike_order]
        threshold_by_unit = threshold_by_unit[spike_order]
        segment_by_unit = segment_by_unit[spike_order]
        snr_by_unit = snr_by_unit[spike_order]

        # max_dup = 0
        # combined_neuron['duplicate_tol_inds'] = self.duplicate_tol_inds
        # for chan in combined_neuron['channel']:
        #     chan_select = channel_selector == chan
        #     main_win = combined_neuron['main_windows'][chan]
        #     duplicate_tol_inds = calc_spike_half_width(
        #         combined_neuron['waveforms'][chan_select, main_win[0]:main_win[1]],
        #         self.sort_info['clip_width'], self.sort_info['sampling_rate'])
        #     duplicate_tol_inds += self.duplicate_tol_inds
        #     if duplicate_tol_inds > max_dup:
        #         combined_neuron['duplicate_tol_inds'] = duplicate_tol_inds

        # Remove duplicates found in binary pursuit
        keep_bool = keep_binary_pursuit_duplicates(combined_neuron["spike_indices"],
                        combined_neuron["new_spike_bool"],
                        tol_inds=combined_neuron['duplicate_tol_inds'])
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][keep_bool]
        combined_neuron["new_spike_bool"] = combined_neuron["new_spike_bool"][keep_bool]
        combined_neuron['waveforms'] = combined_neuron['waveforms'][keep_bool, :]
        channel_selector = channel_selector[keep_bool]
        threshold_by_unit = threshold_by_unit[keep_bool]
        segment_by_unit = segment_by_unit[keep_bool]
        snr_by_unit = snr_by_unit[keep_bool]

        # Get each spike's channel of origin and the waveforms on main channel
        # main_waveforms = np.zeros((combined_neuron['waveforms'].shape[0], self.sort_info['n_samples_per_chan']))
        combined_neuron['channel_selector'] = {}
        combined_neuron["template"] = {}
        for chan in combined_neuron['channel']:
            chan_select = channel_selector == chan
            combined_neuron['channel_selector'][chan] = chan_select
            # main_win = combined_neuron['main_windows'][chan]
            # main_waveforms[chan_select, :] = combined_neuron['waveforms'][chan_select, main_win[0]:main_win[1]]
            combined_neuron["template"][chan] = np.mean(
                combined_neuron['waveforms'][chan_select, :], axis=0)

        # Remove any identical index duplicates (either from error or
        # from combining overlapping segments), preferentially keeping
        # the waveform with largest peak-value to threshold ratio on its main
        # channel
        # keep_bool = remove_spike_event_duplicates_across_chans(combined_neuron["spike_indices"],
        #                 main_waveforms, threshold_by_unit, tol_inds=combined_neuron['duplicate_tol_inds'])
        keep_bool = remove_spike_event_duplicates_across_chans(combined_neuron)
        combined_neuron["spike_indices"] = combined_neuron["spike_indices"][keep_bool]
        combined_neuron["new_spike_bool"] = combined_neuron["new_spike_bool"][keep_bool]
        combined_neuron['waveforms'] = combined_neuron['waveforms'][keep_bool, :]
        for chan in combined_neuron['channel']:
            combined_neuron['channel_selector'][chan] = combined_neuron['channel_selector'][chan][keep_bool]
        channel_selector = channel_selector[keep_bool]
        threshold_by_unit = threshold_by_unit[keep_bool] # NOTE: not currently output...
        segment_by_unit = segment_by_unit[keep_bool]
        snr_by_unit = snr_by_unit[keep_bool]

        # Recompute things of interest like SNR and templates by channel as the
        # average over all data from that channel and store for output
        combined_neuron["template"] = {}
        combined_neuron["snr"] = {}
        chans_to_remove = []
        for chan in combined_neuron['channel']:
            if snr_by_unit[combined_neuron['channel_selector'][chan]].size == 0:
                # Spikes contributing from this channel have been removed so
                # remove all its data below
                chans_to_remove.append(chan)
            else:
                combined_neuron["template"][chan] = np.mean(
                    combined_neuron['waveforms'][combined_neuron['channel_selector'][chan], :], axis=0)
                combined_neuron['snr'][chan] = np.mean(snr_by_unit[combined_neuron['channel_selector'][chan]])
        for chan_ind in reversed(range(0, len(chans_to_remove))):
            chan_num = chans_to_remove[chan_ind]
            del combined_neuron['channel'][chan_ind] # A list so use index
            del combined_neuron['neighbors'][chan_num] # Rest of these are all
            del combined_neuron['chan_neighbor_ind'][chan_num] # dictionaries so
            del combined_neuron['channel_selector'][chan_num] #  use value
            del combined_neuron['main_windows'][chan_num]
        combined_neuron['fraction_mua'] = calc_fraction_mua_to_peak(
                        combined_neuron["spike_indices"],
                        self.sort_info['sampling_rate'],
                        combined_neuron['duplicate_tol_inds'],
                        self.absolute_refractory_period)

        return combined_neuron

    def summarize_neurons_across_channels(self, overlap_time=2.5e-4,
                overlap_ratio_threshold=2, min_segs_per_unit=1):
        """ Creates output neurons list by combining segment-wise neurons across
        segments and across channels based on stitch_segments and then using
        identical spikes found during the overlapping portions of consecutive
        segments. Requires that there be
        overlap between segments, and enough overlap to be useful.
        The overlap time is the window used to consider spikes the same"""
        if not self.is_stitched and self.n_segments > 1:
            summary_message = "Summarizing neurons for multiple data segments" \
                                "without first stitching will result in" \
                                "duplicate units discontinuous throughout the" \
                                "sorting time period. Call 'stitch_segments()' " \
                                "first to combine data across time segments."
            warnings.warn(summary_message, RuntimeWarning, stacklevel=2)

        # Start with neurons as those in the first segment with data
        start_seg = 0
        while start_seg < self.n_segments:
            if len(self.neuron_summary_by_seg[start_seg]) == 0:
                start_seg += 1
                continue
            # All units in first seg previous link to None
            for n in self.neuron_summary_by_seg[start_seg]:
                n['prev_seg_link'] = None
            break

        start_new_seg = False
        for seg in range(start_seg, self.n_segments-1):
            if len(self.neuron_summary_by_seg[seg]) == 0:
                start_new_seg = True
                continue
            if start_new_seg:
                for n in self.neuron_summary_by_seg[seg]:
                    n['prev_seg_link'] = None
                start_new_seg = False
            # For the current seg, we will discover their next seg link
            for n in self.neuron_summary_by_seg[seg]:
                n['next_seg_link'] = None
                # Take this opportunity to set this to default
                n['deleted_as_redundant'] = False
            next_seg = seg + 1
            if len(self.neuron_summary_by_seg[next_seg]) == 0:
                # Current seg neurons can't link with neurons in next seg
                continue
            # For the next seg, we will discover their previous seg link.
            # prev_seg_link is used by remove redundant neurons to match segments
            # across channels that otherwise have links broken by redundancy
            for n in self.neuron_summary_by_seg[next_seg]:
                n['prev_seg_link'] = None
            # Use stitching labels to link segments within channel
            for prev_ind, n in enumerate(self.neuron_summary_by_seg[seg]):
                for next_ind, next_n in enumerate(self.neuron_summary_by_seg[next_seg]):
                    if n['label'] == next_n['label'] and n['channel'] == next_n['channel']:
                        n['next_seg_link'] = next_ind
                        next_n['prev_seg_link'] = prev_ind

        # All units in last seg next link to None
        for n in self.neuron_summary_by_seg[self.n_segments-1]:
            n['next_seg_link'] = None
            n['deleted_as_redundant'] = False

        # Remove redundant items across channels and attempt to maintain
        # linking continuity across channels
        self.remove_redundant_neurons_by_seg(overlap_time=overlap_time,
                                overlap_ratio_threshold=overlap_ratio_threshold)
        # NOTE: This MUST run AFTER remove redundant by seg or else you can
        # end up linking a redundant mixture to a good unit with broken link!
        self.make_overlapping_links(overlap_time)

        neurons = self.stitch_neurons_across_channels()
        # Delete any redundant segs. These shouldn't really be in here anyways
        for n_list in neurons:
            inds_to_delete = []
            for n_seg_ind, n_seg in enumerate(n_list):
                if n_seg['deleted_as_redundant']:
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
        return neuron_summary

    def remove_redundant_within_channel_summaries(self, neurons, overlap_time=2.5e-4, overlap_ratio_threshold=2):
        """
        """
        max_samples = int(round(overlap_time * self.sort_info['sampling_rate']))
        n_total_samples = 0

        # Create list of sets of excessive neuron overlap between all pairwise units
        violation_partners = [set() for x in range(0, len(neurons))]
        for n1_ind, n1 in enumerate(neurons):
            violation_partners[n1_ind].add(n1_ind)
            if n1['spike_indices'][-1] > n_total_samples:
                # Find the maximum number of samples over all neurons while we are here for use later
                n_total_samples = n1['spike_indices'][-1]
            for n2_ind in range(n1_ind+1, len(neurons)):
                n2 = neurons[n2_ind]
                if np.intersect1d(n1['neighbors'][n1['channel'][0]], n2['neighbors'][n2['channel'][0]]).size == 0:
                    # Only count violations in neighborhood
                    continue
                violation_partners[n1_ind].add(n2_ind)
                violation_partners[n2_ind].add(n1_ind)

        overlap_ratio = np.zeros((len(neurons), len(neurons)))
        expected_ratio = np.zeros((len(neurons), len(neurons)))
        for neuron1_ind, neuron1 in enumerate(neurons):
            # Loop through all violators with neuron1
            # We already know these neurons are in each other's neighborhood
            # because violation_partners only includes neighbors
            for neuron2_ind in violation_partners[neuron1_ind]:
                if neuron2_ind <= neuron1_ind:
                    continue # Since our costs are symmetric, we only need to check indices greater than neuron1_ind
                neuron2 = neurons[neuron2_ind]
                if neuron1['channel'][0] == neuron2['channel'][0]:
                    continue # If they are on the same channel, do nothing

                n1_spike_train = compute_spike_trains(neuron1['spike_indices'],
                                                      max_samples, [0, n_total_samples])
                # n1_spike_train = np.ones(n_total_samples, dtype=np.bool)
                n_n1_spikes = np.count_nonzero(n1_spike_train)
                n2_spike_train = compute_spike_trains(neuron2['spike_indices'],
                                                      max_samples, [0, n_total_samples])
                # n2_spike_train = np.ones(n_total_samples, dtype=np.bool)
                n_n2_spikes = np.count_nonzero(n2_spike_train)
                num_hits = np.count_nonzero(np.logical_and(n1_spike_train, n2_spike_train))
                n1_misses = np.count_nonzero(np.logical_and(n1_spike_train, ~n2_spike_train))
                n2_misses = np.count_nonzero(np.logical_and(n2_spike_train, ~n1_spike_train))
                overlap_ratio[neuron1_ind, neuron2_ind] = max(num_hits / (num_hits + n1_misses),
                                                              num_hits / (num_hits + n2_misses))
                overlap_ratio[neuron2_ind, neuron1_ind] = overlap_ratio[neuron1_ind, neuron2_ind]
                expected_hits = calculate_expected_overlap(neuron1, neuron2,
                                    overlap_time, self.sort_info['sampling_rate'])
                expected_misses = min(neuron1['spike_indices'].shape[0], neuron2['spike_indices'].shape[0]) - expected_hits
                expected_ratio[neuron1_ind, neuron2_ind] = expected_hits / (expected_hits + expected_misses)
                expected_ratio[neuron2_ind, neuron1_ind] = expected_ratio[neuron1_ind, neuron2_ind]

        neurons_remaining_indices = [x for x in range(0, len(neurons))]
        neurons_to_remove = []
        max_accepted = 0.
        max_expected = 0.
        while True:
            # Look for our next best pair
            best_ratio = -np.inf
            best_expected = -np.inf
            best_pair = []
            for i in range(0, len(neurons_remaining_indices)):
                for j in range(i+1, len(neurons_remaining_indices)):
                    neuron_1_index = neurons_remaining_indices[i]
                    neuron_2_index = neurons_remaining_indices[j]
                    if (overlap_ratio[neuron_1_index, neuron_2_index] <
                        overlap_ratio_threshold * expected_ratio[neuron_1_index, neuron_2_index]):
                        # Overlap not high enough to merit deletion of one
                        # But track our proximity to input threshold
                        if overlap_ratio[neuron_1_index, neuron_2_index] > max_accepted:
                            max_accepted = overlap_ratio[neuron_1_index, neuron_2_index]
                            max_expected = overlap_ratio_threshold * expected_ratio[neuron_1_index, neuron_2_index]
                        continue
                    if overlap_ratio[neuron_1_index, neuron_2_index] > best_ratio:
                        best_ratio = overlap_ratio[neuron_1_index, neuron_2_index]
                        best_pair = [neuron_1_index, neuron_2_index]
                        best_expected = expected_ratio[neuron_1_index, neuron_2_index]

            if len(best_pair) == 0 or best_ratio == 0:
                # No more pairs exceed ratio threshold
                print("Maximum accepted ratio was", max_accepted, "at expected threshold", max_expected)
                break
            # We now need to choose one of the pair to delete.
            neuron_1 = neurons[best_pair[0]]
            neuron_2 = neurons[best_pair[1]]
            delete_1 = False
            delete_2 = False
            """First doing the MUA and spike number checks because at this point
            the stitch segments function has deleted anything with MUA over the
            input max_mua_ratio. We can then fall back to SNR, since SNR does
            not always correspond to isolation quality, specifically in the case
            where other neurons are present on the same channel. Conversely,
            low MUA can indicate good isolation, or perhaps that the unit has a
            very small number of spikes. So we first consider MUA and spike
            count jointly before deferring to SNR. """
            if neuron_1['fraction_mua'] < 0 and neuron_2['fraction_mua'] < 0:
                print("Both units had BAD MUA")
                # MUA calculation was invalid so just use SNR
                if (neuron_1['snr'] > neuron_2['snr']):
                    print("neuron 1 has higher SNR", neuron_1['snr'] , neuron_2['snr'])
                    delete_2 = True
                else:
                    delete_1 = True
            elif neuron_1['fraction_mua'] < 0 or neuron_2['fraction_mua'] < 0:
                # MUA calculation was invalid for one unit so pick the other
                print("One unit had BAD MUA")
                if neuron_1['fraction_mua'] < 0:
                    delete_2 = True
                else:
                    delete_1 = True
            elif neuron_1['fraction_mua'] < 1e-4 and neuron_2['fraction_mua'] < 1e-4:
                # Both MUA negligible so choose most spikes
                print("Both MUA negligible so choosing most spikes")
                if neuron_1['spike_indices'].shape[0] > neuron_2['spike_indices'].shape[0]:
                    delete_2 = True
                else:
                    delete_1 = True
            elif (neuron_1['snr']*((1-neuron_1['fraction_mua']) * neuron_1['spike_indices'].shape[0])
                   > 1.1*neuron_2['snr']*(1-neuron_2['fraction_mua']) * neuron_2['spike_indices'].shape[0]):
                # Neuron 1 has higher MUA weighted spikes
                print('Neuron 1 has higher MUA weighted spikes')
                print("MUA", neuron_1['fraction_mua'], neuron_2['fraction_mua'], "spikes", neuron_1['spike_indices'].shape[0], neuron_2['spike_indices'].shape[0])
                delete_2 = True
            elif (neuron_2['snr']*((1-neuron_2['fraction_mua']) * neuron_2['spike_indices'].shape[0])
                   > 1.1*neuron_1['snr']*(1-neuron_1['fraction_mua']) * neuron_1['spike_indices'].shape[0]):
                # Neuron 2 has higher MUA weighted spikes
                print('Neuron 2 has higher MUA weighted spikes')
                print("MUA", neuron_1['fraction_mua'], neuron_2['fraction_mua'], "spikes", neuron_1['spike_indices'].shape[0], neuron_2['spike_indices'].shape[0])
                delete_1 = True
            # Defer to choosing max SNR
            elif (neuron_1['snr'][neuron_1['channel'][0]] > neuron_2['snr'][neuron_2['channel'][0]]):
                print("neuron 1 has higher SNR", neuron_1['snr'] , neuron_2['snr'])
                delete_2 = True
            else:
                delete_1 = True

            if delete_1:
                neurons_to_remove.append(best_pair[0])
                neurons_remaining_indices.remove(best_pair[0])
            if delete_2:
                neurons_to_remove.append(best_pair[1])
                neurons_remaining_indices.remove(best_pair[1])

        for n_ind in reversed(range(0, len(neurons))):
            if n_ind in neurons_to_remove:
                del neurons[n_ind]
        return neurons

    def get_sort_data_by_chan(self):
        """ Returns all sorter data concatenated by channel across all segments,
        thus removing the concept of segment-wise sorting and giving all data
        by channel. """
        crossings, labels, waveforms, new_waveforms = [], [], [], []
        thresholds = []
        for chan in range(0, self.n_chans):
            seg_crossings, seg_labels, seg_waveforms, seg_new = [], [], [], []
            seg_thresholds =[]
            for seg in range(0, self.n_segments):
                seg_crossings.append(self.sort_data[chan][seg][0])
                seg_labels.append(self.sort_data[chan][seg][1])
                # Waveforms is 2D so can't stack with empty
                if len(self.sort_data[chan][seg][2]) > 0:
                    seg_waveforms.append(self.sort_data[chan][seg][2])
                seg_new.append(self.sort_data[chan][seg][3])
                seg_thresholds.append(self.work_items[chan][seg]['thresholds'][self.work_items[chan][seg]['chan_neighbor_ind']])
            # stacking with [] casts as float, so ensure maintained types
            crossings.append(np.hstack(seg_crossings).astype(np.int64))
            labels.append(np.hstack(seg_labels).astype(np.int64))
            if len(seg_waveforms) > 0:
                waveforms.append(np.vstack(seg_waveforms).astype(np.float64))
            else:
                waveforms.append([])
            new_waveforms.append(np.hstack(seg_new).astype(np.bool))
            thresholds.append(np.hstack(seg_thresholds).astype(np.float64))
        return crossings, labels, waveforms, new_waveforms, thresholds

    def summarize_neurons_within_channel(self, overlap_time=2.5e-4, overlap_ratio_threshold=2):
        """
        Return a summarized version of the threshold_crossings, labels, etc. for a given
        set of crossings, neuron labels, and waveforms organized by channel.
        This function returns a list of dictionarys (with symbol look-ups)
        with all essential information about the recording session and sorting.
        The dictionary contains:
        channel: The channel on which the neuron was recorded
        neighbors: The neighborhood of the channel on which the neuron was recorded
        clip_width: The clip width used for sorting the neuron
        sampling_rate: The sampling rate of the recording
        filter_band: The filter band used to filter the data before sorting
        spike_indices: The indices (sample number) in the voltage trace of the threshold crossings
         waveforms: The waveform clips across the entire neighborhood for the sorted neuron
         template: The template for the neuron as the mean of all waveforms.
         new_spike_bool: A logical index into waveforms/spike_indices of added secret spikes
            (only if the index 'new_waveforms' is input)
         mean_firing_rate: The mean firing rate of the neuron over the entire recording
        peak_valley: The peak valley of the template on the channel from which the neuron arises
        """
        if not self.is_stitched and self.n_segments > 1:
            summary_message = "Summarizing neurons for multiple data segments" \
                                "without first stitching will result in" \
                                "duplicate units discontinuous throughout the" \
                                "sorting time period. Call 'stitch_segments()' " \
                                "first to combine data across time segments."
            warnings.warn(summary_message, RuntimeWarning, stacklevel=2)
        # To the sort data segments into our conventional sorter items by
        # channel independent of segment/work item
        crossings, labels, waveforms, new_waveforms, thresholds = self.get_sort_data_by_chan()
        neuron_summary = []
        for channel in range(0, len(crossings)):
            if len(crossings[channel]) == 0:
                # This channel has no spikes
                continue
            for ind, neuron_label in enumerate(np.unique(labels[channel])):
                neuron = {}
                neuron['summary_type'] = 'single_channel'
                neuron['sort_info'] = self.sort_info
                neuron["channel"] = [channel]
                # Work items are sorted by channel so just grab neighborhood
                # for the first segment
                neuron['neighbors'] = {channel: self.work_items[channel][0]['neighbors']}
                neuron['chan_neighbor_ind'] = {channel: self.work_items[channel][0]['chan_neighbor_ind']}
                # Thresholds is average over all segments
                neuron['threshold'] = {channel: np.mean(thresholds[channel])}
                neuron['main_win'] = {channel: [self.sort_info['n_samples_per_chan'] * neuron['chan_neighbor_ind'][channel],
                                      self.sort_info['n_samples_per_chan'] * (neuron['chan_neighbor_ind'][channel] + 1)]}

                neuron["spike_indices"] = crossings[channel][labels[channel] == neuron_label]
                neuron['waveforms'] = waveforms[channel][labels[channel] == neuron_label, :]
                neuron["new_spike_bool"] = new_waveforms[channel][labels[channel] == neuron_label]

                # NOTE: This still needs to be done even though segments
                # were ordered because of overlap!
                # Ensure spike times are ordered. Must use 'stable' sort for
                # output to be repeatable because overlapping segments and
                # binary pursuit can return slightly different dupliate spikes
                spike_order = np.argsort(neuron["spike_indices"], kind='stable')
                neuron["spike_indices"] = neuron["spike_indices"][spike_order]
                neuron['waveforms'] = neuron['waveforms'][spike_order, :]
                neuron["new_spike_bool"] = neuron["new_spike_bool"][spike_order]
                # Remove duplicates found in binary pursuit
                duplicate_tol_inds = calc_spike_half_width(
                    neuron['waveforms'][:, neuron['main_win'][channel][0]:neuron['main_win'][channel][1]],
                    self.sort_info['clip_width'], self.sort_info['sampling_rate'])
                duplicate_tol_inds += self.duplicate_tol_inds
                neuron['duplicate_tol_inds'] = duplicate_tol_inds
                keep_bool = keep_binary_pursuit_duplicates(neuron["spike_indices"],
                                neuron["new_spike_bool"],
                                tol_inds=duplicate_tol_inds)
                neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                neuron['waveforms'] = neuron['waveforms'][keep_bool, :]

                # Remove any identical index duplicates (either from error or
                # from combining overlapping segments), preferentially keeping
                # the waveform best aligned to the template
                neuron["template"] = {channel: np.mean(neuron['waveforms'], axis=0)}
                keep_bool = remove_spike_event_duplicates(neuron["spike_indices"],
                                neuron['waveforms'], neuron["template"][channel],
                                tol_inds=duplicate_tol_inds)
                neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
                neuron["new_spike_bool"] = neuron["new_spike_bool"][keep_bool]
                neuron['waveforms'] = neuron['waveforms'][keep_bool, :]
                neuron["template"] = {channel: np.mean(neuron['waveforms'], axis=0)}

                background_noise_std = neuron['threshold'][channel] / self.sort_info['sigma']
                main_template = neuron["template"][channel][neuron['main_win'][channel][0]:neuron['main_win'][channel][1]]
                temp_range = np.amax(main_template) - np.amin(main_template)
                neuron['snr'] = {channel: temp_range / (3 * background_noise_std)}

                neuron['fraction_mua'] = calc_fraction_mua_to_peak(
                                neuron["spike_indices"],
                                self.sort_info['sampling_rate'],
                                neuron['duplicate_tol_inds'],
                                self.absolute_refractory_period)
                neuron_summary.append(neuron)
        print("Redundant removal turned OFF")
        # neuron_summary = self.remove_redundant_within_channel_summaries(neuron_summary,
        #                                 overlap_time, overlap_ratio_threshold)
        return neuron_summary


def calc_m_overlap_ab(clips_a, clips_b, k_neighbors):
    """
    """
    # Check at most 500 random samples from each cluster
    N = min(clips_a.shape[0], clips_b.shape[0])
    N = min(N, 500)
    clips_aUb = np.vstack((clips_a, clips_b))
    if k_neighbors > clips_a.shape[0] or k_neighbors > clips_b.shape[0]:
        k_neighbors = min(clips_a.shape[0], clips_b.shape[0])
        print("Input K neighbors for calc_m_overlap_ab exceeds at least one cluster size. Resetting to", k_neighbors)
    k_in_a = 0.
    for na in range(0, N):
        x = np.random.randint(0, clips_a.shape[0], 1)
        # Distance of x from all points in a union b
        x_dist = np.sum((clips_a[x, :] - clips_aUb) ** 2, axis=1)

        k_nearest_inds = np.argpartition(x_dist, k_neighbors)[:k_neighbors]
        k_in_a += (np.count_nonzero(k_nearest_inds < clips_a.shape[0]) / k_neighbors) / clips_aUb.shape[0]
    print("k in a", k_in_a)
    k_in_b = 0.
    for nb in range(0, N):
        x = np.random.randint(0, clips_b.shape[0], 1)
        # Distance of x from all points in a union b
        x_dist = np.sum((clips_b[x, :] - clips_aUb) ** 2, axis=1)

        k_nearest_inds = np.argpartition(x_dist, k_neighbors)[:k_neighbors]
        k_in_b += (np.count_nonzero(k_nearest_inds >= clips_a.shape[0]) / k_neighbors) / clips_aUb.shape[0]
    print("k in b", k_in_b)
    m_overlap_ab = (k_in_a + k_in_b) #/ 2*N

    return m_overlap_ab


def calc_m_isolation(clips, neuron_labels, k_neighbors):
    """
    """
    cluster_labels = np.unique(neuron_labels)
    if cluster_labels.size == 1:
        return np.ones(1), cluster_labels
    m_overlap_ab = np.zeros((cluster_labels.shape[0], cluster_labels.shape[0]))
    m_isolation = np.zeros(cluster_labels.shape[0])
    # First get all pairwise overlap measures. These are symmetric
    for a in range(0, cluster_labels.shape[0]):
        clips_a = clips[neuron_labels == cluster_labels[a], :]
        for b in range(a+1, cluster_labels.shape[0]):
            clips_b = clips[neuron_labels == cluster_labels[b], :]
            m_overlap_ab[a, b] = calc_m_overlap_ab(clips_a, clips_b, k_neighbors)
            m_overlap_ab[b, a] = m_overlap_ab[a, b]
        m_isolation[a] = np.amax(m_overlap_ab[a, :])

    return m_isolation, cluster_labels
