import numpy as np
from os import path

from spikesorting_fullpursuit.processing.conversions import time_window_to_samples
from spikesorting_fullpursuit.utils.memmap_close import MemMapClose


def keep_valid_inds(keep_data_list: list, valid_inds):
    """

    Args:
        Keep_data_list: List of numpy arrays, such as event indicies or neuron labels
        valid_inds: 1D numpy array of booleans, which specify which elements of 
            keep_data_list are valid

    Returns: Either a tuple of numpy arrays or a single numpy array
        that contains only the valid elements of keep_data_list

    """

    out_data = []
    for data in keep_data_list:
        out_data.append(data[valid_inds])
    return tuple(x for x in out_data) if len(keep_data_list) > 1 else out_data[0]


def memmap_to_mem(memmap, dtype=None, order=None):
    """ 
    Helpful function that takes a numpy memmap as input and copies it to a
    numpy array in memory as output. 
    """
    if not isinstance(memmap, np.memmap):
        raise ValueError("Input object is not instance of numpy.memmap")
    if dtype is None:
        dtype = memmap.dtype
    if order is None:
        order = "F" if memmap.flags['F_CONTIGUOUS'] else "C"
    mem = np.empty(memmap.shape, dtype=dtype, order=order)
    np.copyto(mem, memmap)

    return mem


def get_windows_and_indices(
        clip_width_s,
        sampling_rate,
        channel,
        neighbors):
    """
    Computes some basic info used in many functions about how clips are
    are formatted and provides window indices and clip indices. 
    """

    curr_chan_win, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    chan_neighbor_ind = next((idx[0] for idx, val in np.ndenumerate(neighbors) if val == channel), None)
    samples_per_chan = curr_chan_win[1] - curr_chan_win[0]
    curr_chan_inds = np.arange(
        samples_per_chan * chan_neighbor_ind,
        samples_per_chan * chan_neighbor_ind + samples_per_chan,
        1
        )

    return clip_width_s, chan_neighbor_ind, curr_chan_win, samples_per_chan, curr_chan_inds


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


def get_zero_phase_kernel(x, x_center):
    """
    Zero pads the 1D kernel x, so that it is aligned with the current element
    of x located at x_center.  This ensures that convolution with the kernel
    x will be zero phase with respect to x_center.
    """

    kernel_offset = (x.size
                     - 2 * x_center
                     - 1)  # -1 To center ON the x_center index
    kernel_size = np.abs(kernel_offset) + x.size
    if kernel_size // 2 == 0:  # Kernel must be odd
        kernel_size -= 1
        kernel_offset -= 1
    kernel = np.zeros(kernel_size)
    if kernel_offset > 0:
        kernel_slice = slice(kernel_offset, kernel.size)
    elif kernel_offset < 0:
        kernel_slice = slice(0, kernel.size + kernel_offset)
    else:
        kernel_slice = slice(0, kernel.size)
    kernel[kernel_slice] = x

    return kernel


def get_singlechannel_clips(
        probe_dict,
        chan_voltage,
        event_indices,
        clip_width_s,
        use_memmap=False) -> tuple[np.ndarray | MemMapClose, np.ndarray]:
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

    sampling_rate = probe_dict['sampling_rate']

    window, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices = np.ones(event_indices.shape[0], dtype="bool")

    start_ind = 0
    n = event_indices[start_ind]
    while n + window[0] < 0:
        valid_event_indices[start_ind] = False
        start_ind += 1
        if start_ind == event_indices.size:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[start_ind]

    stop_ind = event_indices.shape[0] - 1
    n = event_indices[stop_ind]
    while n + window[1] > probe_dict['n_samples']:
        valid_event_indices[stop_ind] = False
        stop_ind -= 1
        if stop_ind < 0:
            # There are no valid indices
            valid_event_indices[:] = False
            return None, valid_event_indices
        n = event_indices[stop_ind]

    if use_memmap:
        clip_fname = path.join(probe_dict['memmap_dir'], "{0}clips_{1}.bin".format(probe_dict['memmap_fID'], str(probe_dict['ID'])))
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict['v_dtype'],
            mode='w+',
            shape=(np.count_nonzero(valid_event_indices),
                   window[1] - window[0])
            )
    else:
        spike_clips = np.empty(
            (np.count_nonzero(valid_event_indices), window[1] - window[0]),
            dtype=probe_dict['v_dtype'])

    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        spike_clips[out_ind, :] = chan_voltage[event_indices[spk]+window[0]:event_indices[spk]+window[1]]

    if use_memmap:
        if isinstance(spike_clips, np.memmap):
            spike_clips.flush()
            spike_clips._mmap.close()
            del spike_clips
        # Make output read only
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict['v_dtype'],
            mode='r',
            shape=(
                np.count_nonzero(valid_event_indices),
                window[1] - window[0]
                )
            )

    return spike_clips, valid_event_indices


def get_clips(
        probe_dict,
        voltage,
        neighbors: np.ndarray,
        event_indices,
        clip_width_s: list,
        use_memmap=False,
        check_valid=True) -> tuple[np.ndarray | MemMapClose, np.ndarray]:
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
        raise ValueError(
            "Event_indices must be one dimensional array of indices"
            )

    sampling_rate = probe_dict['sampling_rate']

    window, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    if len(event_indices) == 0:
        # No indices input
        return np.zeros((0, (window[1] - window[0]) * len(neighbors)), dtype=probe_dict['v_dtype']), np.ones(0, dtype="bool")
    # Ignore spikes whose clips extend beyond the data and create mask for removing them
    valid_event_indices: np.ndarray = np.ones(event_indices.shape[0], dtype="bool")
    start_ind = 0
    if check_valid:
        n = event_indices[start_ind]
        while (n + window[0]) < 0:
            valid_event_indices[start_ind] = False
            start_ind += 1
            if start_ind == event_indices.size:
                # There are no valid indices
                valid_event_indices[:] = False
                return None, valid_event_indices
            n = event_indices[start_ind]
        stop_ind = event_indices.shape[0] - 1
        n = event_indices[stop_ind]
        while (n + window[1]) >= probe_dict['n_samples']:
            valid_event_indices[stop_ind] = False
            stop_ind -= 1
            if stop_ind < 0:
                # There are no valid indices
                valid_event_indices[:] = False
                return None, valid_event_indices
            n = event_indices[stop_ind]
    else:
        stop_ind = len(event_indices) - 1

    if use_memmap:
        clip_fname = path.join(probe_dict['memmap_dir'], "{0}clips_{1}.bin".format(probe_dict['memmap_fID'], str(probe_dict['ID'])))
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict['v_dtype'],
            mode='w+',
            shape=(
                np.count_nonzero(valid_event_indices),
                (window[1] - window[0]) * len(neighbors)
                )
            )
    else:
        spike_clips: np.ndarray = np.empty((np.count_nonzero(valid_event_indices), (window[1] - window[0]) * len(neighbors)), dtype=probe_dict['v_dtype'])

    for out_ind, spk in enumerate(range(start_ind, stop_ind+1)): # Add 1 to index through last valid index
        start = 0
        for n_ind, chan in enumerate(neighbors):
            stop = (n_ind + 1) * (window[1] - window[0])
            spike_clips[out_ind, start:stop] = voltage[chan, event_indices[spk]+window[0]:event_indices[spk]+window[1]]
            start = stop

    if use_memmap:
        if isinstance(spike_clips, np.memmap):
            spike_clips.flush()
            spike_clips._mmap.close()
            del spike_clips
        # Make output read only
        spike_clips = MemMapClose(
            clip_fname,
            dtype=probe_dict['v_dtype'],
            mode='r',
            shape=(
                np.count_nonzero(valid_event_indices),
                (window[1] - window[0]) * len(neighbors)
                )
            )

    return spike_clips, valid_event_indices
