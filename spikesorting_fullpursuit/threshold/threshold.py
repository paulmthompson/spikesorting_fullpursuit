import numpy as np

from spikesorting_fullpursuit.processing.conversions import time_window_to_samples


def median_threshold(voltage, sigma):
    """
    Determines the per-channel threshold necessary for the detection of spikes.
    This function returns a vector of thresholds (one for each channel). These
    represent the absolute value of the threshold.
    """
    if voltage.ndim == 1:
        voltage = np.expand_dims(voltage, 0)
    num_channels = voltage.shape[0]
    thresholds = np.empty((num_channels, ))
    for chan in range(0, num_channels):
        abs_voltage = np.abs(voltage[chan, :])
        thresholds[chan] = np.nanmedian(abs_voltage) / 0.6745
    thresholds *= sigma

    return thresholds


def identify_threshold_crossings(
        chan_voltage,
        probe_dict,
        threshold,
        skip=0.,
        align_window=[-5e-4, 5e-4]) -> tuple[np.ndarray, int]:
    """

    Args:
        chan_voltage: 2D numpy array of voltage values for segment
            (channels x samples)
        item_dict = {'sampling_rate',
                     'n_samples',
                     'thresholds': 1D numpy array of thresholds for each channel
                     'v_dtype',
                     'ID',
                     'memmap_dir',
                     'memmap_fID'
        threshold: float, threshold for this work item channel
        skip: float, minimum time between threshold crossings in seconds
        align_window: list of two floats, time window in seconds to align

    Returns:
        events: 1D numpy array of indices of threshold crossings aligned to max
            or min in align_window
        n_crossings: int, total number of threshold crossings (does not
            account for skips or alignment)
    """

    sampling_rate = probe_dict['sampling_rate']
    n_samples = probe_dict['n_samples']

    skip_indices = max(int(round(skip * sampling_rate)), 1) - 1

    # Working with ABSOLUTE voltage here
    voltage = np.abs(chan_voltage)
    first_thresh_index = np.zeros(voltage.shape[0], dtype="bool")

    # Find points above threshold where preceeding sample was below threshold (excludes first point)
    first_thresh_index[1:] = np.logical_and(
        voltage[1:] > threshold,
        voltage[0:-1] <= threshold)
    events: np.ndarray = np.nonzero(first_thresh_index)[0]  # Indices of threshold crossings in array

    # This is the raw total number of threshold crossings
    n_crossings: int = events.shape[0]  # np.count_nonzero(voltage > threshold)

    # Realign event times on min or max in align_window
    window = time_window_to_samples(align_window, sampling_rate)[0]
    for evt in range(0, events.size):
        start = max(0, events[evt] + window[0])  # Start maximally at 0 or event plus window
        stop = min(n_samples - 1, events[evt] + window[1])  # Stop minmally at event plus window or last index
        window_clip = chan_voltage[start:stop]
        max_index = np.argmax(window_clip)  # Gets FIRST max in window
        max_value = window_clip[max_index]
        min_index = np.argmin(window_clip)
        min_value = window_clip[min_index]
        if max_value > -1 * min_value:
            # Max value is greater, use max index
            events[evt] = start + max_index
        elif min_value < -1 * max_value:
            # Min value is greater, use min index
            events[evt] = start + min_index
        else:
            # Arbitrarily choose the negative going peak
            events[evt] = start + min_index

    # Remove events that follow preceeding valid event by less than skip_indices samples
    bad_index = np.zeros(events.shape[0], dtype="bool")
    last_n = 0
    for n in range(1, events.shape[0]):
        if events[n] - events[last_n] < skip_indices:
            bad_index[n] = True
        else:
            last_n = n
    events = events[~bad_index]

    return events, n_crossings