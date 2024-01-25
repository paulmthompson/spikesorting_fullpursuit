import numpy as np
from scipy import signal

from spikesorting_fullpursuit.processing.clip_utils import (
    calculate_templates,
    get_singlechannel_clips,
)
from spikesorting_fullpursuit.processing.conversions import time_window_to_samples


def align_events_with_template(
    probe_dict,
    chan_voltage,
    neuron_labels,
    event_indices,
    clip_width_s,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the input data for ONE channel and computes the cross correlation
    of each spike with each template on the channel USING SINGLE CHANNEL CLIPS
    ONLY.  The spike time is then aligned with the peak cross correlation lag.
    This outputs new event indices reflecting this alignment, that can then be
    used to input into final sorting, as in cluster sharpening.

    Args:
        item_dict = {'sampling_rate',
                     'n_samples',
                     'thresholds': 1D numpy array of thresholds for each channel
                     'v_dtype',
                     'ID',
                     'memmap_dir',
                     'memmap_fID'
        chan_voltage: 1D numpy array of voltage values for single channel
        neuron_labels: 1D ndarray of dtype int64
            Numerical labels indicating the membership of
            each event_indices (spike clip index) as unique neuron.
        event_indices: 1D numpy array of indices of threshold crossings
        clip_width_s: list of two floats, time window in seconds to align

    Returns:
        event_indices: 1D numpy array of indices of threshold crossings
        neuron_labels: 1D numpy array of neuron labels
        valid_inds: 1D numpy array of booleans, which specify
            which of event indices are valid from input
    """

    sampling_rate = probe_dict["sampling_rate"]

    window, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    # Create clips twice as wide as current clip width, IN SAMPLES, for better cross corr
    cc_clip_width = [0, 0]
    cc_clip_width[0] = 2 * window[0] / sampling_rate
    cc_clip_width[1] = 2 * (window[1] - 1) / sampling_rate
    # Find indices within extra wide clips that correspond to the original clipwidth for template
    temp_index = [0, 0]
    temp_index[0] = -1 * min(int(round(clip_width_s[0] * sampling_rate)), 0)
    temp_index[1] = (
        2 * temp_index[0] + max(int(round(clip_width_s[1] * sampling_rate)), 1) + 1
    )  # Add one so that last element is included

    clips, valid_inds = get_singlechannel_clips(
        probe_dict, chan_voltage, event_indices, clip_width_s=cc_clip_width
    )
    event_indices = event_indices[valid_inds]
    neuron_labels = neuron_labels[valid_inds]
    templates, labels = calculate_templates(
        clips[:, temp_index[0] : temp_index[1]], neuron_labels
    )

    # First, align all clips with their own template
    for c in range(0, clips.shape[0]):
        cross_corr = np.correlate(
            clips[c, :],
            templates[np.nonzero(labels == neuron_labels[c])[0][0]],
            mode="valid",
        )
        event_indices[c] += np.argmax(cross_corr) - int(temp_index[0])

    return event_indices, neuron_labels, valid_inds


def align_events_with_best_template(
    probe_dict,
    chan_voltage,
    neuron_labels,
    event_indices,
    clip_width_s,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the input data for ONE channel and computes the cross correlation
    of each spike with each template on the channel USING SINGLE CHANNEL CLIPS
    ONLY.  The spike time is then aligned with the peak cross correlation lag.
    This outputs new event indices reflecting this alignment, that can then be
    used to input into final sorting, as in cluster sharpening.

    Args:
        item_dict = {'sampling_rate',
                     'n_samples',
                     'thresholds': 1D numpy array of thresholds for each channel
                     'v_dtype',
                     'ID',
                     'memmap_dir',
                     'memmap_fID'
        chan_voltage: 1D numpy array of voltage values for single channel
        neuron_labels: 1D ndarray of dtype int64
            Numerical labels indicating the membership of
            each event_indices (spike clip index) as unique neuron.
        event_indices: 1D numpy array of indices of threshold crossings
        clip_width_s: list of two floats, time window in seconds to align

    Returns:
        event_indices: 1D numpy array of indices of threshold crossings
        neuron_labels: 1D numpy array of neuron labels
        valid_inds: 1D numpy array of booleans, which specify
            which of event indices are valid from input
    """

    sampling_rate = probe_dict["sampling_rate"]

    window, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    clips, valid_inds = get_singlechannel_clips(
        probe_dict, chan_voltage, event_indices, clip_width_s=clip_width_s
    )
    event_indices = event_indices[valid_inds]
    neuron_labels = neuron_labels[valid_inds]
    overlaps = np.zeros(event_indices.size, dtype="bool")
    templates, labels = calculate_templates(clips, neuron_labels)
    templates = [(t) / np.amax(np.abs(t)) for t in templates]
    window = np.abs(window)
    center = max(window)

    # Align all clips with best template
    for c in range(0, clips.shape[0]):
        best_peak = -np.inf
        best_shift = 0
        for temp_ind in range(0, len(templates)):
            cross_corr = np.correlate(clips[c, :], templates[temp_ind], mode="full")
            max_ind = np.argmax(cross_corr)
            if cross_corr[max_ind] > best_peak:
                best_peak = cross_corr[max_ind]
                shift = max_ind - center - window[0]
                if shift <= -window[0] // 2 or shift >= window[1] // 2:
                    overlaps[c] = True
                    continue
                best_shift = shift
        event_indices[c] += best_shift
    event_indices = event_indices[~overlaps]
    neuron_labels = neuron_labels[~overlaps]

    return event_indices, neuron_labels, valid_inds


def align_templates(
    probe_dict,
    chan_voltage,
    neuron_labels,
    event_indices,
    clip_width_s,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns templates to each other and shift their event indices accordingly.

    This function determines the template for each cluster and
    then asks whether each unit has a template with larger peak
    or valley. All templates are then aligned such that their maximum
    absolute value is centered on the clip width.

    Args:
        item_dict = {'sampling_rate',
                     'n_samples',
                     'thresholds': 1D numpy array of thresholds for each channel
                     'v_dtype',
                     'ID',
                     'memmap_dir',
                     'memmap_fID'
        chan_voltage: 1D numpy array of voltage values for single channel
        neuron_labels: 1D ndarray of dtype int64
            Numerical labels indicating the membership of
            each event_indices (spike clip index) as unique neuron.
        event_indices: 1D numpy array of indices of threshold crossings
        clip_width_s: list of two floats, time window in seconds to align

    Returns:
        event_indices: 1D numpy array of indices of threshold crossings
        neuron_labels: 1D numpy array of neuron labels
        valid_inds: 1D numpy array of booleans, which specify
            which of event indices are valid from input
    """

    sampling_rate = probe_dict["sampling_rate"]

    window, clip_width_s = time_window_to_samples(clip_width_s, sampling_rate)
    clips, valid_inds = get_singlechannel_clips(
        probe_dict, chan_voltage, event_indices, clip_width_s=clip_width_s
    )
    event_indices = event_indices[valid_inds]
    neuron_labels = neuron_labels[valid_inds]
    window = np.abs(window)
    templates, labels = calculate_templates(clips, neuron_labels)

    for t_ind in range(0, len(templates)):
        t = templates[t_ind]
        t_select = neuron_labels == labels[t_ind]
        min_t = np.abs(np.amin(t))
        max_t = np.abs(np.amax(t))
        if max_t > min_t:
            # Align everything on peak
            shift = np.argmax(t)
        else:
            # Align everything on valley
            shift = np.argmin(t)
        event_indices[t_select] += shift - window[0] - 1

    return event_indices, neuron_labels, valid_inds


def wavelet_align_events(
    clips,
    event_indices,
    window,
    band_width,
    sampling_rate,
) -> np.ndarray:
    """
    Takes the input data for ONE channel and computes the cross correlation
    of each spike with each template on the channel USING SINGLE CHANNEL CLIPS
    ONLY.  The spike time is then aligned with the peak cross correlation lag.
    This outputs new event indices reflecting this alignment, that can then be
    used to input into final sorting, as in cluster sharpening.

    Args:
        clips
        event_indices: 1D numpy array of indices of threshold crossings
        window:
        band_width: Frequency band of bandpass filter
        sampling_rate

    Returns:
        event_indices: 1D numpy array of indices of threshold crossings
            aligned to peak of wavelet
            These can be deleted, so the length of event_indicies input
            may not be the same as output
    """

    overlaps = np.zeros(event_indices.size, dtype="bool")
    # Create a mexican hat central template, centered on the current clip width
    window = np.abs(window)
    center = max(window)
    temp_scales = []
    scale = 1
    # Minimum oscillation that will fit in this clip width
    min_win_freq = 1.0 / ((window[1] + window[0]) / sampling_rate)
    align_band_width = [band_width[0], band_width[1]]
    align_band_width[0] = max(min_win_freq, align_band_width[0])

    # Find center frequency of wavelet Fc. Uses the method in PyWavelets
    # central_frequency function
    central_template = signal.ricker(2 * center + 1, scale)
    index = np.argmax(np.abs(np.fft.fft(central_template)[1:])) + 2
    if index > len(central_template) / 2:
        index = len(central_template) - index + 2
    Fc = 1.0 / (central_template.shape[0] / (index - 1))

    # Start scale at max bandwidth
    scale = Fc * sampling_rate / align_band_width[1]
    # Build scaled templates for multiple of two frequencies within band width
    pseudo_frequency = Fc / (scale * (1 / sampling_rate))
    while pseudo_frequency >= align_band_width[0]:
        # Clips have a center and are odd, so this will match
        central_template = signal.ricker(2 * center + 1, scale)
        temp_scales.append(central_template)
        scale *= 2
        pseudo_frequency = Fc / (scale * (1 / sampling_rate))

    if len(temp_scales) == 0:
        # Choose single template at center of frequency band
        scale = (
            Fc
            * sampling_rate
            / (align_band_width[0] + (align_band_width[1] - align_band_width[0]))
        )
        central_template = signal.ricker(2 * center + 1, scale)
        temp_scales.append(central_template)

    # Align all waves on the mexican hat central template
    for c in range(0, clips.shape[0]):
        best_peak = -np.inf
        # First find the best frequency (here 'template') for this clip
        for temp_ind in range(0, len(temp_scales)):
            cross_corr = np.convolve(clips[c, :], temp_scales[temp_ind], mode="full")
            max_ind = np.argmax(cross_corr)
            min_ind = np.argmin(cross_corr)

            if cross_corr[max_ind] > best_peak:
                curr_peak = cross_corr[max_ind]
            elif -1.0 * cross_corr[min_ind] > best_peak:
                curr_peak = -1.0 * cross_corr[min_ind]

            if curr_peak > best_peak:
                best_temp_ind = temp_ind
                best_peak = curr_peak
                best_corr = cross_corr
                best_max = max_ind
                best_min = min_ind

        # Now use the best frequency convolution to align by weighting the clip
        # values by the convolution
        if -best_corr[best_min] > best_corr[best_max]:
            # Dip in best corr is greater than peak, so invert it so we can
            # use following logic assuming working from peak
            best_corr *= -1
            best_max, best_min = best_min, best_max

        prev_min_ind = best_max
        while prev_min_ind > 0:
            prev_min_ind -= 1
            if best_corr[prev_min_ind] >= best_corr[prev_min_ind + 1]:
                prev_min_ind += 1
                break

        prev_max_ind = prev_min_ind
        while prev_max_ind > 0:
            prev_max_ind -= 1
            if best_corr[prev_max_ind] <= best_corr[prev_max_ind + 1]:
                prev_max_ind += 1
                break

        next_min_ind = best_max
        while next_min_ind < best_corr.shape[0] - 1:
            next_min_ind += 1
            if best_corr[next_min_ind] >= best_corr[next_min_ind - 1]:
                next_min_ind -= 1
                break

        next_max_ind = next_min_ind
        while next_max_ind < best_corr.shape[0] - 1:
            next_max_ind += 1
            if best_corr[next_max_ind] <= best_corr[next_max_ind - 1]:
                next_max_ind -= 1
                break

        # Weighted average from 1 cycle before to 1 cycle after peak
        avg_win = np.arange(prev_max_ind, next_max_ind + 1)
        # Weighted by convolution values
        corr_weights = np.abs(best_corr[avg_win])
        best_arg = np.average(avg_win, weights=corr_weights)
        best_arg = np.around(best_arg).astype(np.int64)

        shift = best_arg - center - window[0]
        if shift <= -window[0] or shift >= window[1]:
            # If optimal shift is finding a different spike beyond window,
            # delete this spike as it violates our dead time between spikes
            overlaps[c] = True
            continue
        event_indices[c] += shift
    event_indices = event_indices[~overlaps]

    return event_indices
