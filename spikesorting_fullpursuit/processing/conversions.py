import copy


def time_window_to_samples(
        time_window_s: list,
        sampling_rate) -> tuple[list[int], list]:
    """
    Converts a two element list time window in seconds to a corresponding two
    element list window in units of samples.  Assumes the window is centered
    on a time and therefore the first element MUST be negative or it will be
    converted to a negative number. Second element has 1 added so that
    sample_window[1] is INCLUSIVELY SLICEABLE without adjustment. Also
    returns a copy of the input time_window which may have had its first
    element's sign inverted.

    """

    new_time_window_s = copy.copy(time_window_s)
    if new_time_window_s[0] > 0:
        new_time_window_s[0] *= -1
    sample_window = [0, 0]
    sample_window[0] = min(int(round(new_time_window_s[0] * sampling_rate)), 0)
    sample_window[1] = max(int(round(new_time_window_s[1] * sampling_rate)), 1) + 1  # Add one so that last element is included

    return sample_window, new_time_window_s
