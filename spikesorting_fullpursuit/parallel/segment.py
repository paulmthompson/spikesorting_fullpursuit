import numpy as np

import spikesorting_fullpursuit.processing

"""
These functions are for dealing with "segments"
Segments are periods of time for ephys data that should overlap
These are processed sequentially.

"""

def get_segment_onsets_and_offsets(Probe, settings):
    """

    Parameters
    ----------
    Probe
    settings

    Returns
    -------
    segment_offsets
    segment_onsets
    """
    segment_onsets = []
    segment_offsets = []
    curr_onset = 0
    while curr_onset < Probe.n_samples:
        segment_onsets.append(curr_onset)
        segment_offsets.append(
            min(curr_onset + settings["segment_duration"], Probe.n_samples)
        )
        if segment_offsets[-1] >= Probe.n_samples:
            break
        curr_onset += settings["segment_duration"] - settings["segment_overlap"]
    print("Using ", len(segment_onsets), "segments per channel for sorting.")
    return segment_offsets, segment_onsets


def adjust_segment_duration_and_overlap(Probe, settings):
    if (
        (settings["segment_duration"] is None)
        or (settings["segment_duration"] == np.inf)
        or (settings["segment_duration"] * Probe.sampling_rate >= Probe.n_samples)
    ):
        settings["segment_overlap"] = 0
        settings["segment_duration"] = Probe.n_samples
    else:
        if settings["segment_overlap"] is None:
            # If segment is specified with no overlap, use minimal overlap that
            # will not miss spikes on the edges
            clip_samples = (
                spikesorting_fullpursuit.processing.conversions.time_window_to_samples(
                    settings["clip_width"], Probe.sampling_rate
                )[0]
            )
            settings["segment_overlap"] = int(3 * (clip_samples[1] - clip_samples[0]))
        elif settings["segment_overlap"] <= 0:
            # If segment is specified with no overlap, use minimal overlap that
            # will not miss spikes on the edges
            clip_samples = (
                spikesorting_fullpursuit.processing.conversions.time_window_to_samples(
                    settings["clip_width"], Probe.sampling_rate
                )[0]
            )
            settings["segment_overlap"] = int(3 * (clip_samples[1] - clip_samples[0]))
        else:
            settings["segment_overlap"] = int(
                np.ceil(settings["segment_overlap"] * Probe.sampling_rate)
            )
        input_duration_seconds = settings["segment_duration"]
        settings["segment_duration"] = int(
            np.floor(settings["segment_duration"] * Probe.sampling_rate)
        )
        if settings["segment_overlap"] >= settings["segment_duration"]:
            raise ValueError("Segment overlap must be <= segment duration.")
        # Minimum number of segments at current segment duration and overlap
        # needed to cover all samples. Using floor will round to find the next
        # multiple that is >= the input segment duration.
        n_segs = np.floor(
            (Probe.n_samples - settings["segment_duration"])
            / (settings["segment_duration"] - settings["segment_overlap"])
        )
        # Modify segment duration to next larger multiple of recording duration
        # given fixed, unaltered input overlap duration
        settings["segment_duration"] = int(
            np.ceil(
                (Probe.n_samples + n_segs * settings["segment_overlap"]) / (n_segs + 1)
            )
        )

        print(
            "Input segment duration was rounded from",
            input_duration_seconds,
            "up to",
            settings["segment_duration"] / Probe.sampling_rate,
            "seconds to make segments equal length.",
        )
