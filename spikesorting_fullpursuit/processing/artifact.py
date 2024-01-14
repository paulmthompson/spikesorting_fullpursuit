import numpy as np

from spikesorting_fullpursuit.parallel.segment_parallel import median_threshold


def remove_artifacts(Probe, sigma, artifact_cushion, artifact_tol, n_artifact_chans):
    """
    Zero voltage for threshold crossings within artifact_tol samples that
    cross threshold on >= artifact_chans number of chans. Zero'ing goes from
    first thresh crossing - clip_width[0] to last threshold
    crossing + clip_width[1]. The voltage is modified in place for the
    voltage stored in Probe.voltage and the Probe is returned for clarity.
    """
    # Catch common input types/errors
    if Probe.num_channels == 1:
        print(
            "Attempting artifact detection on only 1 channel is not allowed! Skipping"
        )
        return Probe
    if not isinstance(artifact_cushion, list):
        artifact_cushion = [artifact_cushion]
    if len(artifact_cushion) == 1:
        artifact_cushion[0] = np.abs(artifact_cushion[0])
        artifact_cushion.append(artifact_cushion[0])
        artifact_cushion[0] *= -1
    elif len(artifact_cushion) == 2:
        if artifact_cushion[0] > 0:
            artifact_cushion[0] *= -1
    else:
        raise ValueError(
            "artifact_cushion must be a single number or a list of 1 or 2 numbers, but {0} was given.".format(
                artifact_cushion
            )
        )
    artifact_tol = int(round(artifact_tol))
    if n_artifact_chans <= 0:
        raise ValueError(
            "Invalid value for n_artifact_chans {0}.".format(n_artifact_chans)
        )
    if n_artifact_chans <= 1:
        n_artifact_chans *= Probe.num_channels
    n_artifact_chans = int(round(n_artifact_chans))
    if n_artifact_chans <= 1:
        raise ValueError(
            "Inalid value of {0} computed for n_artifact_chans. Number must compute to greater than 1 channel.".format(
                n_artifact_chans
            )
        )
    skip_indices = [
        min(int(round(artifact_cushion[0] * Probe.sampling_rate)), 0),
        max(int(round(artifact_cushion[1] * Probe.sampling_rate)), 0),
    ]

    # need to find thresholds for all channels first
    thresholds = median_threshold(Probe.voltage, sigma)

    # Try to save some memory here by doing one channel at a time and tracking in one big array
    # Working with ABSOLUTE voltage here
    total_chan_crossings = np.zeros((Probe.n_samples,), dtype=np.uint16)
    for chan in range(0, Probe.num_channels):
        voltage = np.abs(Probe.voltage[chan, :])
        first_thresh_index = np.zeros(voltage.shape[0], dtype="bool")
        # Find points above threshold where preceeding sample was below threshold (excludes first point)
        first_thresh_index[1:] = np.logical_and(
            voltage[1:] > thresholds[chan], voltage[0:-1] <= thresholds[chan]
        )
        events = np.nonzero(first_thresh_index)[0]

        # Go through each event start on this channel
        for evt in range(0, events.size):
            # Add this event +/- artifact_tol to total crossings (+1 for even slicing on either end)
            total_chan_crossings[
                max(0, events[evt] - artifact_tol) : min(
                    Probe.n_samples, events[evt] + artifact_tol + 1
                )
            ] += 1

    # Go through all artifacts and zero out their voltage window
    artifacts = np.nonzero(total_chan_crossings >= n_artifact_chans)[0]
    print(
        "Found {0} total ARTIFACTS to remove using {1} channels within {2} indices.".format(
            artifacts.size, n_artifact_chans, artifact_tol
        )
    )
    for atf in range(0, artifacts.size):
        # find when this event ends by going under threshold on artifact chans
        t = artifacts[atf]
        stop_ind = artifacts[atf]
        while t < Probe.n_samples:
            if (
                np.count_nonzero(np.abs(Probe.voltage[:, t]) > thresholds)
                < n_artifact_chans
            ):
                stop_ind = t
                break
            t += 1
        # Zero out the voltage in this artifact window
        Probe.voltage[
            :,
            max(0, artifacts[atf] + skip_indices[0]) : min(
                Probe.n_samples, stop_ind + skip_indices[1] + 1
            ),
        ] = 0

    return Probe
