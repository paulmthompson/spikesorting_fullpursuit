import numpy as np

# setting path
import sys

sys.path.append("..")

from generate_voltage_traces import generate_2d_voltage_array

from spikesorting_fullpursuit.threshold.threshold import (
    single_thresholds,
    identify_threshold_crossings,
)


def test_threshold_dimensions():
    voltages, timestamps = generate_2d_voltage_array()

    sigma = 4.0
    threshold_list = single_thresholds(voltages, 4.0)

    assert len(threshold_list) == voltages.shape[0]


def test_threshold_counts():
    sigma = 4.0
    chan = 0
    sampling_rate = 40000
    n_samples = 40000 * 30

    voltages, timestamps = generate_2d_voltage_array()

    threshold_list = single_thresholds(voltages, 4.0)
    for chan in range(voltages.shape[0]):
        (
            crossings,
            n_crossings,
        ) = identify_threshold_crossings(
            voltages[chan, :], sampling_rate, n_samples, threshold_list[chan]
        )

        print(n_crossings)
