import numpy as np
from spikesorting_fullpursuit.test import gen_dataset


def generate_voltage_traces():
    n_chans = 4  # Number of channels to make in test dataset
    v_duration = 30  # Test dataset duration in seconds
    random_seed = 99  # Set seed of numpy random number generator for spike times
    neuron_templates = None  # Just use default pre-loaded template waveforms to generate spike voltage traces
    frequency_range = (300, 6000)  # Frequencies of dataset in Hz
    samples_per_second = 40000  # Sampling rate of 40kHz
    amplitude = 1  # Amplitude of 1 standard deviations of noise
    percent_shared_noise = 0.3  # Create shared noise across channels
    correlate1_2 = (
        0.10,
        10,
    )  # Set 10% of neuron 2 spikes to occur within 10 samples of a neuron 1 spike
    electrode_type = "tetrode"  # Choose pre-loaded electrode type of tetrode and all channels in neighborhood
    voltage_dtype = np.float32  # Create voltage array as float 32

    # Create the test dataset object
    test_data = gen_dataset.TestDataset(
        n_chans,
        v_duration,
        random_seed,
        neuron_templates,
        frequency_range,
        samples_per_second,
        amplitude,
        percent_shared_noise,
        correlate1_2,
        electrode_type,
        voltage_dtype,
    )

    # Generate the noise voltage array, without spikes, assigned to test_date.voltage_array
    test_data.gen_noise_voltage_array()

    # Specify the neurons' properties in the dataset
    firing_rates = np.array([90, 100])  # Firing rates
    template_inds = np.array([1, 0])  # Templates used for waveforms
    chan_scaling_factors = np.array(
        [[1.85, 2.25, 1.65, 0.5], [3.85, 3.95, 1.95, 3.7]]
    )  # Amplitude of neurons on each of the 4 channels
    refractory_win = 1.5e-3  # Set refractory period at 1.5 ms
    # Generate the test dataset by choosing spike times and adding them according to the specified properties
    test_data.gen_test_dataset(
        firing_rates,
        template_inds,
        chan_scaling_factors,
        refractory_win,
    )

    return test_data.Probe.voltage, test_data.actual_IDs


def generate_2d_voltage_array():
    voltages, timestamps = generate_voltage_traces()

    voltages = np.float32(voltages)

    raw_voltages = np.frombuffer(voltages, dtype=np.float32).reshape(
        (4, len(voltages[0]))
    )

    return raw_voltages, timestamps
