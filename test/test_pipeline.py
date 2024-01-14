import sys
import numpy as np
import pickle
import spikesorting_fullpursuit as fbp
from spikesorting_fullpursuit import electrode
from spikesorting_fullpursuit.parallel.spikesorting_parallel import spike_sort_parallel
from spikesorting_fullpursuit.postprocessing import WorkItemSummary

"""
"""


def test_spike_sort_parallel():
    volt_fname = "test_voltage.npy"
    fname_ground_truth = "test_ground_truth.pickle"
    save_fname = "sorted_demo.pickle"
    log_dir = save_fname[0:-7] + "_std_logs"  # Set default log name to match save name

    np.random.seed(99)

    # Setup the sorting parameters dictionary.
    spike_sort_args = {
        "sigma": 4.0,  # Threshold based on noise level
        "clip_width": [-10e-4, 10e-4],  # Width of clip in seconds
        "p_value_cut_thresh": 0.01,
        "segment_duration": 300,
        "segment_overlap": 150,
        "do_branch_PCA": True,
        "do_branch_PCA_by_chan": True,
        "do_overlap_recheck": True,
        "filter_band": (300, 6000),
        "do_ZCA_transform": True,
        "check_components": 20,
        "max_components": 5,
        "min_firing_rate": 0.1,
        "use_rand_init": False,
        "add_peak_valley": False,
        "max_gpu_memory": 0.1 * (1024 * 1024 * 1024),
        "save_1_cpu": True,
        "sort_peak_clips_only": True,
        "n_cov_samples": 20000,
        "sigma_bp_noise": 2.326,
        "sigma_bp_CI": 12.0,
        "absolute_refractory_period": 10e-4,
        "get_adjusted_clips": False,
        "max_binary_pursuit_clip_width_factor": 1.0,
        "verbose": True,
        "test_flag": True,
        "log_dir": log_dir,
    }

    # Load the numpy voltage array
    raw_voltage = np.load(volt_fname)
    raw_voltage = np.float32(raw_voltage)
    # Load ground truth data

    with open(fname_ground_truth, "rb") as fp:
        ground_truth = pickle.load(fp)

    # Create the electrode object that specifies neighbor function for our current
    # tetrode test dataset
    Probe = electrode.SingleTetrode(sampling_rate=40000, voltage_array=raw_voltage)
    # We need to filter our voltage before passing it to the spike sorter. Just
    # use the one in Probe class
    Probe.bandpass_filter_parallel(
        spike_sort_args["filter_band"][0], spike_sort_args["filter_band"][1]
    )

    print("Start sorting")
    sort_data, work_items, sort_info = spike_sort_parallel(Probe, **spike_sort_args)

    # First step in automated post-processing
    # Set a few variables that can allow easy detection of units that are poor
    absolute_refractory_period = 10e-4  # Refractory period (in ms) will be used to determine potential violations in sorting accuracy
    # Max allowable ratio between refractory period violations and maximal bin of ACG. Units that violate will be deleted. Setting to >= 1. allows all units
    max_mua_ratio = 1.0
    min_snr = 0  # Minimal SNR a unit must have to be included in post-processing
    min_overlapping_spikes = 0.75  # Percentage of spikes required with nearly identical spike times in adjacent segments for them to combine in stitching

    # Create the work_summary postprocessing object
    work_summary = WorkItemSummary(
        sort_data,
        work_items,
        sort_info,
        absolute_refractory_period=absolute_refractory_period,
        max_mua_ratio=max_mua_ratio,
        min_snr=min_snr,
        min_overlapping_spikes=min_overlapping_spikes,
        verbose=False,
    )

    # No segments in the demo (segment_duration > duration of synthetic data) but done as example
    work_summary.stitch_segments()

    # Summarize the sorted output data into dictionaries by time segment.
    work_summary.summarize_neurons_by_seg()

    # Finally summarize neurons across channels (combining and removing duplicate
    # neurons across space) to get a list of sorted "neurons"
    neurons = work_summary.summarize_neurons_across_channels(
        overlap_ratio_threshold=np.inf, min_segs_per_unit=1, remove_clips=False
    )

    # Print out some basic information about our sorted units like number of spikes, firing rate, SNR, proportion MUA ISI violations
    print("Found", len(neurons), "total units with properties:")
    fmtL = "Unit: {:.0f} on chans {}; n spikes = {:.0f}; FR = {:.0f}; Dur = {:.0f}; SNR = {:.2f}; MUA = {:.2f}; TolInds = {:.0f}"
    for ind, n in enumerate(neurons):
        print_vals = [
            ind,
            n["channel"],
            n["spike_indices"].size,
            n["firing_rate"],
            n["duration_s"],
            n["snr"]["average"],
            n["fraction_mua"],
            n["duplicate_tol_inds"],
        ]
        print(fmtL.format(*print_vals))

    assert neurons[0]["spike_indices"].size == 2926
    assert neurons[1]["spike_indices"].size == 2769

    # Match the ground truth units to the sorted neurons with the most true positives
    test_match_to_neurons = {}
    for test_num in range(0, len(ground_truth)):
        max_true_positives = -np.inf
        for unit_num in range(0, len(neurons)):
            overlapping_spike_bool = (
                fbp.analyze_spike_timing.find_overlapping_spike_bool(
                    ground_truth[test_num],
                    neurons[unit_num]["spike_indices"],
                    overlap_tol=10,
                )
            )
            true_positives = np.count_nonzero(overlapping_spike_bool)
            if true_positives > max_true_positives:
                max_true_positives = true_positives
                test_match_to_neurons[test_num] = unit_num

    for test_num in range(0, len(ground_truth)):
        print(
            "Matched actual unit",
            test_num,
            "to sorted neuron",
            test_match_to_neurons[test_num],
        )

    # Print true positive and false discoveries for best matching to ground truth neuron 1
    ground_truth_unit = 0
    tol_inds = 10  # Match within a tolerance of 10 time samples
    overlapping_spike_bool = fbp.analyze_spike_timing.find_overlapping_spike_bool(
        ground_truth[ground_truth_unit],
        neurons[test_match_to_neurons[ground_truth_unit]]["spike_indices"],
        overlap_tol=tol_inds,
    )
    true_positives = np.count_nonzero(overlapping_spike_bool)

    print(
        "False discoveries are",
        100
        * (
            neurons[test_match_to_neurons[ground_truth_unit]]["spike_indices"].size
            - true_positives
        )
        / neurons[test_match_to_neurons[ground_truth_unit]]["spike_indices"].size,
    )
    print(
        "True positives are",
        100 * true_positives / ground_truth[ground_truth_unit].size,
    )

    # Print true positive and false discoveries for best matching to ground truth neuron 2
    ground_truth_unit = 1
    tol_inds = 10  # Match within a tolerance of 10 time samples
    overlapping_spike_bool = fbp.analyze_spike_timing.find_overlapping_spike_bool(
        ground_truth[ground_truth_unit],
        neurons[test_match_to_neurons[ground_truth_unit]]["spike_indices"],
        overlap_tol=tol_inds,
    )
    true_positives = np.count_nonzero(overlapping_spike_bool)

    print(
        "False discoveries are",
        100
        * (
            neurons[test_match_to_neurons[ground_truth_unit]]["spike_indices"].size
            - true_positives
        )
        / neurons[test_match_to_neurons[ground_truth_unit]]["spike_indices"].size,
    )
    print(
        "True positives are",
        100 * true_positives / ground_truth[ground_truth_unit].size,
    )
