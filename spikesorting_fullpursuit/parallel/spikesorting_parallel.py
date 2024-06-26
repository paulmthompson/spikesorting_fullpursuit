import os
import sys

import spikesorting_fullpursuit.alignment.alignment
import spikesorting_fullpursuit.clustering.cluster_utils
import spikesorting_fullpursuit.clustering.isocut
import spikesorting_fullpursuit.clustering.kmeanspp
import spikesorting_fullpursuit.dim_reduce.pca
import spikesorting_fullpursuit.processing.clip_utils
import spikesorting_fullpursuit.processing.conversions
import spikesorting_fullpursuit.threshold.threshold
from spikesorting_fullpursuit.dim_reduce.pca import branch_pca_2_0
from spikesorting_fullpursuit.parallel.segment import (
    get_segment_onsets_and_offsets,
    adjust_segment_duration_and_overlap,
)
from spikesorting_fullpursuit.threshold.threshold import (
    single_thresholds,
    single_thresholds_and_samples,
)

sys.path.append(os.getcwd())

from shutil import rmtree
import mkl
import numpy as np
import multiprocessing as mp
import psutil
import time
from traceback import print_tb
from spikesorting_fullpursuit.parallel import segment_parallel
from spikesorting_fullpursuit.overlap import full_binary_pursuit
from spikesorting_fullpursuit.processing.wiener_filter import wiener_filter_segment
from spikesorting_fullpursuit.utils.memmap_close import MemMapClose
from spikesorting_fullpursuit.processing import zca, clip_utils
from spikesorting_fullpursuit.processing import artifact
from spikesorting_fullpursuit.processing.clip_utils import get_singlechannel_clips
from spikesorting_fullpursuit.processing.conversions import time_window_to_samples


def spike_sorting_settings_parallel(**kwargs):
    settings = {
        "filename": None,  # Not used by the sorter, but will store the desired name of the sorted file with the output neurons for user reference
        "sigma": 4.0,  # Threshold based on noise level
        "clip_width": [
            -15e-4,
            15e-4,
        ],  # Width of clip in seconds, used for clustering. Made symmetric with largest value for binary pursuit!
        "p_value_cut_thresh": 0.01,  # Statistical criterion for splitting clusters during iso-cut
        "match_cluster_size": False,  # Pairwise comparisons during isocut cluster merge testing are matched in sample size. This makes the test more robust to comparisons of small clusters with large ones but could result in an increased number of clusters
        "check_splits": False,  # Check isocut splits to ensure they are not doing anything that brings clusters closer together, which may indicate a bad cut point
        "segment_duration": 600,  # Seconds (None/Inf uses the entire recording) Can be increased but not decreased by sorter to be same size
        "segment_overlap": 120,  # Seconds of overlap between adjacent segments
        "do_branch_PCA": True,  # Use branch PCA method to split clusters
        "do_branch_PCA_by_chan": True,  # Repeat branch PCA on each single channel separately
        "do_overlap_recheck": True,  # Explicitly check if each spike is better accounted for as a sum of 2 spikes (templates)
        "filter_band": (
            300,
            8000,
        ),  # Sorting DOES NOT FILTER THE DATA! This is information for the sorter to use. Filter voltage as desired BEFORE sorting
        "do_ZCA_transform": True,  # Whether to perform ZCA whitening on voltage before sorting
        "check_components": 100,  # Number of PCs to check for clustering. None means all
        "max_components": 5,  # Max number of PCs to use to form the clustering space, out of those checked
        "min_firing_rate": 0.0,  # Neurons with fewer threshold crossings than satisfy this rate are removed
        "use_rand_init": True,  # If true, initial clustering uses at least some randomly chosen centers
        "add_peak_valley": False,  # Use peak valley in addition to PCs for clustering space
        "max_gpu_memory": None,  # Maximum bytes to tryto store on GPU during sorting. None means use as much memory as possible
        "save_1_cpu": True,  # If true, leaves one CPU not in use during parallel clustering
        "remove_artifacts": False,  # If true the artifact removal settings will be used to detect and zero out artifacts defined by the number of channels with simultaneous threshold crossing. Changes Probe.voltage in place.
        "artifact_cushion": None,  # Same format as clip_width defining a pre/post window for removal around artifacts. None defaults to same as clip_width
        "artifact_tol": 0,  # +/- tolerance, in samples, for counting an event as "simultaneous" across channels.
        "n_artifact_chans": 1.0,  # Amount of channels event must cross threshold on to be considered an artifact. Numbers <= 1 are treated as proportions of channels. Numbers >= 2 are treated as an absolute number of channels.
        "sort_peak_clips_only": True,  # If True, each sort only uses clips with peak on the main channel. Improves speed and accuracy but can miss clusters for low firing rate units on multiple channels
        "n_cov_samples": 100000,  # Number of random clips to use to estimate noise covariance matrix. Empirically and qualitatively, 100,000 tends to produce nearly identical results across attempts, 10,000 has some small variance.
        # e.g., sigma_bp_noise = 95%: 1.645, 97.5%: 1.96, 99%: 2.326; 99.9%: 3.090; 99.99% 3.719 NOTE: these are one sided
        "sigma_bp_noise": 3.719,  # Number of noise standard deviations an expected template match must exceed the decision boundary by. Otherwise it is a candidate for deletion or increased threshold. Higher values = lower false positives and higher false negatives
        "sigma_bp_CI": None,  # Number of noise standard deviations a template match must fall within for a spike to be added. np.inf or None ignores this parameter.
        "bp_chan_snr": None,  # SNR required for a template on a given channel to be used for binary pursuit. Channels lower than this are set to zero template signal.
        "absolute_refractory_period": 10e-4,  # Absolute refractory period expected between spikes of a single neuron. This is used in postprocesing.
        "get_adjusted_clips": False,  # Probably outdated and should be left as False. Returns spike clips after the waveforms of any potentially overlapping spikes have been removed.
        "max_binary_pursuit_clip_width_factor": 2.0,  # The factor by which binary pursuit template matching can be increased relative to clip width for clustering. The best values for clustering and template matching are not always the same.
        # Factor of 1.0 means use the same clip width. Less than 1 is invalid and will use the clip width.
        "verbose": False,  # Set to true for more things to be printed while the sorter runs
        "test_flag": False,  # Indicates a test run of parallel code that does NOT spawn multiple processes
        "log_dir": None,  # Directory where output logs will be saved as text files for each parallel process during clustering. Processes can not usually print to the main screen.
        "output_separability_metrics": True,  # Setting True will output the separability metrics dictionary for each segment. This contains a lot of information not currently used after sorting, such as noise covariance matrices and templates used by binary pursuit.
        "wiener_filter": True,  # Use wiener filter on data before binary pursuit.
        "wiener_filter_smoothing": 100,  # Hz or None for no smoothing
        "same_wiener": False,  # If true, compute Wiener filter over all channels at once, using the same filter for every channel
        "use_memmap": False,  # Will keep clips and voltages stored in memmap files (voltage is preloaded as needed into ram for faster processing)
        "memmap_dir": None,  # Location to memmap numpy arrays. None uses os.getcwd(). Should all be deleted after successfully running
        "memmap_fID": None,  # Optional identifier for naming memmap files for this specific file sort. Useful to prevent multiple simultaneous sorts from repeating file names and overwritting each other's data or causing an error
        "save_clips": True,  # Saves all discovered clips in output file. These can get VERY large, so it's optional. Can be recomputed from voltage for postprocessing.
        "parallel_zca": True,  # If True, do ZCA in parallel processes instead of serially. Parallel can load a LOT of voltage arrays/copies into memory
        "seg_work_order": False,  # If True, workers will be deployed in segment order to minimize memory usage. Otherwise they are in order of most crossings for greatest speed
    }

    for k in kwargs.keys():
        if k not in settings:
            raise TypeError(f"Unknown parameter key {k}.")
        settings[k] = kwargs[k]

    # Check validity of settings
    if settings["clip_width"][0] > 0.0:
        print(
            "First element of clip width: ",
            settings["clip_width"][0],
            " is positive. Using negative value of: ",
            -1 * settings["clip_width"][0],
        )
        settings["clip_width"][0] *= -1
    if settings["filter_band"][0] < 0 or settings["filter_band"][1] < 0:
        raise ValueError("Input setting 'filter_band' must be a positve numbers")
    if settings["artifact_cushion"] is None:
        settings["artifact_cushion"] = settings["clip_width"]
    if settings["artifact_cushion"][0] > 0.0:
        print(
            "First element of clip width: ",
            settings["artifact_cushion"][0],
            " is positive. Using negative value of: ",
            -1 * settings["artifact_cushion"][0],
        )
        settings["artifact_cushion"][0] *= -1
    try:
        settings["artifact_tol"] = np.abs(settings["artifact_tol"])
    except:
        raise ValueError(
            f"Setting 'artifact_tol' must be convertable to a positive numerical integer, \
            not {settings['artifact_tol']}."
        )

    # Check validity of most other settings
    for key in settings.keys():
        if key in [
            "do_branch_PCA",
            "do_branch_PCA_by_chan",
            "do_ZCA_transform",
            "use_rand_init",
            "add_peak_valley",
            "save_1_cpu",
            "sort_peak_clips_only",
            "get_adjusted_clips",
            "output_separability_metrics",
            "wiener_filter",
            "same_wiener",
            "use_memmap",
            "save_clips",
            "parallel_zca",
            "seg_work_order",
            "remove_artifacts",
            "match_cluster_size",
            "check_splits",
        ]:
            if type(settings[key]) != bool:
                if settings[key] != "False" and settings[key] != 0:
                    settings[key] = True
                else:
                    settings[key] = False
                print(
                    f"Input setting '{key}' was converted to \
                        boolean value: ",
                    settings[key],
                )
        if key in ["segment_duration", "segment_overlap"]:
            # Note actual relative values for overlap are checked in main function
            if settings[key] <= 0:
                raise ValueError(f"Input setting '{key}' must be a postive number")
        if key in ["check_components", "max_components"]:
            if settings[key] <= 0 or type(settings[key]) != int:
                raise ValueError(f"Input setting '{key}' must be a postive integer")
        if key in [
            "min_firing_rate",
            "sigma_bp_noise",
            "max_binary_pursuit_clip_width_factor",
        ]:
            if settings[key] < 0:
                print(
                    f"Input setting '{key}' was invalid and \
                        converted to zero"
                )
        if key in ["sigma_bp_CI"]:
            if settings[key] is None:
                settings[key] = np.inf
        if key in ["wiener_filter"]:
            if not settings["sort_peak_clips_only"]:
                print(
                    "Wiener filter must use sort peak clips only. \
                        Setting to True."
                )

            settings["sort_peak_clips_only"] = True
        if key == "memmap_dir":
            if settings["memmap_dir"] is None:
                settings["memmap_dir"] = os.getcwd()
        if key == "memmap_fID":
            if settings["memmap_fID"] is None:
                settings["memmap_fID"] = ""
            if not isinstance(settings["memmap_fID"], str):
                settings["memmap_fID"] = str(settings["memmap_fID"])
            if len(settings["memmap_fID"]) > 0:
                settings["memmap_fID"] = settings["memmap_fID"] + "_"

    return settings


# Some helpful functions for watching memory usage
def bytes_to_GiB_MiB(n_bytes):
    return n_bytes / 2**30, (n_bytes % (2**30)) / 2**20


def print_mem_usage(num=None):
    if num is None:
        num = ""
    process = psutil.Process(os.getpid())
    print(
        "Mem Usage num {0} is {1} GiB {2} MiB".format(
            num,
            bytes_to_GiB_MiB(process.memory_info().rss)[0],
            bytes_to_GiB_MiB(process.memory_info().rss)[1],
        ),
        flush=True,
    )
    return process.memory_info().rss


########################################################


def init_zca_voltage(
    seg_voltages_to_share,
    shapes_to_share,
    v_dtype,
):
    # Make voltages global so they don't need to be pickled to share with processes
    global zca_pool_dict
    zca_pool_dict = {}
    zca_pool_dict["share_voltage"] = []
    zca_pool_dict["voltage_dtype"] = v_dtype

    for seg in range(0, len(seg_voltages_to_share)):
        zca_pool_dict["share_voltage"].append(
            [seg_voltages_to_share[seg], shapes_to_share[seg]]
        )

    return


def init_zca_voltage_mmap(seg_voltages):
    # Make voltages global so they don't need to be pickled to share with processes
    global zca_pool_dict
    zca_pool_dict = {}
    zca_pool_dict["share_voltage"] = []

    for seg in range(0, len(seg_voltages)):
        zca_pool_dict["share_voltage"].append(seg_voltages[seg])

    return


def parallel_zca_and_threshold(
    seg_num,
    sigma,
    zca_cushion,
    n_samples,
):
    """
    Multiprocessing wrapper for single_thresholds_and_samples and
    preprocessing.get_noise_sampled_zca_matrix to get the ZCA voltage for each
    segment in parallel.

    Args:
        seg_num
        sigma (float): Number of standard deviations to use for thresholding
        zca_cushion (int): Number of samples to pad around each threshold
            crossing for ZCA whitening
            By default, this is 2 * the largest absolute value of
            the clip width
        n_samples (int): Number of noise samples to use for ZCA whitening

    Global variables:
    """
    mkl.set_num_threads(1)  # Only 1 thread per process
    # Get shared raw array voltage as numpy view
    seg_voltage = np.frombuffer(
        zca_pool_dict["share_voltage"][seg_num][0], dtype=zca_pool_dict["voltage_dtype"]
    ).reshape(zca_pool_dict["share_voltage"][seg_num][1])

    # Plug numpy view into required functions
    thresholds = single_thresholds(seg_voltage, sigma)
    zca_matrix = zca.get_noise_sampled_zca_matrix(
        seg_voltage, thresholds, sigma, zca_cushion, n_samples
    )

    # @ makes new copy
    zca_seg_voltage = (zca_matrix @ seg_voltage).astype(zca_pool_dict["voltage_dtype"])
    # Get thresholds for newly ZCA'ed voltage
    thresholds, seg_over_thresh = single_thresholds_and_samples(zca_seg_voltage, sigma)

    # Copy ZCA'ed segment voltage to the raw array buffer so we can re-use it for sorting
    # Doesn't need to be returned since its written to shared dictionary buffer
    np.copyto(seg_voltage, zca_seg_voltage)

    return thresholds, seg_over_thresh, zca_matrix


def parallel_zca_and_threshold_mmap(
    seg_num,
    sigma,
    zca_cushion,
    n_samples,
):
    """
    Multiprocessing wrapper for single_thresholds_and_samples and
    preprocessing.get_noise_sampled_zca_matrix to get the ZCA voltage for each
    segment in parallel.

    Args:
        seg_num
        sigma (float): Number of standard deviations to use for thresholding
        zca_cushion (int): Number of samples to pad around each threshold
            crossing for ZCA whitening
            By default, this is 2 * the largest absolute value of
            the clip width
        n_samples (int): Number of noise samples to use for ZCA whitening

    Global variables:
    """
    mkl.set_num_threads(1)  # Only 1 thread per process
    # Get shared raw array voltage as numpy view
    seg_voltage_mmap = MemMapClose(
        zca_pool_dict["share_voltage"][seg_num][0],
        dtype=zca_pool_dict["share_voltage"][seg_num][1],
        mode="r+",
        shape=zca_pool_dict["share_voltage"][seg_num][2],
    )
    # Copy to memory cause ZCA selection/indexing is crazy
    seg_voltage = segment_parallel.memmap_to_mem(seg_voltage_mmap)
    # Plug numpy memmap view into required functions
    thresholds = single_thresholds(seg_voltage, sigma)
    zca_matrix = zca.get_noise_sampled_zca_matrix(
        seg_voltage, thresholds, sigma, zca_cushion, n_samples
    )
    # @ makes new copy
    zca_seg_voltage = (zca_matrix @ seg_voltage).astype(
        zca_pool_dict["share_voltage"][seg_num][1]
    )
    # Get thresholds for newly ZCA'ed voltage
    thresholds, seg_over_thresh = single_thresholds_and_samples(zca_seg_voltage, sigma)

    # Copy ZCA'ed segment voltage to the memmap array buffer so we can re-use it for sorting
    # Doesn't need to be returned since its written to shared dictionary buffer
    np.copyto(seg_voltage_mmap, zca_seg_voltage)
    if isinstance(seg_voltage_mmap, np.memmap):
        seg_voltage_mmap.flush()
        seg_voltage_mmap._mmap.close()
        del seg_voltage_mmap

    return thresholds, seg_over_thresh, zca_matrix


def threshold_and_zca_voltage_parallel(
    seg_voltages,
    sigma,
    zca_cushion,
    n_samples=1e6,
    use_memmap=False,
):
    """
    Use parallel processing to get thresholds and ZCA voltage for each segment

    Args:
        seg_voltages (list): List of numpy arrays with shape
            (num_channels, num_samples). One array per segment.
        sigma (float): Number of standard deviations to use for thresholding
        zca_cushion (int): Number of samples to pad around
            each threshold crossing for ZCA whitening
            By default, this is 2 * the largest absolute value of the
            clip width
        n_samples (int): Number of noise samples to use for ZCA whitening
        use_memmap (bool): If True, use memmap files to store voltages and
            ZCA matrices. Otherwise use shared multiprocessing arrays.
    """
    n_threads = mkl.get_max_threads()  # Incoming number of threads
    n_processes = psutil.cpu_count(logical=True)  # Use maximum processors

    order_results = []
    # Run in main process so available in main
    if use_memmap:
        init_zca_voltage_mmap(seg_voltages)
        with mp.Pool(
            processes=n_processes,
            initializer=init_zca_voltage_mmap,
            initargs=(seg_voltages,),
        ) as pool:
            try:
                for seg in range(0, len(seg_voltages)):
                    order_results.append(
                        pool.apply_async(
                            parallel_zca_and_threshold_mmap,
                            args=(seg, sigma, zca_cushion, n_samples),
                        )
                    )
            finally:
                pool.close()
                pool.join()
    else:
        # Copy segment voltages to shared multiprocessing array
        seg_voltages_to_share = []
        shapes_to_share = []
        for seg in range(0, len(seg_voltages)):
            share_voltage = mp.RawArray(
                np.ctypeslib.as_ctypes_type(seg_voltages[seg].dtype),
                seg_voltages[seg].size,
            )
            share_voltage_np = np.frombuffer(
                share_voltage, dtype=seg_voltages[seg].dtype
            ).reshape(seg_voltages[seg].shape)
            np.copyto(share_voltage_np, seg_voltages[seg])
            seg_voltages_to_share.append(share_voltage)
            shapes_to_share.append(seg_voltages[seg].shape)

        # Run in main process so available in main
        init_zca_voltage(seg_voltages_to_share, shapes_to_share, seg_voltages[0].dtype)
        with mp.Pool(
            processes=n_processes,
            initializer=init_zca_voltage,
            initargs=(seg_voltages_to_share, shapes_to_share, seg_voltages[0].dtype),
        ) as pool:
            try:
                for seg in range(0, len(seg_voltages)):
                    order_results.append(
                        pool.apply_async(
                            parallel_zca_and_threshold,
                            args=(seg, sigma, zca_cushion, n_samples),
                        )
                    )
            finally:
                pool.close()
                pool.join()
    # Everything should be in segment order since pool async results were put in
    # list according to seg order. Read out results and return
    results_tuple = [x.get() for x in order_results]
    thresholds_list = [x[0] for x in results_tuple]
    seg_over_thresh_list = [x[1] for x in results_tuple]
    seg_zca_mats = [x[2] for x in results_tuple]

    mkl.set_num_threads(n_threads)  # Reset threads back

    return thresholds_list, seg_over_thresh_list, seg_zca_mats


def allocate_cpus_by_chan(samples_over_thresh):
    """Assign CPUs/threads according to number of threshold crossings,
    THIS IS EXTREMELY APPROXIMATE since it counts time points
    above threshold, and each spike will have more than one of these.
    Sort time also depends on number of units and other factors but this
    is a decent starting point without any other information available."""
    cpu_alloc = []
    median_crossings = np.median(samples_over_thresh)
    for magnitude in samples_over_thresh:
        if magnitude > 5 * median_crossings:
            cpu_alloc.append(2)
        else:
            cpu_alloc.append(1)

    return cpu_alloc


class NoSpikesError(Exception):
    # Dummy exception class soley for exiting the 'try' loop if there are no spikes
    pass


def print_process_info(title):
    print(title, flush=True)
    print("module name:", __name__, flush=True)
    print("parent process:", os.getppid(), flush=True)
    print("process id:", os.getpid(), flush=True)


def check_spike_alignment(
    multi_channel_clips,
    event_indices,
    neuron_labels,
    curr_chan_inds,
    settings,
):
    """
    Wavelet alignment can bounce back and forth based on noise blips if
    the spike waveform is nearly symmetric in peak/valley. This is all done
    in memory without memmap-ing because selection and copies is complex.

    Parameters
    ----------
    multi_channel_clips: np.ndarray | memmap
        2D matrix of clips where first dimension is length of event_indices
        and second dimension is clip for each channel
    event_indices: np.ndarray
        positions of clips in samples
    neuron_labels: np.ndarray[int]
        array of length of event indexes where each neuron label is the ID
        of the cluster it is currently assigned to
    curr_chan_inds: np.ndarray
        In each multichannel clip, these are the indices of the single
        channel of interest clip
    settings

    Returns
    -------

    """
    (
        templates,
        labels,
    ) = spikesorting_fullpursuit.processing.clip_utils.calculate_templates(
        multi_channel_clips[:, curr_chan_inds[0] : curr_chan_inds[1] + 1], neuron_labels
    )
    any_merged = False
    unit_inds_to_check = [x for x in range(0, len(templates))]
    previously_aligned_dict = {}
    while len(unit_inds_to_check) > 1:
        # Find nearest cross corr template matched pair
        best_corr = -np.inf
        best_shift = 0
        for i in range(0, len(unit_inds_to_check)):
            for j in range(i + 1, len(unit_inds_to_check)):
                t_ind_1 = unit_inds_to_check[i]
                t_ind_2 = unit_inds_to_check[j]
                cross_corr = np.correlate(
                    templates[t_ind_1], templates[t_ind_2], mode="full"
                )
                max_corr_ind = np.argmax(cross_corr)
                if cross_corr[max_corr_ind] > best_corr:
                    best_corr = cross_corr[max_corr_ind]
                    best_shift = max_corr_ind - cross_corr.shape[0] // 2
                    best_pair_inds = [t_ind_1, t_ind_2]

        # Get clips for best pair and optimally align them with each other
        select_n_1 = neuron_labels == labels[best_pair_inds[0]]
        select_n_2 = neuron_labels == labels[best_pair_inds[1]]
        clips_1 = multi_channel_clips[select_n_1, :][:, curr_chan_inds]
        clips_2 = multi_channel_clips[select_n_2, :][:, curr_chan_inds]

        # Align and truncate clips for best match pair
        if best_shift > 0:
            clips_1 = clips_1[:, best_shift:]
            clips_2 = clips_2[:, : -1 * best_shift]
        elif best_shift < 0:
            clips_1 = clips_1[:, :best_shift]
            clips_2 = clips_2[:, -1 * best_shift :]
        else:
            # No need to shift, or even check these further
            if clips_1.shape[0] >= clips_2.shape[0]:
                unit_inds_to_check.remove(best_pair_inds[1])
            else:
                unit_inds_to_check.remove(best_pair_inds[0])
            continue
        # Check if the main merges with its best aligned leftover
        combined_clips = np.vstack((clips_1, clips_2))
        pseudo_labels = np.ones(combined_clips.shape[0], dtype=np.int64)
        pseudo_labels[clips_1.shape[0] :] = 2
        scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca(
            combined_clips,
            settings["check_components"],
            settings["max_components"],
            add_peak_valley=settings["add_peak_valley"],
            curr_chan_inds=np.arange(0, combined_clips.shape[1]),
        )
        # pseudo_labels = isosplit6(scores)
        # pseudo_labels = isosplit6(scores, initial_labels=pseudo_labels)

        pseudo_labels = spikesorting_fullpursuit.clustering.isocut.merge_clusters(
            scores,
            pseudo_labels,
            split_only=False,
            merge_only=True,
            p_value_cut_thresh=settings["p_value_cut_thresh"],
            match_cluster_size=settings["match_cluster_size"],
            check_splits=settings["check_splits"],
        )

        if np.all(pseudo_labels == 1) or np.all(pseudo_labels == 2):
            any_merged = True
            if clips_1.shape[0] >= clips_2.shape[0]:
                # Align all neuron 2 clips with neuron 1 template
                event_indices[select_n_2] += -1 * best_shift
                unit_inds_to_check.remove(best_pair_inds[1])
                if best_pair_inds[1] in previously_aligned_dict:
                    for unit in previously_aligned_dict[best_pair_inds[1]]:
                        select_unit = neuron_labels == unit
                        event_indices[select_unit] += -1 * best_shift
                if best_pair_inds[0] not in previously_aligned_dict:
                    previously_aligned_dict[best_pair_inds[0]] = []
                previously_aligned_dict[best_pair_inds[0]].append(best_pair_inds[1])
            else:
                # Align all neuron 1 clips with neuron 2 template
                event_indices[select_n_1] += best_shift
                unit_inds_to_check.remove(best_pair_inds[0])
                # Check if any previous units are tied to this one and should
                # also shift
                if best_pair_inds[0] in previously_aligned_dict:
                    for unit in previously_aligned_dict[best_pair_inds[0]]:
                        select_unit = neuron_labels == unit
                        event_indices[select_unit] += best_shift
                # Make this unit follow neuron 1 in the event neuron 1 changes
                # in a future iteration
                if best_pair_inds[1] not in previously_aligned_dict:
                    previously_aligned_dict[best_pair_inds[1]] = []
                previously_aligned_dict[best_pair_inds[1]].append(best_pair_inds[0])
        else:
            unit_inds_to_check.remove(best_pair_inds[0])
            unit_inds_to_check.remove(best_pair_inds[1])

    return event_indices, any_merged


def spike_sort_item_parallel(
    data_dict,
    use_cpus,
    work_item,
    settings,
):
    """
    do_ZCA_transform, filter_band is not used here but prevents errors from passing kwargs.
    """
    # Initialize variables in case this exits on error
    crossings, neuron_labels = [], []
    exit_type = None

    def wrap_up():
        delete_clip_memmap()
        data_dict["results_dict"][work_item["ID"]] = [crossings, neuron_labels]
        data_dict["completed_items"].append(work_item["ID"])
        data_dict["exits_dict"][work_item["ID"]] = exit_type
        data_dict["completed_items_queue"].put(work_item["ID"])
        for cpu in use_cpus:
            data_dict["cpu_queue"].put(cpu)
        return

    def delete_clip_memmap():
        try:
            clip_fname = os.path.join(
                settings["memmap_dir"],
                f"{settings['memmap_fID']}clips_{str(work_item['ID'])}.bin",
            )
            if os.path.exists(clip_fname):
                os.remove(clip_fname)
        except:
            pass
        finally:
            return

    def create_nparray_from_raw_array(raw_array, dtype, shape):
        return np.frombuffer(raw_array, dtype=dtype).reshape(shape)

    try:
        # Print this process' errors and output to a file
        if not settings["test_flag"] and settings["log_dir"] is not None:
            move_stdout_to_logdir(settings, work_item)

            print_process_info(
                f"spike_sort_item_parallel item {work_item['ID']}, \
                channel {work_item['channel']}, \
                segment {work_item['seg_number'] + 1}."
            )

        # Setup threads and affinity based on use_cpus if not on mac OS
        if "win32" == sys.platform:
            proc = psutil.Process()  # get self pid
            # proc.cpu_affinity(use_cpus)
        if settings["test_flag"]:
            mkl.set_num_threads(8)
        else:
            mkl.set_num_threads(len(use_cpus))

        # Get the all the needed info for this work item
        # Functions that get this dictionary only ever use these items since
        # we separately extract the voltage and the neighbors
        item_dict = {
            "sampling_rate": data_dict["sampling_rate"],
            "n_samples": work_item["n_samples"],
            "thresholds": work_item["thresholds"],
            "v_dtype": data_dict["v_dtype"],
            "ID": work_item["ID"],
            "memmap_dir": settings["memmap_dir"],
            "memmap_fID": settings["memmap_fID"],
        }
        chan = work_item["channel"]

        voltage = create_nparray_from_raw_array(
            data_dict["segment_voltages"][work_item["seg_number"]][0],
            item_dict["v_dtype"],
            data_dict["segment_voltages"][work_item["seg_number"]][1],
        )

        neighbors = work_item["neighbors"]

        # if settings['verbose']:
        #     print_process_info("spike_sort_item_parallel item {0}, channel {1}, segment {2}.".format(work_item['ID'], work_item['channel'], work_item['seg_number']+1))
        #     print_mem_usage(1)

        skip = np.amax(np.abs(settings["clip_width"])) / 2
        align_window = [skip, skip]
        if settings["verbose"]:
            print("Identifying threshold crossings", flush=True)

        # Note that n_crossings is NOT just len(crossings)! It is the raw number
        # of threshold crossings. Values of crossings obey skip and align window.
        (
            crossings,
            n_crossings,
        ) = spikesorting_fullpursuit.threshold.threshold.identify_threshold_crossings(
            voltage[chan, :],
            item_dict["sampling_rate"],
            item_dict["n_samples"],
            item_dict["thresholds"][chan],
            skip=skip,
            align_window=align_window,
        )
        if crossings.size == 0:
            exit_type = "No crossings over threshold."
            # Raise error to force exit and wrap_up()
            crossings, neuron_labels = [], []
            raise NoSpikesError
        # Save this number for later
        settings["n_threshold_crossings"][chan] = n_crossings

        min_cluster_size = calculate_min_cluster_size(item_dict, settings)

        (
            _,
            _,
            clip_samples,
            _,
            curr_chan_inds,
        ) = spikesorting_fullpursuit.processing.clip_utils.get_windows_and_indices(
            settings["clip_width"],
            item_dict["sampling_rate"],
            chan,
            neighbors,
        )

        exit_type = "Found crossings"

        window, clip_width_s = time_window_to_samples(
            settings["clip_width"], item_dict["sampling_rate"]
        )

        single_channel_clips, valid_inds = get_singlechannel_clips(
            item_dict,
            voltage[chan, :],
            crossings,
            clip_width_s=clip_width_s,
            use_memmap=settings["use_memmap"],
        )
        crossings = crossings[valid_inds]

        # Realign spikes based on a common wavelet
        crossings = spikesorting_fullpursuit.alignment.alignment.wavelet_align_events(
            single_channel_clips,
            crossings,
            window,
            settings["filter_band"],
            item_dict["sampling_rate"],
        )

        (
            multi_channel_clips,
            valid_event_indices,
        ) = spikesorting_fullpursuit.processing.clip_utils.get_clips(
            item_dict,
            voltage,
            neighbors,
            crossings,
            clip_width_s=settings["clip_width"],
            use_memmap=settings["use_memmap"],
        )
        crossings = segment_parallel.keep_valid_inds([crossings], valid_event_indices)

        if settings["sort_peak_clips_only"]:
            (
                multi_channel_clips,
                crossings,
            ) = remove_clips_without_max_on_current_channel(
                multi_channel_clips,
                crossings,
                curr_chan_inds,
                item_dict,
                neighbors,
                neuron_labels,
                settings,
                voltage,
            )

        if crossings.size == 0:
            exit_type = "No crossings over threshold."
            # Raise error to force exit and wrap_up()
            crossings, neuron_labels = [], []
            raise NoSpikesError

        exit_type = "Found first clips"

        if settings["verbose"]:
            print("Start initial clustering and merge", flush=True)

        # Do initial single channel sort. Start with single channel only because
        # later branching can split things out using multichannel info, but it
        # can't put things back together again
        multi_channel_clips, crossings, neuron_labels = initial_channel_sort(
            chan,
            multi_channel_clips,
            crossings,
            curr_chan_inds,
            item_dict,
            neighbors,
            settings,
            voltage,
        )

        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)

        (
            cc_clip_width_s,
            clip_width_s,
        ) = spikesorting_fullpursuit.alignment.alignment.double_clip_width(
            settings["clip_width"],
            settings["sampling_rate"],
        )

        single_channel_clips, valid_inds = get_singlechannel_clips(
            item_dict,
            voltage[chan, :],
            crossings,
            clip_width_s=cc_clip_width_s,
        )
        crossings = crossings[valid_inds]
        neuron_labels = neuron_labels[valid_inds]

        crossings = (
            spikesorting_fullpursuit.alignment.alignment.align_events_with_template(
                single_channel_clips,
                neuron_labels,
                crossings,
                clip_width_s=settings["clip_width"],
                sampling_rate=settings["sampling_rate"],
            )
        )

        if isinstance(single_channel_clips, np.memmap):
            single_channel_clips._mmap.close()
            del single_channel_clips

        # Get clips from all channels
        (
            multi_channel_clips,
            valid_event_indices,
        ) = spikesorting_fullpursuit.processing.clip_utils.get_clips(
            item_dict,
            voltage,
            neighbors,
            crossings,
            clip_width_s=settings["clip_width"],
            use_memmap=settings["use_memmap"],
        )
        crossings, neuron_labels = segment_parallel.keep_valid_inds(
            [crossings, neuron_labels],
            valid_event_indices,
        )

        # Remove deviant clips *before* doing branch PCA to avoid getting clusters
        # of overlaps or garbage
        keep_clips = clip_utils.cleanup_clusters(
            multi_channel_clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1],
            neuron_labels,
        )
        crossings, neuron_labels = segment_parallel.keep_valid_inds(
            [crossings, neuron_labels],
            keep_clips,
        )
        if settings["use_memmap"]:
            # Need to recompute clips here because we can't get a memmap view
            if isinstance(multi_channel_clips, np.memmap):
                multi_channel_clips._mmap.close()
                del multi_channel_clips
            (
                multi_channel_clips,
                valid_event_indices,
            ) = spikesorting_fullpursuit.processing.clip_utils.get_clips(
                item_dict,
                voltage,
                neighbors,
                crossings,
                clip_width_s=settings["clip_width"],
                use_memmap=settings["use_memmap"],
            )
        else:
            multi_channel_clips = multi_channel_clips[keep_clips, :]

        # Single channel branch
        if curr_num_clusters.size > 1 and settings["do_branch_PCA"]:
            neuron_labels = branch_pca_2_0(
                neuron_labels,
                multi_channel_clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1],
                np.arange(0, curr_chan_inds.size),
                p_value_cut_thresh=settings["p_value_cut_thresh"],
                add_peak_valley=settings["add_peak_valley"],
                check_components=settings["check_components"],
                max_components=settings["max_components"],
                use_rand_init=settings["use_rand_init"],
                method="pca",
                match_cluster_size=settings["match_cluster_size"],
                check_splits=settings["check_splits"],
            )
            curr_num_clusters, n_per_cluster = np.unique(
                neuron_labels, return_counts=True
            )
            if settings["verbose"]:
                print(
                    "After SINGLE BRANCH",
                    curr_num_clusters.size,
                    "different clusters",
                    flush=True,
                )

        if settings["do_branch_PCA"]:
            # Remove deviant clips before doing branch PCA to avoid getting clusters
            # of overlaps or garbage, this time on full neighborhood
            keep_clips = clip_utils.cleanup_clusters(multi_channel_clips, neuron_labels)
            crossings, neuron_labels = segment_parallel.keep_valid_inds(
                [crossings, neuron_labels],
                keep_clips,
            )
            if settings["use_memmap"]:
                # Need to recompute clips here because we can't get a memmap view
                if isinstance(multi_channel_clips, np.memmap):
                    multi_channel_clips._mmap.close()
                    del multi_channel_clips
                (
                    multi_channel_clips,
                    valid_event_indices,
                ) = spikesorting_fullpursuit.processing.clip_utils.get_clips(
                    item_dict,
                    voltage,
                    neighbors,
                    crossings,
                    clip_width_s=settings["clip_width"],
                    use_memmap=settings["use_memmap"],
                )
            else:
                multi_channel_clips = multi_channel_clips[keep_clips, :]

        # Multi channel branch
        if data_dict["num_channels"] > 1 and settings["do_branch_PCA"]:
            neuron_labels = branch_pca_2_0(
                neuron_labels,
                multi_channel_clips,
                curr_chan_inds,
                p_value_cut_thresh=settings["p_value_cut_thresh"],
                add_peak_valley=settings["add_peak_valley"],
                check_components=settings["check_components"],
                max_components=settings["max_components"],
                use_rand_init=settings["use_rand_init"],
                method="pca",
                match_cluster_size=settings["match_cluster_size"],
                check_splits=settings["check_splits"],
            )
            curr_num_clusters, n_per_cluster = np.unique(
                neuron_labels, return_counts=True
            )
            if settings["verbose"]:
                print(
                    "After MULTI BRANCH",
                    curr_num_clusters.size,
                    "different clusters",
                    flush=True,
                )
        # Multi channel branch by channel
        if (
            data_dict["num_channels"] > 1
            and settings["do_branch_PCA_by_chan"]
            and settings["do_branch_PCA"]
        ):
            neuron_labels = branch_pca_2_0(
                neuron_labels,
                multi_channel_clips,
                curr_chan_inds,
                p_value_cut_thresh=settings["p_value_cut_thresh"],
                add_peak_valley=settings["add_peak_valley"],
                check_components=settings["check_components"],
                max_components=settings["max_components"],
                use_rand_init=settings["use_rand_init"],
                method="chan_pca",
                match_cluster_size=settings["match_cluster_size"],
                check_splits=settings["check_splits"],
            )
            curr_num_clusters, n_per_cluster = np.unique(
                neuron_labels, return_counts=True
            )
            if settings["verbose"]:
                print(
                    "After MULTI BRANCH by channel",
                    curr_num_clusters.size,
                    "different clusters",
                    flush=True,
                )

        # Delete any clusters under min_cluster_size before binary pursuit
        if settings["verbose"]:
            print(
                "Current smallest cluster has",
                np.amin(n_per_cluster),
                "spikes",
                flush=True,
            )

        if np.any(n_per_cluster < min_cluster_size):
            for l_ind in range(0, curr_num_clusters.size):
                if n_per_cluster[l_ind] < min_cluster_size:
                    keep_inds = ~(neuron_labels == curr_num_clusters[l_ind])
                    crossings = crossings[keep_inds]
                    neuron_labels = neuron_labels[keep_inds]
                    # We don't use clips after this point so only update crossings and labels
                    if settings["verbose"]:
                        print(
                            "Deleted cluster",
                            curr_num_clusters[l_ind],
                            "with",
                            n_per_cluster[l_ind],
                            "spikes",
                            flush=True,
                        )

        if neuron_labels.size == 0:
            exit_type = "No clusters over min_firing_rate."
            # Raise error to force exit and wrap_up()
            crossings, neuron_labels = [], []
            raise NoSpikesError

        exit_type = "Finished sorting clusters"

        # Realign spikes based on correlation with current cluster templates before doing binary pursuit
        (
            cc_clip_width_s,
            clip_width_s,
        ) = spikesorting_fullpursuit.alignment.alignment.double_clip_width(
            settings["clip_width"],
            settings["sampling_rate"],
        )

        single_channel_clips, valid_inds = get_singlechannel_clips(
            item_dict,
            voltage[chan, :],
            crossings,
            clip_width_s=cc_clip_width_s,
        )
        crossings = crossings[valid_inds]
        neuron_labels = neuron_labels[valid_inds]

        crossings = (
            spikesorting_fullpursuit.alignment.alignment.align_events_with_template(
                single_channel_clips,
                neuron_labels,
                crossings,
                clip_width_s=settings["clip_width"],
                sampling_rate=settings["sampling_rate"],
            )
        )

        if settings["verbose"]:
            print(
                "currently",
                np.unique(neuron_labels).size,
                "different clusters",
                flush=True,
            )

        # Map labels starting at zero and put labels in order
        spikesorting_fullpursuit.clustering.cluster_utils.reorder_labels(neuron_labels)

        if settings["verbose"]:
            print("Successfully completed item ", str(work_item["ID"]), flush=True)

        exit_type = "Success"
        # if settings['verbose']:
        #     print_process_info("spike_sort_item_parallel item {0}, channel {1}, segment {2}.".format(work_item['ID'], work_item['channel'], work_item['seg_number']+1))
        #     print_mem_usage("END")
    except NoSpikesError:
        if settings["verbose"]:
            print("No spikes to sort.")
        if settings["verbose"]:
            print("Successfully completed item ", str(work_item["ID"]), flush=True)
        exit_type = "Success"
    except Exception as err:
        exit_type = err
        print_tb(err.__traceback__)
        if settings["test_flag"]:
            raise  # Reraise any exceptions in test mode only
    finally:
        wrap_up()


def remove_clips_without_max_on_current_channel(
    multi_channel_clips,
    crossings,
    curr_chan_inds,
    item_dict,
    neighbors,
    neuron_labels,
    settings,
    voltage,
):
    """


    Parameters
    ----------
    multi_channel_clips: np.ndarray | memmap
        2D matrix of clips where first dimension is length of event_indices
        and second dimension is clip for each channel
    crossings: np.ndarray
        positions of clips in samples
    curr_chan_inds: np.ndarray
        In each multichannel clip, these are the indices of the single
        channel of interest clip
    item_dict:

    neighbors: np.ndarray of int
        Array of neighbors for channel of interest. Usually in
        numerical order (first channel is not necessarily channel
        of interest)
    neuron_labels: np.ndarray[int]
        array of length of event indexes where each neuron label is the ID
        of the cluster it is currently assigned to
    settings:

    voltage: np.ndarray
        channel x segment_samples array of raw voltage values

    Returns
    -------
    clips
    crossings
    """
    keep_clips = clip_utils.keep_max_on_main(multi_channel_clips, curr_chan_inds)
    crossings = crossings[keep_clips]
    if settings["use_memmap"]:
        # Need to recompute clips here because we can't get a memmap view
        if isinstance(multi_channel_clips, np.memmap):
            multi_channel_clips._mmap.close()
            del multi_channel_clips
        (
            multi_channel_clips,
            valid_event_indices,
        ) = spikesorting_fullpursuit.processing.clip_utils.get_clips(
            item_dict,
            voltage,
            neighbors,
            crossings,
            clip_width_s=settings["clip_width"],
            use_memmap=settings["use_memmap"],
        )
        crossings = segment_parallel.keep_valid_inds([crossings], valid_event_indices)
    else:
        multi_channel_clips = multi_channel_clips[keep_clips, :]

    curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
    if settings["verbose"]:
        print(
            "After keep max on main removed",
            np.count_nonzero(~keep_clips),
            "clips",
            flush=True,
        )
    return multi_channel_clips, crossings


def calculate_min_cluster_size(item_dict, settings):
    min_cluster_size = (
        np.floor(
            settings["min_firing_rate"]
            * item_dict["n_samples"]
            / item_dict["sampling_rate"]
        )
    ).astype(np.int64)
    if min_cluster_size < 1:
        min_cluster_size = 1
    if settings["verbose"]:
        print("Using minimum cluster size of", min_cluster_size, flush=True)
    return min_cluster_size


def move_stdout_to_logdir(settings, work_item):
    if sys.platform == "win32":
        sys.stdout = open(
            settings["log_dir"] + "\\SpikeSortItem" + str(work_item["ID"]) + ".out",
            "w",
        )
        sys.stderr = open(
            settings["log_dir"]
            + "\\SpikeSortItem"
            + str(work_item["ID"])
            + "_errors.out",
            "w",
        )
    else:
        sys.stdout = open(
            settings["log_dir"] + "/SpikeSortItem" + str(work_item["ID"]) + ".out",
            "w",
        )
        sys.stderr = open(
            settings["log_dir"]
            + "/SpikeSortItem"
            + str(work_item["ID"])
            + "_errors.out",
            "w",
        )


def initial_channel_sort(
    chan,
    multi_channel_clips,
    crossings,
    curr_chan_inds,
    item_dict,
    neighbors,
    settings,
    voltage,
):
    """

    Parameters
    ----------
    chan: int
        current channel of interest (ID of voltage channel)
    multi_channel_clips: np.ndarray | memmap
        2D matrix of clips where first dimension is length of event_indices
        and second dimension is clip for each channel
    crossings:
        positions of clips in samples
    curr_chan_inds: np.ndarray
        In each multichannel clip, these are the indices of the single
        channel of interest clip
    item_dict:

    neighbors: np.ndarray of int
        Array of neighbors for channel of interest. Usually in
        numerical order (first channel is not necessarily channel
        of interest)
    settings:

    voltage: np.ndarray
        channel x segment_samples array of raw voltage values

    Returns
    -------

    """

    if crossings.size <= 1:
        neuron_labels = np.zeros(1, dtype=np.int64)
        return multi_channel_clips, crossings, neuron_labels

    median_cluster_size = min(100, int(np.around(crossings.size / 1000)))

    # MUST SLICE curr_chan_inds to get a view instead of copy
    scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca(
        multi_channel_clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1],
        settings["check_components"],
        settings["max_components"],
        add_peak_valley=settings["add_peak_valley"],
        curr_chan_inds=np.arange(0, curr_chan_inds.size),
    )
    n_random = (
        max(100, np.around(crossings.size / 100)) if settings["use_rand_init"] else 0
    )
    neuron_labels = (
        spikesorting_fullpursuit.clustering.kmeanspp.initial_cluster_farthest(
            scores,
            median_cluster_size,
            n_random=n_random,
        )
    )
    # neuron_labels = isosplit6(scores)
    # neuron_labels = isosplit6(scores, initial_labels=neuron_labels)

    neuron_labels = spikesorting_fullpursuit.clustering.isocut.merge_clusters(
        scores,
        neuron_labels,
        split_only=False,
        p_value_cut_thresh=settings["p_value_cut_thresh"],
        match_cluster_size=settings["match_cluster_size"],
        check_splits=settings["check_splits"],
    )

    if settings["verbose"]:
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        print(
            "After first sort",
            curr_num_clusters.size,
            "different clusters",
            flush=True,
        )

    if settings["sort_peak_clips_only"]:
        (
            cc_clip_width_s,
            clip_width_s,
        ) = spikesorting_fullpursuit.alignment.alignment.double_clip_width(
            settings["clip_width"],
            settings["sampling_rate"],
        )

        single_channel_clips, valid_inds = get_singlechannel_clips(
            item_dict,
            voltage[chan, :],
            crossings,
            clip_width_s=cc_clip_width_s,
        )
        crossings = crossings[valid_inds]
        neuron_labels = neuron_labels[valid_inds]

        crossings = (
            spikesorting_fullpursuit.alignment.alignment.align_events_with_template(
                single_channel_clips,
                neuron_labels,
                crossings,
                clip_width_s=settings["clip_width"],
                sampling_rate=settings["sampling_rate"],
            )
        )

        window, clip_width_s = time_window_to_samples(
            settings["clip_width"], settings["sampling_rate"]
        )
        single_channel_clips, valid_inds = get_singlechannel_clips(
            item_dict,
            voltage[chan, :],
            crossings,
            clip_width_s=settings["clip_width"],
        )
        crossings = crossings[valid_inds]
        neuron_labels = neuron_labels[valid_inds]
        crossings = spikesorting_fullpursuit.alignment.alignment.align_templates(
            single_channel_clips,
            crossings,
            neuron_labels,
            window,
        )

        if isinstance(single_channel_clips, np.memmap):
            single_channel_clips._mmap.close()
            del single_channel_clips

        (
            multi_channel_clips,
            valid_event_indices,
        ) = spikesorting_fullpursuit.processing.clip_utils.get_clips(
            item_dict,
            voltage,
            neighbors,
            crossings,
            clip_width_s=settings["clip_width"],
            use_memmap=settings["use_memmap"],
        )
        crossings, neuron_labels = segment_parallel.keep_valid_inds(
            [crossings, neuron_labels],
            valid_event_indices,
        )

        scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca(
            multi_channel_clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1],
            settings["check_components"],
            settings["max_components"],
            add_peak_valley=settings["add_peak_valley"],
            curr_chan_inds=np.arange(0, curr_chan_inds.size),
        )
        n_random = (
            max(100, np.around(crossings.size / 100))
            if settings["use_rand_init"]
            else 0
        )

        neuron_labels = (
            spikesorting_fullpursuit.clustering.kmeanspp.initial_cluster_farthest(
                scores,
                median_cluster_size,
                n_random=n_random,
            )
        )
        # neuron_labels = isosplit6(scores)
        # neuron_labels = isosplit6(scores, initial_labels=neuron_labels)

        neuron_labels = spikesorting_fullpursuit.clustering.isocut.merge_clusters(
            scores,
            neuron_labels,
            split_only=False,
            p_value_cut_thresh=settings["p_value_cut_thresh"],
            match_cluster_size=settings["match_cluster_size"],
            check_splits=settings["check_splits"],
        )

        if settings["verbose"]:
            curr_num_clusters, n_per_cluster = np.unique(
                neuron_labels, return_counts=True
            )
            print(
                "After re-sort",
                curr_num_clusters.size,
                "different clusters",
                flush=True,
            )

    crossings, any_merged = check_spike_alignment(
        multi_channel_clips,
        crossings,
        neuron_labels,
        curr_chan_inds,
        settings,
    )
    if any_merged:
        # Resort based on new clip alignment
        if settings["verbose"]:
            print("Re-sorting after check spike alignment")

        if isinstance(multi_channel_clips, np.memmap):
            multi_channel_clips._mmap.close()
            del multi_channel_clips
        (
            clips,
            valid_event_indices,
        ) = spikesorting_fullpursuit.processing.clip_utils.get_clips(
            item_dict,
            voltage,
            neighbors,
            crossings,
            clip_width_s=settings["clip_width"],
            use_memmap=settings["use_memmap"],
        )
        crossings = segment_parallel.keep_valid_inds(
            [crossings],
            valid_event_indices,
        )

        scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca(
            multi_channel_clips[:, curr_chan_inds[0] : curr_chan_inds[-1] + 1],
            settings["check_components"],
            settings["max_components"],
            add_peak_valley=settings["add_peak_valley"],
            curr_chan_inds=np.arange(0, curr_chan_inds.size),
        )

        n_random = (
            max(100, np.around(crossings.size / 100))
            if settings["use_rand_init"]
            else 0
        )

        neuron_labels = (
            spikesorting_fullpursuit.clustering.kmeanspp.initial_cluster_farthest(
                scores,
                median_cluster_size,
                n_random=n_random,
            )
        )
        """
        Using isosplit6 here results in overclustering on testing. Not clear what the difference is
        """
        # neuron_labels = isosplit6(scores)
        # neuron_labels = isosplit6(scores, initial_labels=neuron_labels)

        neuron_labels = spikesorting_fullpursuit.clustering.isocut.merge_clusters(
            scores,
            neuron_labels,
            split_only=False,
            p_value_cut_thresh=settings["p_value_cut_thresh"],
            match_cluster_size=settings["match_cluster_size"],
            check_splits=settings["check_splits"],
        )

    if settings["verbose"]:
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        print("Currently", curr_num_clusters.size, "different clusters", flush=True)

    return multi_channel_clips, crossings, neuron_labels


def deploy_parallel_sort(
    manager,
    cpu_queue,
    cpu_alloc,
    work_items,
    init_dict,
    settings,
):
    """
    This function takes the basic pre-made inputs for calling
    spike_sort_item_parallel and handles deployment and deletion of the
    parallel processes for each work item and initialization of the global
    "data_dict" from the inputs of "init_dict".

    Parameters
    ----------
    manager
    cpu_queue
    cpu_alloc
    work_items: List
        List of dictionaries, one for each channel in a segment
        Entries of the dictionary are as follows:
            channel: int
                channel ID
            neighbors: np.ndarray
                array of channels (including channel of interest) nearby channel
            chan_neighbor_ind: int
                index of channel in neighbors array
            n_samples: segment_offsets[x] - segment_onsets[x],
            seg_number: x,
            index_window: [segment_onsets[x], segment_offsets[x]],
            overlap": settings["segment_overlap"],
            thresholds: thresholds_list[x],
    init_dict
    settings

    Returns
    -------

    """
    # Need to reset certain elements of init_dict, but NOT THE VOLTAGE!
    init_dict["results_dict"] = manager.dict()
    init_dict["completed_items"] = manager.list()
    init_dict["exits_dict"] = manager.dict()
    init_dict["completed_items"] = manager.list()
    completed_items_queue = manager.Queue(len(work_items))
    init_dict["completed_items_queue"] = completed_items_queue
    for x in range(0, len(work_items)):
        # Initializing keys for each result seems to prevent broken pipe errors
        init_dict["results_dict"][x] = None
        init_dict["exits_dict"][x] = None

    # Call init function to ensure data_dict is globally available before passing
    # it into each process
    if settings["use_memmap"]:
        init_data_dict_mmap(init_dict)
        # Attempt some memory management
        seg_voltage_users = [[] for x in range(0, len(data_dict["seg_v_files"]))]
    else:
        init_data_dict(init_dict)
    processes = []
    proc_item_index = []
    completed_items_index = 0
    process_errors_list = []
    print("Starting sorting pool")
    # Put the work items through the sorter
    for wi_ind, w_item in enumerate(work_items):
        w_item["ID"] = wi_ind  # Assign ID number in order of deployment
        # With timeout=None, this will block until sufficient cpus are available
        # as requested by cpu_alloc
        # NOTE: this currently goes in order, so if 1 CPU is available but next
        # work item wants 2, it will wait until 2 are available rather than
        # starting a different item that only wants 1 CPU...
        use_cpus = [cpu_queue.get(timeout=None) for x in range(cpu_alloc[wi_ind])]
        n_complete = len(data_dict["completed_items"])  # Do once to avoid race
        if n_complete > completed_items_index:
            for ci in range(completed_items_index, n_complete):
                print(
                    "Completed item",
                    work_items[data_dict["completed_items"][ci]]["ID"],
                    "from chan",
                    work_items[data_dict["completed_items"][ci]]["channel"],
                    "segment",
                    work_items[data_dict["completed_items"][ci]]["seg_number"] + 1,
                )

                print(
                    "Exited with status: ",
                    data_dict["exits_dict"][data_dict["completed_items"][ci]],
                )

                completed_items_index += 1
                if not settings["test_flag"]:
                    done_index = proc_item_index.index(data_dict["completed_items"][ci])
                    del proc_item_index[done_index]
                    processes[done_index].join()
                    processes[done_index].close()
                    del processes[done_index]
                    # Store error type if any
                    if (
                        data_dict["exits_dict"][data_dict["completed_items"][ci]]
                        != "Success"
                    ):
                        process_errors_list.append(
                            [
                                work_items[data_dict["completed_items"][ci]]["ID"],
                                data_dict["exits_dict"][
                                    data_dict["completed_items"][ci]
                                ],
                            ]
                        )

                if settings["use_memmap"]:
                    # Remove completed items from voltage users
                    seg_voltage_users[
                        work_items[data_dict["completed_items"][ci]]["seg_number"]
                    ].remove(work_items[data_dict["completed_items"][ci]]["ID"])

        if settings["use_memmap"]:
            # Check if shared voltage array exists for next worker and create if needed
            seg_voltage_users[w_item["seg_number"]].append(wi_ind)
            # Make sure voltage buffer is available for this process
            if w_item["seg_number"] not in data_dict["segment_voltages"].keys():
                # Get memmap segment voltage
                seg_voltage_mmap = MemMapClose(
                    data_dict["seg_v_files"][w_item["seg_number"]][0],
                    dtype=data_dict["seg_v_files"][w_item["seg_number"]][1],
                    mode="r",
                    shape=data_dict["seg_v_files"][w_item["seg_number"]][2],
                )
                # Create shared raw voltage array
                data_dict["segment_voltages"][w_item["seg_number"]] = [
                    mp.RawArray(
                        np.ctypeslib.as_ctypes_type(
                            data_dict["seg_v_files"][w_item["seg_number"]][1]
                        ),
                        seg_voltage_mmap.size,
                    ),
                    seg_voltage_mmap.shape,
                ]
                np_view = np.frombuffer(
                    data_dict["segment_voltages"][w_item["seg_number"]][0],
                    dtype=data_dict["seg_v_files"][w_item["seg_number"]][1],
                ).reshape(seg_voltage_mmap.shape)
                np.copyto(
                    np_view, seg_voltage_mmap
                )  # Copy segment voltage to voltage buffer
                if isinstance(seg_voltage_mmap, np.memmap):
                    seg_voltage_mmap._mmap.close()
                    del seg_voltage_mmap

        if not settings["test_flag"]:
            print(
                f"Starting item {wi_ind + 1}/{len(work_items)} \
                on CPUs {use_cpus} for channel {w_item['channel']} \
                segment {w_item['seg_number'] + 1}"
            )
            time.sleep(0.5)  # NEED SLEEP SO PROCESSES AREN'T MADE TOO FAST AND FAIL!!!

            proc_item_index.append(wi_ind)
            proc = mp.Process(
                target=spike_sort_item_parallel,
                args=(data_dict, use_cpus, w_item, settings),
            )
            processes.append(proc)
            proc.start()
        else:
            print(
                f"Starting item {wi_ind + 1}/{len(work_items)} \
                on CPUs {use_cpus} for channel {w_item['channel']} \
                segment {w_item['seg_number'] + 1}"
            )
            spike_sort_item_parallel(data_dict, use_cpus, w_item, settings)
            print("finished sort one item")

        if settings["use_memmap"]:
            # Delete voltage arrays not in use to try to save memory
            for seg_n, seg_u in enumerate(seg_voltage_users):
                if len(seg_u) == 0:
                    # No users for this segment
                    if seg_n in data_dict["segment_voltages"]:
                        # But seg is still holding voltage mempory
                        data_dict["segment_voltages"][seg_n] = None
                        del data_dict["segment_voltages"][seg_n]

    if not settings["test_flag"]:
        # Wait here a bit to print out items as they complete and to ensure
        # no process are left behind, as can apparently happen if you attempt to
        # join() too soon without being sure everything is finished (especially using queues)
        while completed_items_index < len(work_items) and not settings["test_flag"]:
            finished_item = completed_items_queue.get()
            try:
                done_index = proc_item_index.index(finished_item)
            except ValueError:
                # This item was already finished above so just clearing out
                # completed_items_queue
                continue
            print(
                "Completed item",
                finished_item + 1,
                "from chan",
                work_items[finished_item]["channel"],
                "segment",
                work_items[finished_item]["seg_number"] + 1,
            )

            print("Exited with status: ", data_dict["exits_dict"][finished_item])
            completed_items_index += 1

            del proc_item_index[done_index]
            processes[done_index].join()
            processes[done_index].close()
            del processes[done_index]
            # Store error type if any
            if data_dict["exits_dict"][finished_item] != "Success":
                process_errors_list.append(
                    [finished_item, data_dict["exits_dict"][finished_item]]
                )

            if settings["use_memmap"]:
                # Remove completed items from voltage users
                seg_voltage_users[work_items[finished_item]["seg_number"]].remove(
                    finished_item
                )
                # Delete voltage arrays not in use to try to save memory
                for seg_n, seg_u in enumerate(seg_voltage_users):
                    if len(seg_u) == 0:
                        # No users for this segment
                        if seg_n in data_dict["segment_voltages"]:
                            # But seg is still holding voltage mempory
                            data_dict["segment_voltages"][seg_n] = None
                            del data_dict["segment_voltages"][seg_n]

    # Make sure all the processes finish up and close even though they should
    # have finished above
    while len(processes) > 0:
        p = processes.pop()
        p.join()
        p.close()
        del p

    if settings["use_memmap"]:
        # Delete voltage arrays not in use to try to save memory
        for seg_n, seg_u in enumerate(seg_voltage_users):
            if len(seg_u) == 0:
                # No users for this segment
                if seg_n in data_dict["segment_voltages"]:
                    # But seg is still holding voltage mempory
                    del data_dict["segment_voltages"][seg_n]

    # Return possible errors during processes for display later
    return process_errors_list


def init_data_dict(init_dict=None):
    global data_dict
    data_dict = {}
    data_dict["segment_voltages"] = init_dict["segment_voltages"]
    if init_dict is not None:
        for k in init_dict.keys():
            data_dict[k] = init_dict[k]
    return


def init_data_dict_mmap(init_dict=None):
    global data_dict
    data_dict = {}
    data_dict["segment_voltages"] = {}
    data_dict["seg_v_files"] = init_dict["segment_voltages"]
    if init_dict is not None:
        for k in init_dict.keys():
            if k in ["segment_voltages", "seg_v_files"]:
                # Cannot overwrite these
                continue
            data_dict[k] = init_dict[k]
    return


def spike_sort_parallel(Probe, **kwargs):
    """Perform spike sorting algorithm using python multiprocessing module.
    See 'spike_sorting_settings_parallel' above for a list of allowable kwargs.

    Note: The temporary directory to store spike clips is created manually, not
    using the python tempfile module. Multiprocessing and tempfile seem to have
    some problems across platforms. For certain errors or keyboard interrupts
    the file may not be appropriately deleted. Before using the directory, the
    temp directory is deleted if it exists, so subsequent successful runs of
    sorting using the same directory will remove the temp directory.

    Note: Clips and voltages will be output in the data type Probe.v_dtype.
    However, most of the arithmetic is computed in np.float64. Clips are cast
    as np.float64 for determining PCs and cast back when done. All of binary
    pursuit is conducted as np.float32 for memory and GPU compatibility.
    See also:
    '
    """
    n_threads = mkl.get_max_threads()  # Incoming number of threads
    # Get our settings
    settings = spike_sorting_settings_parallel(**kwargs)
    # Check that filter is appropriate
    is_filter_within_nyquist(Probe, settings)

    # Check that Probe neighborhood function is appropriate. Otherwise it can
    # generate seemingly mysterious errors
    check_electrode_neighbor_sites(Probe)

    # For convenience, necessary to define clip width as negative for first entry
    if settings["clip_width"][0] > 0:
        settings["clip_width"] *= -1
    manager = mp.Manager()
    init_dict = {
        "num_channels": Probe.num_channels,
        "sampling_rate": Probe.sampling_rate,
        "v_dtype": Probe.v_dtype,
        "gpu_lock": manager.Lock(),
        "filter_band": settings["filter_band"],
    }

    create_log_dir(settings)

    # Perform artifact removal on input Probe voltage
    if settings["remove_artifacts"]:
        if Probe.num_channels == 1:
            raise ValueError(
                "Cannot do artifact detection on only 1 channel. Check input settings."
            )

        Probe = artifact.remove_artifacts(
            Probe,
            settings["sigma"],
            settings["artifact_cushion"],
            settings["artifact_tol"],
            settings["n_artifact_chans"],
        )
    # Convert segment duration and overlaps to indices from their values input
    # in seconds and adjust as needed
    adjust_segment_duration_and_overlap(Probe, settings)

    settings["n_threshold_crossings"] = np.zeros(Probe.num_channels)

    segment_offsets, segment_onsets = get_segment_onsets_and_offsets(Probe, settings)

    if settings["do_ZCA_transform"]:
        zca_cushion = (
            2 * np.ceil(np.amax(np.abs(settings["clip_width"])) * Probe.sampling_rate)
        ).astype(np.int64)

    try:
        # Everything here in try so that on error, we can delete all the memmap
        # files !
        # Build the sorting work items
        ############ THE VOLTAGE MEMMAP FILES ARE MADE HERE! #################
        seg_voltages = []
        init_dict[
            "segment_voltages"
        ] = []  # built differently below depending on use_memmap
        if settings["use_memmap"]:
            if settings["verbose"]:
                print(
                    "Attempting Memmap in directory {0}.".format(settings["memmap_dir"])
                )
        for x in range(0, len(segment_onsets)):
            # Slice over num_channels should keep same shape
            # Build list in segment order
            if settings["use_memmap"]:
                # Create list of filename, dtype, shape, for the memmaped voltages
                file_info = [
                    os.path.join(
                        settings["memmap_dir"],
                        "{0}volt_seg{1}.bin".format(settings["memmap_fID"], str(x)),
                    ),
                    Probe.v_dtype,
                    (Probe.num_channels, segment_offsets[x] - segment_onsets[x]),
                ]
                seg_voltages.append(file_info)
                init_dict["segment_voltages"].append(file_info)
                v_mmap = MemMapClose(
                    file_info[0], dtype=file_info[1], mode="w+", shape=file_info[2]
                )
                np.copyto(
                    v_mmap, Probe.voltage[:, segment_onsets[x] : segment_offsets[x]]
                )
                # Save memmap changes to disk
                if isinstance(v_mmap, np.memmap):
                    v_mmap.flush()
                    v_mmap._mmap.close()
                    del v_mmap
            else:
                seg_voltages.append(
                    Probe.voltage[:, segment_onsets[x] : segment_offsets[x]]
                )

        ############ THE VOLTAGE ARRAY BUFFERS IN MEMORY ARE MADE HERE! #################
        samples_over_thresh = []
        if (
            (not settings["test_flag"])
            and (settings["do_ZCA_transform"])
            and (settings["parallel_zca"])
        ):
            # If doing ZCA, voltage array buffers are made in
            # "threshold_and_zca_voltage_parallel" to update data and avoid
            # making twice
            # Use parallel processing to get zca voltage and thresholds
            if settings["verbose"]:
                print(
                    "Doing parallel ZCA transform and thresholding for",
                    len(segment_onsets),
                    "segments",
                )

            (
                thresholds_list,
                seg_over_thresh_list,
                seg_zca_mats,
            ) = threshold_and_zca_voltage_parallel(
                seg_voltages,
                settings["sigma"],
                zca_cushion,
                n_samples=1e6,
                use_memmap=settings["use_memmap"],
            )
            for x in seg_over_thresh_list:
                samples_over_thresh.extend(x)
            if not settings["use_memmap"]:
                init_dict["segment_voltages"] = zca_pool_dict["share_voltage"]
        else:
            thresholds_list = []
            seg_over_thresh_list = []
            seg_zca_mats = []
            for x in range(0, len(segment_onsets)):
                # Need to copy or else ZCA transforms will duplicate in overlapping
                # time segments. Copy happens during matrix multiplication
                if settings["use_memmap"]:
                    seg_voltage_mmap = MemMapClose(
                        seg_voltages[x][0],
                        dtype=seg_voltages[x][1],
                        mode="r+",
                        shape=seg_voltages[x][2],
                    )
                    # Copy to memory cause ZCA selction/indexing is crazy
                    seg_voltage = segment_parallel.memmap_to_mem(seg_voltage_mmap)
                else:
                    seg_voltage = seg_voltages[x]
                if settings["do_ZCA_transform"]:
                    if settings["verbose"]:
                        print(
                            "Finding voltage and thresholds for segment",
                            x + 1,
                            "of",
                            len(segment_onsets),
                        )

                    if settings["verbose"]:
                        print("Doing ZCA transform")

                    thresholds = single_thresholds(seg_voltage, settings["sigma"])

                    zca_matrix = zca.get_noise_sampled_zca_matrix(
                        seg_voltage,
                        thresholds,
                        settings["sigma"],
                        zca_cushion,
                        n_samples=1e6,
                    )
                    # Set seg_voltage to ZCA transformed voltage
                    # @ makes new copy
                    seg_voltage = (zca_matrix @ seg_voltage).astype(Probe.v_dtype)
                    seg_zca_mats.append(zca_matrix)
                if settings["use_memmap"] and settings["do_ZCA_transform"]:
                    # copy ZCA voltage to voltage memmap file if we changed it
                    np.copyto(seg_voltage_mmap, seg_voltage)
                elif settings["use_memmap"] and not settings["do_ZCA_transform"]:
                    pass  # Memmap data files already set and unchanged, do nothing
                else:
                    # Allocate shared voltage buffer. List is appended in SEGMENT ORDER
                    init_dict["segment_voltages"].append(
                        [
                            mp.RawArray(
                                np.ctypeslib.as_ctypes_type(Probe.v_dtype),
                                seg_voltage.size,
                            ),
                            seg_voltage.shape,
                        ]
                    )
                    np_view = np.frombuffer(
                        init_dict["segment_voltages"][x][0], dtype=Probe.v_dtype
                    ).reshape(
                        seg_voltage.shape
                    )  # Create numpy view
                    np.copyto(
                        np_view, seg_voltage
                    )  # Copy segment voltage to voltage buffer

                thresholds, seg_over_thresh = single_thresholds_and_samples(
                    seg_voltage, settings["sigma"]
                )
                thresholds_list.append(thresholds)
                samples_over_thresh.extend(seg_over_thresh)
                if settings["use_memmap"]:
                    if isinstance(seg_voltage_mmap, np.memmap):
                        seg_voltage_mmap.flush()
                        seg_voltage_mmap._mmap.close()
                        del seg_voltage_mmap

        work_items = []
        chan_neighbors = []
        chan_neighbor_inds = []
        for x in range(0, len(segment_onsets)):
            for chan in range(0, Probe.num_channels):
                # Ensure we just get neighbors once in case its complicated
                if x == 0:
                    chan_neighbors.append(Probe.get_neighbors(chan))
                    cn_ind = next(
                        (
                            idx[0]
                            for idx, val in np.ndenumerate(chan_neighbors[chan])
                            if val == chan
                        ),
                        None,
                    )
                    if cn_ind is None:
                        raise ValueError(
                            "Probe get_neighbors(chan) function must return a neighborhood that includes the channel 'chan'."
                        )
                    chan_neighbor_inds.append(cn_ind)

                work_items.append(
                    {
                        "channel": chan,
                        "neighbors": chan_neighbors[chan],
                        "chan_neighbor_ind": chan_neighbor_inds[chan],
                        "n_samples": segment_offsets[x] - segment_onsets[x],
                        "seg_number": x,
                        "index_window": [segment_onsets[x], segment_offsets[x]],
                        "overlap": settings["segment_overlap"],
                        "thresholds": thresholds_list[x],
                    }
                )
                # Check potential threshold problems, especially due to artifact removal
                if np.any(thresholds_list[x] == 0):
                    raise RuntimeError(
                        "At least 1 work item channel has a voltage threshold value of zero! \
                        Either a segment/channel has no data or has been made to have a median \
                        value of zero possibly due to inappropriate artifact detection parameters."
                    )

        if (not settings["test_flag"]) and (not settings["seg_work_order"]):
            if settings["log_dir"] is None:
                print(
                    "No log dir specified. Won't be able to see output from processes"
                )
            # Sort  work_items and samples_over_thresh by descending order of
            # samples over threshold. If testing we do not do this to keep
            # random numbers consistent with single channel sorter
            # Zip only returns tuple, so map it to a list
            samples_over_thresh, work_items = map(
                list,
                zip(
                    *[
                        [x, y]
                        for x, y in reversed(
                            sorted(
                                zip(samples_over_thresh, work_items),
                                key=lambda pair: pair[0],
                            )
                        )
                    ]
                ),
            )
        n_cpus = psutil.cpu_count(logical=True)
        if settings["save_1_cpu"]:
            n_cpus -= 1
        cpu_queue = manager.Queue(n_cpus)
        for cpu in range(n_cpus):
            cpu_queue.put(cpu)
        # cpu_alloc returned in order of samples_over_thresh/work_items
        cpu_alloc = allocate_cpus_by_chan(samples_over_thresh)
        # Make sure none exceed number available
        for x in range(0, len(cpu_alloc)):
            if cpu_alloc[x] > n_cpus:
                cpu_alloc[x] = n_cpus
        init_dict["cpu_queue"] = cpu_queue

        # Sort info is just settings with some extra stuff added for the output
        sort_info = settings
        (
            curr_chan_win,
            _,
        ) = spikesorting_fullpursuit.processing.conversions.time_window_to_samples(
            settings["clip_width"], Probe.sampling_rate
        )
        sort_info.update(
            {
                "n_samples": Probe.n_samples,
                "n_channels": Probe.num_channels,
                "n_samples_per_chan": curr_chan_win[1] - curr_chan_win[0],
                "sampling_rate": Probe.sampling_rate,
                "n_segments": len(segment_onsets),
                "seg_zca_mats": seg_zca_mats,
            }
        )
        if sort_info["output_separability_metrics"]:
            # Initialize elements for separability metrics from each segment
            sort_info["separability_metrics"] = [
                [] for x in range(0, sort_info["n_segments"])
            ]

        if settings["wiener_filter"]:
            wiener_vals = {
                "do_branch_PCA": False,
                "do_branch_PCA_by_chan": False,
                "check_components": 20,
                "max_components": 5,
                "min_firing_rate": 0.1,
                "use_rand_init": False,
                "sort_peak_clips_only": True,
                "sigma": settings["sigma"],
            }
            wiener_settings = {}
            for key in settings:
                if key in wiener_vals.keys():
                    wiener_settings[key] = wiener_vals[key]
                else:
                    wiener_settings[key] = settings[key]
            if settings["verbose"]:
                print("Start clustering for Wiener filter templates.")
            process_errors_list_wf = deploy_parallel_sort(
                manager,
                cpu_queue,
                cpu_alloc,
                work_items,
                init_dict,
                wiener_settings,
            )

            # Set threads/processes back to normal for Wiener filter and filter
            # each voltage segment
            mkl.set_num_threads(n_threads)
            if settings["verbose"]:
                print("Starting segment-wise Wiener filter")
            for seg_number in range(0, len(segment_onsets)):
                if settings["verbose"]:
                    print(
                        f"Start Wiener filter on segment {seg_number + 1}/{len(segment_onsets)}"
                    )
                # This will overwrite the segment voltage data!
                filtered_voltage = wiener_filter_segment(
                    work_items,
                    data_dict,
                    seg_number,
                    sort_info,
                    Probe.v_dtype,
                    use_memmap=settings["use_memmap"],
                )
                # Need to recompute the thresholds for the Wiener filtered data
                thresholds = single_thresholds(filtered_voltage, settings["sigma"])
                for wi in work_items:
                    if wi["seg_number"] == seg_number:
                        wi["thresholds"] = thresholds

        # Re-deploy parallel clustering now using the Wiener filtered voltage
        process_errors_list = deploy_parallel_sort(
            manager,
            cpu_queue,
            cpu_alloc,
            work_items,
            init_dict,
            settings,
        )

        # Set threads/processes back to normal now that we are done
        mkl.set_num_threads(n_threads)
        sort_data = []
        # Run binary pursuit for each segment using the discovered templates
        for seg_number in range(0, len(segment_onsets)):
            if settings["verbose"]:
                print(
                    f"Start full binary pursuit on segment \
                    {seg_number + 1}/{len(segment_onsets)}"
                )

            # Determine the set of work items for this segment
            seg_w_items = [w for w in work_items if w["seg_number"] == seg_number]

            seg_data = full_binary_pursuit.full_binary_pursuit(
                seg_w_items,
                data_dict,
                seg_number,
                sort_info,
                Probe.v_dtype,
                overlap_ratio_threshold=2,
                absolute_refractory_period=settings["absolute_refractory_period"],
                kernels_path=None,
                max_gpu_memory=settings["max_gpu_memory"],
                use_memmap=settings["use_memmap"],
                output_clips=settings["save_clips"],
            )
            sort_data.extend(seg_data)

        # Re-print any errors so more visible at the end of sorting
        if settings["wiener_filter"]:
            for pe in process_errors_list_wf:
                print("Wiener filter item number", pe[0], "had the following error:")
                print("            ", pe[1])
        for pe in process_errors_list:
            print("Item number", pe[0], "had the following error:")
            print("            ", pe[1])
    except:
        raise
    finally:
        if settings["use_memmap"]:
            # Delete voltage memmap files. Clips should be delete during cluster
            for x in range(0, len(seg_voltages)):
                try:
                    if os.path.exists(seg_voltages[x][0]):
                        os.remove(seg_voltages[x][0])
                except:
                    pass

    if settings["verbose"]:
        print("Done.")
    return sort_data, work_items, sort_info


def create_log_dir(settings):
    if settings["log_dir"] is not None:
        if os.path.exists(settings["log_dir"]):
            rmtree(settings["log_dir"])
            time.sleep(0.5)  # NEED SLEEP SO CAN DELETE BEFORE RECREATING!!!
        os.makedirs(settings["log_dir"])


def check_electrode_neighbor_sites(Probe):
    try:
        check_neighbors = Probe.get_neighbors(0)
    except:
        raise ValueError("Input Probe object must have a valid get_neighbors() method.")
    if type(check_neighbors) != np.ndarray:
        raise ValueError(
            "Probe get_neighbors() method must return a numpy ndarray of \
                dtype np.int64."
        )
    elif check_neighbors.dtype != np.int64:
        print(check_neighbors.dtype)
        raise ValueError(
            "Probe get_neighbors() method must return a \
                numpy ndarray of dtype np.int64."
        )
    elif np.any(np.diff(check_neighbors) <= 0):
        raise ValueError(
            "Probe get_neighbors() method must return \
                neighbor channels IN ORDER without duplicates."
        )


def is_filter_within_nyquist(Probe, settings):
    if (
        settings["filter_band"][0] > Probe.sampling_rate / 2
        or settings["filter_band"][1] > Probe.sampling_rate / 2
    ):
        raise ValueError(
            "Input setting 'filter_band' exceeds Nyquist limit for sampling rate of",
            Probe.sampling_rate,
        )
