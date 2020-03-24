import numpy as np
from spikesorting_python.src import segment
from spikesorting_python.src import preprocessing
from spikesorting_python.src import sort
from spikesorting_python.src import overlap
from spikesorting_python.src import binary_pursuit
from spikesorting_python.src import consolidate
import warnings
from copy import copy



def spike_sorting_settings(**kwargs):
    settings = {}

    settings['sigma'] = 4.0 # Threshold based on  noise level
    settings['verbose'] = False
    # settings['verbose_merge'] = False
    # settings['threshold_type'] = "absolute"
    # settings['sharpen'] = True
    settings['clip_width'] = [-6e-4, 10e-4]# Width of clip in seconds
    # settings['compute_noise'] = False # Estimate noise per cluster
    # settings['remove_false_positives'] = True # Remove false positives? Requires compute_noise = true
    settings['do_branch_PCA'] = True # Use branch pca method to split clusters
    # settings['preprocess'] = True
    settings['filter_band'] = (300, 6000)
    # settings['compute_lfp'] = False
    # settings['lfp_filter'] = (25, 500)
    settings['do_ZCA_transform'] = True
    settings['use_rand_init'] = True # Initial clustering uses at least some randomly chosen centers
    settings['add_peak_valley'] = False # Use peak valley in addition to PCs for sorting
    settings['check_components'] = None # Number of PCs to check. None means all, else integer
    settings['max_components'] = 10 # Max number to use, of those checked, integer
    settings['min_firing_rate'] = 1. # Neurons that first less than this threshold (spk/s) are removed
    settings['p_value_cut_thresh'] = 0.05
    settings['do_binary_pursuit'] = True
    settings['use_GPU'] = True # Force algorithms to run on the CPU rather than the GPU
    settings['max_gpu_memory'] = None # Use as much memory as possible
    settings['segment_duration'] = None # Seconds (nothing/Inf uses the entire recording)
    settings['segment_overlap'] = None # Seconds of overlap between adjacent segments
    settings['cleanup_neurons'] = False # Remove garbage at the end
    # settings['random_seed'] = None # The random seed to use (or nothing, if unspecified)

    for k in kwargs.keys():
        if k not in settings:
            raise TypeError("Unknown parameter key {0}.".format(k))
        settings[k] = kwargs[k]

    return settings


"""
    Since alignment is biased toward down neurons, up can be shifted. """
def check_upward_neurons(clips, event_indices, neuron_labels, curr_chan_inds,
                            clip_width, sampling_rate):
    templates, labels = segment.calculate_templates(clips[:, curr_chan_inds], neuron_labels)
    window, clip_width = segment.time_window_to_samples(clip_width, sampling_rate)
    center_index = -1 * min(int(round(clip_width[0] * sampling_rate)), 0)
    units_shifted = []
    for t_ind, temp in enumerate(templates):
        if np.amax(temp) > np.abs(np.amin(temp)):
            # Template peak is greater than absolute valley so realign on max
            label_ind = neuron_labels == labels[t_ind]
            event_indices[label_ind] += np.argmax(clips[label_ind, :][:, curr_chan_inds], axis=1) - int(center_index)
            units_shifted.append(labels[t_ind])

    return event_indices, units_shifted


def branch_pca_2_0(neuron_labels, clips, curr_chan_inds, p_value_cut_thresh=0.01,
                    add_peak_valley=False, check_components=None,
                    max_components=None, use_rand_init=True, method='pca'):
    """
    """
    neuron_labels_copy = np.copy(neuron_labels)
    clusters_to_check = [ol for ol in np.unique(neuron_labels_copy)]
    next_label = int(np.amax(clusters_to_check) + 1)
    while len(clusters_to_check) > 0:
        curr_clust = clusters_to_check.pop()
        curr_clust_bool = neuron_labels_copy == curr_clust
        clust_clips = clips[curr_clust_bool, :]
        if clust_clips.ndim == 1:
            clust_clips = np.expand_dims(clust_clips, 0)
        if clust_clips.shape[0] == 1:
            # Only one spike so don't try to sort
            continue
        median_cluster_size = min(100, int(np.around(clust_clips.shape[0] / 1000)))

        # Re-cluster and sort using only clips from current cluster
        if method.lower() == 'pca':
            scores = preprocessing.compute_pca(clust_clips, check_components, max_components,
                        add_peak_valley=add_peak_valley, curr_chan_inds=curr_chan_inds)
        elif method.lower() == 'chan_pca':
            scores = preprocessing.compute_pca_by_channel(clust_clips, curr_chan_inds,
                        check_components, max_components, add_peak_valley=add_peak_valley)
        else:
            raise ValueError("Sharpen method must be either 'pca', or 'chan_pca'.")
        n_random = max(100, np.around(clust_clips.shape[0] / 100)) if use_rand_init else 0
        clust_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        clust_labels = sort.merge_clusters(scores, clust_labels,
                        p_value_cut_thresh=p_value_cut_thresh)
        new_labels = np.unique(clust_labels)
        if new_labels.size > 1:
            # Found at least one new cluster within original so reassign labels
            for nl in new_labels:
                temp_labels = neuron_labels_copy[curr_clust_bool]
                temp_labels[clust_labels == nl] = next_label
                neuron_labels_copy[curr_clust_bool] = temp_labels
                clusters_to_check.append(next_label)
                next_label += 1

    return neuron_labels_copy


def spike_sort_segment(Probe, seg_dict, settings):
    """
    """
    skip = np.amax(np.abs(settings['clip_width'])) / 2
    align_window = [skip, skip]
    if settings['verbose']: print("Identifying threshold crossings")
    crossings = segment.identify_threshold_crossings(Probe, seg_dict['thresholds'], skip=skip, align_window=align_window)
    min_cluster_size = (np.floor(settings['min_firing_rate'] * Probe.n_samples / Probe.sampling_rate)).astype(np.int64)
    if min_cluster_size < 1:
        min_cluster_size = 1
    if settings['verbose']: print("Using minimum cluster size of", min_cluster_size)

    labels = [[] for x in range(0, Probe.num_electrodes)]
    waveforms  = [[] for x in range(0, Probe.num_electrodes)]
    new_waveforms = [[] for x in range(0, Probe.num_electrodes)]
    for chan in range(0, Probe.num_electrodes):
        if settings['verbose']: print("Working on electrode ", chan)
        median_cluster_size = min(100, int(np.around(crossings[chan].size / 1000)))
        if settings['verbose']: print("Getting clips")
        clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=settings['clip_width'], thresholds=seg_dict['thresholds'])
        crossings[chan] = segment.keep_valid_inds([crossings[chan]], valid_event_indices)
        _, _, clip_samples, _, curr_chan_inds = segment.get_windows_and_indices(settings['clip_width'], Probe.sampling_rate, chan, Probe.get_neighbors(chan))

        if settings['verbose']: print("Start initial clustering and merge")
        # Do initial single channel sort
        scores = preprocessing.compute_pca(clips[:, curr_chan_inds],
                    settings['check_components'], settings['max_components'], add_peak_valley=settings['add_peak_valley'],
                    curr_chan_inds=np.arange(0, curr_chan_inds.size))
        n_random = max(100, np.around(crossings[chan].size / 100)) if settings['use_rand_init'] else 0
        neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=settings['p_value_cut_thresh'])
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("Currently", curr_num_clusters.size, "different clusters")

        # Single channel branch
        if curr_num_clusters.size > 1 and settings['do_branch_PCA']:
            neuron_labels = branch_pca_2_0(neuron_labels, clips[:, curr_chan_inds],
                                np.arange(0, curr_chan_inds.size),
                                p_value_cut_thresh=settings['p_value_cut_thresh'],
                                add_peak_valley=settings['add_peak_valley'],
                                check_components=settings['check_components'],
                                max_components=settings['max_components'],
                                use_rand_init=settings['use_rand_init'],
                                method='pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if settings['verbose']: print("After SINGLE BRANCH", curr_num_clusters.size, "different clusters")

        # Multi channel branch
        if Probe.num_electrodes > 1 and settings['do_branch_PCA']:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=settings['p_value_cut_thresh'],
                                add_peak_valley=settings['add_peak_valley'],
                                check_components=settings['check_components'],
                                max_components=settings['max_components'],
                                use_rand_init=settings['use_rand_init'],
                                method='pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if settings['verbose']: print("After MULTI BRANCH", curr_num_clusters.size, "different clusters")

        # Multi channel branch by channel
        if Probe.num_electrodes > 1 and settings['do_branch_PCA']:
            neuron_labels = branch_pca_2_0(neuron_labels, clips, curr_chan_inds,
                                p_value_cut_thresh=settings['p_value_cut_thresh'],
                                add_peak_valley=settings['add_peak_valley'],
                                check_components=settings['check_components'],
                                max_components=settings['max_components'],
                                use_rand_init=settings['use_rand_init'],
                                method='chan_pca')
            curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
            if settings['verbose']: print("After MULTI BY CHAN BRANCH", curr_num_clusters.size, "different clusters")

        # Delete any clusters under min_cluster_size before binary pursuit
        if settings['verbose']: print("Current smallest cluster has", np.amin(n_per_cluster), "spikes")
        if np.any(n_per_cluster < min_cluster_size):
            for l_ind in range(0, curr_num_clusters.size):
                if n_per_cluster[l_ind] < min_cluster_size:
                    keep_inds = ~(neuron_labels == curr_num_clusters[l_ind])
                    crossings[chan] = crossings[chan][keep_inds]
                    neuron_labels = neuron_labels[keep_inds]
                    clips = clips[keep_inds, :]
                    print("Deleted cluster", curr_num_clusters[l_ind], "with", n_per_cluster[l_ind], "spikes")

        if neuron_labels.size == 0:
            if settings['verbose']: print("No clusters over min_firing_rate")
            labels[chan] = neuron_labels
            waveforms[chan] = clips
            new_waveforms[chan] = np.array([])
            if settings['verbose']: print("Done.")
            continue

        # Realign any units that have a template with peak > valley
        crossings[chan], units_shifted = check_upward_neurons(clips,
                                            crossings[chan], neuron_labels,
                                            curr_chan_inds, settings['clip_width'],
                                            Probe.sampling_rate)
        if settings['verbose']: print("Found", len(units_shifted), "upward neurons that were realigned", flush=True)
        if len(units_shifted) > 0:
            clips, valid_event_indices = segment.get_multichannel_clips(Probe,
                                            Probe.get_neighbors(chan),
                                            crossings[chan],
                                            clip_width=settings['clip_width'],
                                            thresholds=seg_dict['thresholds'])
            crossings[chan], neuron_labels = segment.keep_valid_inds(
                    [crossings[chan], neuron_labels], valid_event_indices)

        # Realign spikes based on correlation with current cluster templates before doing binary pursuit
        crossings[chan], neuron_labels, _ = segment.align_events_with_template(Probe, chan, neuron_labels, crossings[chan], clip_width=settings['clip_width'])
        if settings['do_binary_pursuit']:
            if settings['verbose']: print("currently", np.unique(neuron_labels).size, "different clusters")
            if settings['verbose']: print("Doing binary pursuit")
            if not settings['use_GPU']:
                crossings[chan], neuron_labels, new_inds = overlap.binary_pursuit_secret_spikes(
                                        Probe, chan, neuron_labels, crossings[chan],
                                        seg_dict['thresholds'][chan], settings['clip_width'])
                clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=settings['clip_width'], thresholds=seg_dict['thresholds'])
                crossings[chan], neuron_labels = segment.keep_valid_inds([crossings[chan], neuron_labels], valid_event_indices)
            else:
                crossings[chan], neuron_labels, new_inds, clips = binary_pursuit.binary_pursuit(
                    Probe, chan, crossings[chan], neuron_labels, settings['clip_width'],
                    thresholds=seg_dict['thresholds'], kernels_path=None,
                    max_gpu_memory=settings['max_gpu_memory'])
        else:
            # Need to get newly aligned clips and new_inds = False
            clips, valid_event_indices = segment.get_multichannel_clips(Probe, Probe.get_neighbors(chan), crossings[chan], clip_width=settings['clip_width'], thresholds=seg_dict['thresholds'])
            crossings[chan], neuron_labels = segment.keep_valid_inds([crossings[chan], neuron_labels], valid_event_indices)
            new_inds = np.zeros(crossings[chan].size, dtype=np.bool)

        if settings['verbose']: print("currently", np.unique(neuron_labels).size, "different clusters")
        # Adjust crossings for segment start time
        crossings[chan] += seg_dict['index_window'][0]
        # Map labels starting at zero
        sort.reorder_labels(neuron_labels)
        labels[chan] = neuron_labels
        waveforms[chan] = clips
        new_waveforms[chan] = new_inds
        if settings['verbose']: print("Done.")

    return crossings, labels, waveforms, new_waveforms


def spike_sort(Probe, **kwargs):
    """

    See 'spike_sorting_settings' above for a list of allowable kwargs.
    Example:
    import electrode
    import SpikeSorting
    class TestProbe(electrode.AbstractProbe):
        def __init__(self, sampling_rate, voltage_array, num_channels=None):
            if num_channels is None:
                num_channels = voltage_array.shape[1]
            electrode.AbstractProbe.__init__(self, sampling_rate, num_channels, voltage_array=voltage_array)

        def get_neighbors(self, channel):
            # This is a tetrode that uses all other contacts as its neighbors
            start = 0
            stop = 4
            return np.arange(start, stop)

    spike_sort_kwargs = {'sigma': 3., 'clip_width': [-6e-4, 8e-4],
                         'p_value_cut_thresh': 0.01, 'max_components': None,
                         'min_firing_rate': 1, do_binary_pursuit=True,
                         'cleanup_neurons': False, 'verbose': True}
    Probe = TestProbe(samples_per_second, voltage_array, num_channels=4)
    neurons = SpikeSorting.spike_sort(Probe, **spike_sort_kwargs)
    """
    # Get our settings
    settings = spike_sorting_settings(**kwargs)
    if not settings['use_GPU'] and settings['do_binary_pursuit']:
        use_GPU_message = ("Using CPU binary pursuit to find " \
                            "secret spikes. This can be MUCH MUCH " \
                            "slower and uses more " \
                            "memory than the GPU version. Returned \
                            clips will NOT be adjusted.")
        warnings.warn(use_GPU_message, RuntimeWarning, stacklevel=2)

    if settings['segment_duration'] is None or settings['segment_duration'] == np.inf:
        settings['segment_overlap'] = 0
        settings['segment_duration'] = Probe.n_samples
    else:
        if settings['segment_overlap'] is None:
            # If segment is specified with no overlap, use minimal overlap that
            # will not miss spikes on the edges
            clip_samples = segment.time_window_to_samples(settings['clip_width'], Probe.sampling_rate)[0]
            settings['segment_overlap'] = int(3 * (clip_samples[1] - clip_samples[0]))
        else:
            settings['segment_overlap'] = int(np.ceil(settings['segment_overlap'] * Probe.sampling_rate))
        settings['segment_duration'] = int(np.floor(settings['segment_duration'] * Probe.sampling_rate))
    segment_onsets = []
    segment_offsets = []
    curr_onset = 0
    while (curr_onset < Probe.n_samples):
        segment_onsets.append(curr_onset)
        segment_offsets.append(min(curr_onset + settings['segment_duration'], Probe.n_samples))
        curr_onset += settings['segment_duration'] - settings['segment_overlap']
    print("Using ", len(segment_onsets), "segments per channel for sorting.")

    segProbe = copy(Probe) # shallow copy
    if settings['do_ZCA_transform']: zca_cushion = \
                    (2 * np.ceil(np.amax(np.abs(settings['clip_width'])) \
                     * Probe.sampling_rate)).astype(np.int64)
    sort_data = []
    work_items = []
    w_ID = 0
    for x in range(0, len(segment_onsets)):
        if settings['verbose']: print("Starting segment number", x+1, "of", len(segment_onsets))
        # Need to copy or else ZCA transforms will duplicate in overlapping
        # time segments. Slice over num_electrodes should keep same shape
        segProbe.voltage = np.copy(Probe.voltage[0:Probe.num_electrodes,
                                   segment_onsets[x]:segment_offsets[x]])
        segProbe.n_samples = segment_offsets[x] - segment_onsets[x]
        seg_dict = {}
        seg_dict['n_samples'] = segment_offsets[x] - segment_onsets[x]
        seg_dict['seg_number'] = x
        seg_dict['index_window'] = [segment_onsets[x], segment_offsets[x]]
        seg_dict['overlap'] = settings['segment_overlap']
        if settings['do_ZCA_transform']:
            if settings['verbose']: print("Doing ZCA transform")
            seg_dict['thresholds'] = segment.median_threshold(segProbe.voltage, settings['sigma'])
            zca_matrix = preprocessing.get_noise_sampled_zca_matrix(
                            segProbe.voltage, seg_dict['thresholds'],
                            settings['sigma'], zca_cushion, n_samples=1e7)
            segProbe.voltage = zca_matrix @ segProbe.voltage
        seg_dict['thresholds'] = segment.median_threshold(segProbe.voltage, settings['sigma'])
        crossings, labels, waveforms, new_waveforms = spike_sort_segment(segProbe, seg_dict, settings)
        for chan in range(0, Probe.num_electrodes):
            sort_data.append([crossings[chan], labels[chan], waveforms[chan],
                              new_waveforms[chan]])
            work_items.append({'channel': chan,
                               'segment_dict': seg_dict, 'ID': w_ID})
            w_ID += 1


    sort_data, work_items = consolidate.organize_sort_data(sort_data, work_items, n_chans=Probe.num_electrodes)
    crossings, labels, waveforms, new_waveforms = consolidate.stitch_segments(sort_data, work_items)
    # Now that everything has been sorted, condense our representation of the
    # neurons by first combining data for each channel
    if settings['verbose']: print("Summarizing neurons")
    neurons = consolidate.summarize_neurons(Probe, crossings, labels,
                waveforms, new_waveforms=new_waveforms)

    # if settings['verbose']: print("Ordering neurons and finding peak valleys")
    # neurons = consolidate.recompute_template_wave_properties(neurons)
    # neurons = consolidate.reorder_neurons_by_raw_peak_valley(neurons)
    # # Consolidate neurons across channels
    # if settings['cleanup_neurons']:
    #     if settings['verbose']: print("Removing noise units")
    #     neurons = consolidate.remove_noise_neurons(neurons)
    #     if settings['verbose']: print("Combining same neurons in neighborhood")
    #     neurons = consolidate.combine_stolen_spikes(neurons, max_offset_samples=20, p_value_combine=.05, p_value_cut_thresh=p_value_cut_thresh, max_components=max_components)
    #     neurons = consolidate.combine_neighborhood_neurons(neurons, overlap_time=5e-4, overlap_threshold=5, max_offset_samples=10, p_value_cut_thresh=p_value_cut_thresh, max_components=max_components)
    #     neurons = consolidate.recompute_template_wave_properties(neurons)
    #     neurons = consolidate.reorder_neurons_by_raw_peak_valley(neurons)
    #     neurons = consolidate.remove_across_channel_duplicate_neurons(neurons, overlap_time=5e-4, overlap_threshold=5)

    if settings['verbose']: print("Done.")
    print("YOU WANT THIS TO SAVE THINGS LIKE THE SIGMA VALUE USED, AND THE MEDIAN DEVIATION (FOR COMPUTING SNR) AND REALLY ALL PARAMETERS")
    print("You want to use here, and in consolidate the SNR based on the threshold value found divided by SIGMA instead of Matt method.")
    print("David did this by just passing the input kwgs into a settings dictionary for the output")
    print("Also I need to make sure that all ints are np.int64 so this is compatible with windows compiled sort_cython, including in the segment files etc...")
    print("Also need to delete the load file functionality in electrode.py since it only assumes .npy file")
    return neurons
