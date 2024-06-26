import numpy as np
from scipy.stats import norm
from isosplit6 import isosplit6

import spikesorting_fullpursuit.dim_reduce.pca
from spikesorting_fullpursuit.clustering.isocut import merge_clusters
from spikesorting_fullpursuit.analyze_spike_timing import remove_spike_event_duplicates


def optimal_align_templates(
    temp_1,
    temp_2,
    n_chans,
    max_shift=None,
    align_abs=False,
    zero_pad=False,
):
    """ """
    n_samples_per_chan = int(temp_1.shape[0] / n_chans)
    if temp_1.shape[0] != temp_2.shape[0] or temp_1.ndim > 1 or n_samples_per_chan == 0:
        raise ValueError("Input templates must be 1D vectors of the same size")
    if temp_1.shape[0] % n_chans != 0:
        raise ValueError(
            "Template shape[0] must be evenly divisible by n_chans (i.e. there are the same number of samples per channel)."
        )

    if align_abs:
        # Use absolute value of cross correlation function to align
        cross_corr = np.abs(np.correlate(temp_1, temp_2, mode="full"))
    else:
        cross_corr = np.correlate(temp_1, temp_2, mode="full")
    xcorr_center = cross_corr.shape[0] // 2
    if max_shift is None:
        max_xcorr_ind = np.argmax(cross_corr)
        optimal_shift = max_xcorr_ind - xcorr_center
    else:
        max_xcorr_ind = np.argmax(
            cross_corr[xcorr_center - max_shift : xcorr_center + max_shift + 1]
        )
        max_xcorr_ind += xcorr_center - max_shift
        optimal_shift = max_xcorr_ind - xcorr_center

    if optimal_shift == 0:
        return temp_1, temp_2, optimal_shift, n_samples_per_chan

    # Align and truncate templates separately on each channel
    shift_temp1 = []
    shift_temp2 = []
    for chan in range(0, n_chans):
        chan_temp_1 = temp_1[
            chan * n_samples_per_chan : (chan + 1) * n_samples_per_chan
        ]
        chan_temp_2 = temp_2[
            chan * n_samples_per_chan : (chan + 1) * n_samples_per_chan
        ]
        if optimal_shift > 0:
            shift_temp1.append(chan_temp_1[optimal_shift:])
            shift_temp2.append(chan_temp_2[: -1 * optimal_shift])
        elif optimal_shift < 0:
            shift_temp1.append(chan_temp_1[:optimal_shift])
            shift_temp2.append(chan_temp_2[-1 * optimal_shift :])
        else:
            # Should be impossible since we return above if optimal_shift == 0
            raise RuntimeError("Optimal shift not found")

    shift_temp1 = np.hstack(shift_temp1)
    shift_temp2 = np.hstack(shift_temp2)
    shift_samples_per_chan = shift_temp1.shape[0] // n_chans

    if zero_pad:
        # Instead of truncating, zero pad the shift to output same size templates
        pad_shift_temp1 = np.zeros(temp_1.shape[0])
        pad_shift_temp2 = np.zeros(temp_2.shape[0])
        for chan in range(0, n_chans):
            t_win = [chan * n_samples_per_chan, (chan + 1) * n_samples_per_chan]
            s_win = [chan * shift_samples_per_chan, (chan + 1) * shift_samples_per_chan]
            if optimal_shift >= 0:
                pad_shift_temp1[t_win[0] + optimal_shift : t_win[1]] = shift_temp1[
                    s_win[0] : s_win[1]
                ]
                pad_shift_temp2[t_win[0] + optimal_shift : t_win[1]] = shift_temp2[
                    s_win[0] : s_win[1]
                ]
            else:
                pad_shift_temp1[t_win[0] : t_win[1] + optimal_shift] = shift_temp1[
                    s_win[0] : s_win[1]
                ]
                pad_shift_temp2[t_win[0] : t_win[1] + optimal_shift] = shift_temp2[
                    s_win[0] : s_win[1]
                ]
        return pad_shift_temp1, pad_shift_temp2, optimal_shift, n_samples_per_chan

    return shift_temp1, shift_temp2, optimal_shift, shift_samples_per_chan


def check_template_pair(
    template_1,
    template_2,
    chan_covariance_mats,
    sort_info,
):
    """
    Intended for testing whether a sum of templates is equal to a given
    template. Templates are assumed to be aligned with one another as no
    shifting is performed. Probability of confusiong the templates is
    returned. This confusion is symmetric, i.e. p_confusion template_1 assigned
    to template_2 equals p_confusion template_2 assigned to template_1."""
    if template_1.shape[0] != template_2.shape[0] or template_1.ndim > 1:
        raise ValueError("Input templates must be 1D vectors of the same size")
    n_chans = sort_info["n_channels"]
    if template_1.shape[0] % n_chans != 0:
        raise ValueError(
            "Template shape[0] must be evenly divisible by n_chans (i.e. there are the same number of samples per channel)."
        )
    template_samples_per_chan = template_1.shape[0] // n_chans
    for cov_mat in chan_covariance_mats:
        if (
            template_samples_per_chan != cov_mat.shape[0]
            and template_samples_per_chan != cov_mat.shape[1]
        ):
            print(
                "SHAPES LINE 87 CONSOLIDATE",
                template_1.shape[0],
                cov_mat.shape[0],
                template_1.shape[0],
                cov_mat.shape[1],
            )
            raise ValueError(
                "Each channel covariance matrix in chan_covariance_mats must be square matrix with dimensions equal to template length."
            )

    # Compute separability given V = template_1.
    E_L_t1 = 0.5 * np.dot(template_1, template_1)
    E_L_t2 = np.dot(template_1, template_2) - 0.5 * np.dot(template_2, template_2)
    var_diff = 0
    for chan in range(0, n_chans):
        t_win = [
            chan * template_samples_per_chan,
            (chan + 1) * template_samples_per_chan,
        ]
        diff_template = (
            template_1[t_win[0] : t_win[1]] - template_2[t_win[0] : t_win[1]]
        )
        var_diff += (
            diff_template[None, :] @ chan_covariance_mats[chan] @ diff_template[:, None]
        )

    # Expected difference between t1 and t2 likelihood functions
    E_diff_t1_t2 = E_L_t1 - E_L_t2
    if var_diff > 0:
        # Probability likelihood nt - t2 < 0
        p_confusion = norm.cdf(0, E_diff_t1_t2, np.sqrt(var_diff))
    else:
        p_confusion = 1.0

    return p_confusion


def get_snr(
    template,
    threshold,
    sigma,
):
    """
    Get SNR on the main channel relative to 3 STD of background noise.

    Parameters
    ----------
    template: np.ndarray
        single channel template
    threshold: float
        threshold for the channel
    sigma: float
        noise standard deviation

    Returns
    -------

    """
    background_noise_std = threshold / sigma
    temp_range = np.amax(template) - np.amin(template)
    return temp_range / (3 * background_noise_std)


def get_snr_across_all_channels(
    all_channel_template,
    thresholds,
    n_channels,
    n_samples_per_chan,
    sigma,
):
    """
    Get SNR for each channel for a given neuron

    Parameters
    ----------
    all_channel_template: np.ndarray
        template across all channels. Each channel is n_sample_per_chan
    thresholds: np.ndarray[float]
        thresholds for each channel
    n_channels: int
        total number of channels
    n_samples_per_chan: int
        Number of samples in each channels template
    sigma: float
        noise standard deviation

    Returns
    -------

    """
    snr_by_channel = np.zeros(n_channels)
    for chan in range(0, n_channels):
        this_template_start_ind = n_samples_per_chan * chan
        this_template_end_ind = n_samples_per_chan * (chan + 1)

        this_channel_threshold = thresholds[chan]

        snr_by_channel[chan] = get_snr(
            all_channel_template[this_template_start_ind:this_template_end_ind],
            this_channel_threshold,
            sigma,
        )
    return snr_by_channel


class SegSummary(object):
    """
    Main class that gathers all the sorted data and consolidates it within
    each segment. This is called after clustering, but before running binary
    pursuit to consolidate units across channels within each segment.

    Attributes
    ----------
    sort_data: List
        List containing 5 items
        crossings
        neuron_labels
        clips
        binary_pursuit_spikes (boolean array)
        work_item_id
    work_items
    sort_info
    v_dtype
    absolute_refractory_period: float

    verbose: bool

    half_clip_inds:
    full_clip_inds:
    n_items:
    summaries:

    """

    def __init__(
        self,
        sort_data,
        work_items,
        sort_info,
        v_dtype,
        absolute_refractory_period=20e-4,
        verbose=False,
    ):
        self.sort_data = sort_data
        self.work_items = work_items
        self.sort_info = sort_info
        self.v_dtype = v_dtype
        self.absolute_refractory_period = absolute_refractory_period
        self.verbose = verbose
        # NOTE: half_clip_inds is the larger piece of clip width so
        # 2*half_clip_inds is not necessarily the full clip inds
        self.half_clip_inds = int(
            round(
                np.amax(np.abs(self.sort_info["clip_width"]))
                * self.sort_info["sampling_rate"]
            )
        )
        self.full_clip_inds = int(
            round(
                (self.sort_info["clip_width"][1] - self.sort_info["clip_width"][0])
                * self.sort_info["sampling_rate"]
            )
        )
        self.n_items = len(work_items)
        self.make_summaries()

    def zero_low_snr_chans(self):
        """Sets bp_template values for channels under bp_chan_snr threshold
        to values of zero."""
        if (self.sort_info["bp_chan_snr"] is None) or (
            self.sort_info["bp_chan_snr"] <= 0
        ):
            return None
        for neuron_summary in self.summaries:
            for chan in range(0, self.sort_info["n_channels"]):
                if neuron_summary["snr_by_chan"][chan] < self.sort_info["bp_chan_snr"]:
                    chan_win = [
                        self.sort_info["n_samples_per_chan"] * chan,
                        self.sort_info["n_samples_per_chan"] * (chan + 1),
                    ]
                    neuron_summary["bp_template"][chan_win[0] : chan_win[1]] = 0
        return None

    def set_bp_templates(self, bp_templates):
        """Assign the templates in the input bp_templates to the
        corresponding seg summary units.
        bp_templates must be a n_units x n_samples numpy array or list of
        numpy array templates. These will be added to the summaries in the order
        in which they appear!"""
        if not isinstance(bp_templates, list):
            if bp_templates.ndim == 1:
                bp_templates = [bp_templates]
            else:
                templates_list = []
                for t in range(0, bp_templates.shape[0]):
                    templates_list.append(bp_templates[t, :])
                bp_templates = templates_list
        for ind, neuron_summary in enumerate(self.summaries):
            neuron_summary["bp_template"] = bp_templates[ind]
        self.zero_low_snr_chans()

    def make_summaries(self):
        """Make a neuron summary for each unit in each segment and add them to
        a new class attribute 'summaries'.
        """
        self.summaries = []
        for work_item_index in range(0, self.n_items):
            if len(self.sort_data[work_item_index][0]) == 0:
                # No data for this item
                continue
            if (
                self.sort_data[work_item_index][2].shape[1]
                != self.sort_info["n_samples_per_chan"] * self.sort_info["n_channels"]
            ):
                raise ValueError("Clips must include data for all channels")
            cluster_labels = np.unique(self.sort_data[work_item_index][1])
            for neuron_label in cluster_labels:
                neuron = self.create_neuron_summary(work_item_index, neuron_label)

                if len(neuron["high_snr_neighbors"]) == 0:
                    # Neuron is total trash so don't even append to summaries
                    continue
                elif neuron["snr"] < 1.5:
                    # SNR this low indicates true garbage that will only slow
                    # binary pursuit so skip it outright
                    # Remember that SNR can only go down from here as binary
                    # pursuit can add spikes that didn't cross threshold
                    continue

                self.summaries.append(neuron)

    def create_neuron_summary(self, work_item_index, neuron_label):
        neuron = {}
        neuron["summary_type"] = "single_segment"
        neuron["channel"] = self.work_items[work_item_index]["channel"]
        neuron["neighbors"] = self.work_items[work_item_index]["neighbors"]
        # This assumes that input has all channels in order!
        neuron["main_win"] = [
            self.sort_info["n_samples_per_chan"] * neuron["channel"],
            self.sort_info["n_samples_per_chan"] * (neuron["channel"] + 1),
        ]
        neuron["threshold"] = self.work_items[work_item_index]["thresholds"][
            neuron["channel"]
        ]
        select_label = self.sort_data[work_item_index][1] == neuron_label
        neuron["spike_indices"] = self.sort_data[work_item_index][0][select_label]
        neuron["clips"] = self.sort_data[work_item_index][2][select_label, :]
        # NOTE: This still needs to be done even though segments
        # were ordered because of overlap!
        # Ensure spike times are ordered. Must use 'stable' sort for
        # output to be repeatable because overlapping segments and
        # binary pursuit can return slightly different dupliate spikes
        spike_order = np.argsort(neuron["spike_indices"], kind="stable")
        neuron["spike_indices"] = neuron["spike_indices"][spike_order]
        neuron["clips"] = neuron["clips"][spike_order, :]

        # Set duplicate tolerance as full clip width since we are only
        # looking to get a good template here
        neuron["duplicate_tol_inds"] = self.full_clip_inds

        # Remove any identical index duplicates (either from error or
        # from combining overlapping segments), preferentially keeping
        # the waveform best aligned to the template
        neuron["template"] = np.median(neuron["clips"], axis=0).astype(
            neuron["clips"].dtype
        )
        keep_bool = remove_spike_event_duplicates(
            neuron["spike_indices"],
            neuron["clips"],
            neuron["template"],
            tol_inds=neuron["duplicate_tol_inds"],
        )
        neuron["spike_indices"] = neuron["spike_indices"][keep_bool]
        neuron["clips"] = neuron["clips"][keep_bool, :]

        # Recompute template and store output
        neuron["template"] = np.median(neuron["clips"], axis=0).astype(
            neuron["clips"].dtype
        )
        # Get SNR for each channel separately
        snr_by_channel = get_snr_across_all_channels(
            neuron["template"],
            self.work_items[work_item_index]["thresholds"],
            self.sort_info["n_channels"],
            self.sort_info["n_samples_per_chan"],
            self.sort_info["sigma"],
        )

        neuron["snr_by_chan"] = snr_by_channel
        neuron["snr"] = neuron["snr_by_chan"][neuron["channel"]]

        # Preserve template over full neighborhood for certain comparisons
        neuron["full_template"] = np.copy(neuron["template"])
        # Set new neighborhood of all channels with SNR over SNR threshold.
        # This will be used for align shifting and merge testing
        # NOTE: This new neighborhood only applies for use internally
        high_snr_neighbors = []
        for chan in range(0, self.sort_info["n_channels"]):
            if neuron["snr_by_chan"][chan] > 0.5:
                high_snr_neighbors.append(chan)
        neuron["high_snr_neighbors"] = np.array(high_snr_neighbors, dtype=np.int64)
        neuron["deleted_as_redundant"] = False
        return neuron

    def find_nearest_shifted_pair(self, remaining_inds, previously_compared_pairs):
        """Alternative to sort_cython.identify_clusters_to_compare that simply
        chooses the nearest template after shifting to optimal alignment.
        Intended as helper function so that neurons do not fail to stitch in the
        event their alignment changes between segments."""
        best_distance = np.inf
        for n1_ind in remaining_inds:
            n1 = self.summaries[n1_ind]
            for n2_ind in remaining_inds:
                if (n1_ind <= n2_ind) or (
                    [n1_ind, n2_ind] in previously_compared_pairs
                ):
                    # Do not perform repeat or identical comparisons
                    continue
                n2 = self.summaries[n2_ind]
                if n2["channel"] not in n1["neighbors"]:
                    # Must be within each other's neighborhoods
                    previously_compared_pairs.append([n1_ind, n2_ind])
                    continue
                # NOTE: This is a 'lazy' shift because it does not shift within
                # each channel window separately
                cross_corr = np.correlate(
                    n1["full_template"], n2["full_template"], mode="full"
                )
                max_corr_ind = np.argmax(cross_corr)
                curr_shift = max_corr_ind - cross_corr.shape[0] // 2
                if np.abs(curr_shift) > self.half_clip_inds:
                    # Do not allow shifts to extend unreasonably
                    continue
                # Align and truncate template and compute distance
                if curr_shift > 0:
                    shiftn1 = n1["full_template"][curr_shift:]
                    shiftn2 = n2["full_template"][: -1 * curr_shift]
                elif curr_shift < 0:
                    shiftn1 = n1["full_template"][:curr_shift]
                    shiftn2 = n2["full_template"][-1 * curr_shift :]
                else:
                    shiftn1 = n1["full_template"]
                    shiftn2 = n2["full_template"]
                # Must normalize distance per data point else reward big shifts
                curr_distance = np.sum((shiftn1 - shiftn2) ** 2) / shiftn1.shape[0]
                if curr_distance < best_distance:
                    best_distance = curr_distance
                    best_shift = curr_shift
                    best_pair = [n1_ind, n2_ind]
        if np.isinf(best_distance):
            # Never found a match
            best_pair = []
            best_shift = 0
            clips_1 = None
            clips_2 = None
            return best_pair, best_shift, clips_1, clips_2, None, None

        # Reset n1 and n2 to match the best then calculate clips
        n1 = self.summaries[best_pair[0]]
        n2 = self.summaries[best_pair[1]]

        # Create extended clips and align and truncate them for best match pair
        shift_samples_per_chan = self.sort_info["n_samples_per_chan"] - np.abs(
            best_shift
        )
        clips_1 = np.zeros(
            (
                n1["clips"].shape[0],
                shift_samples_per_chan * self.sort_info["n_channels"],
            ),
            dtype=self.v_dtype,
        )
        clips_2 = np.zeros(
            (
                n2["clips"].shape[0],
                shift_samples_per_chan * self.sort_info["n_channels"],
            ),
            dtype=self.v_dtype,
        )
        sample_select = np.zeros(
            shift_samples_per_chan * self.sort_info["n_channels"], dtype="bool"
        )

        # Get clips for each channel, shift them, and assign for output, which
        # will be clips that have each channel individually aligned and
        # truncated
        chans_used_for_clips = []
        for chan in range(0, self.sort_info["n_channels"]):
            # Only keep channels with high SNR data from at least one unit
            if chan in n1["high_snr_neighbors"] or chan in n2["high_snr_neighbors"]:
                chan_clips_1 = n1["clips"][
                    :,
                    chan
                    * self.sort_info["n_samples_per_chan"] : (chan + 1)
                    * self.sort_info["n_samples_per_chan"],
                ]
                if best_shift >= 0:
                    clips_1[
                        :,
                        chan
                        * shift_samples_per_chan : (chan + 1)
                        * shift_samples_per_chan,
                    ] = chan_clips_1[:, best_shift:]
                elif best_shift < 0:
                    clips_1[
                        :,
                        chan
                        * shift_samples_per_chan : (chan + 1)
                        * shift_samples_per_chan,
                    ] = chan_clips_1[:, :best_shift]

                chan_clips_2 = n2["clips"][
                    :,
                    chan
                    * self.sort_info["n_samples_per_chan"] : (chan + 1)
                    * self.sort_info["n_samples_per_chan"],
                ]
                if best_shift > 0:
                    clips_2[
                        :,
                        chan
                        * shift_samples_per_chan : (chan + 1)
                        * shift_samples_per_chan,
                    ] = chan_clips_2[:, : -1 * best_shift]
                elif best_shift <= 0:
                    clips_2[
                        :,
                        chan
                        * shift_samples_per_chan : (chan + 1)
                        * shift_samples_per_chan,
                    ] = chan_clips_2[:, -1 * best_shift :]
                sample_select[
                    chan * shift_samples_per_chan : (chan + 1) * shift_samples_per_chan
                ] = True
                chans_used_for_clips.append(chan)
        chans_used_for_clips = np.int64(np.hstack(chans_used_for_clips))

        # Compare best distance to size of the template SSE to see if its reasonable
        min_template_SSE = min(
            np.sum(self.summaries[best_pair[0]]["full_template"] ** 2),
            np.sum(self.summaries[best_pair[1]]["full_template"] ** 2),
        )
        min_template_SSE /= self.summaries[best_pair[0]]["full_template"].shape[
            0
        ] - np.abs(best_shift)
        if np.any(sample_select) and (best_distance < 0.5 * min_template_SSE):
            clips_1 = clips_1[:, sample_select]
            clips_2 = clips_2[:, sample_select]
            return (
                best_pair,
                best_shift,
                clips_1,
                clips_2,
                chans_used_for_clips,
                shift_samples_per_chan,
            )
        else:
            # This is probably not a good match afterall, so try again
            previously_compared_pairs.append(best_pair)
            (
                best_pair,
                best_shift,
                clips_1,
                clips_2,
                chans_used_for_clips,
                shift_samples_per_chan,
            ) = self.find_nearest_shifted_pair(
                remaining_inds, previously_compared_pairs
            )
            return (
                best_pair,
                best_shift,
                clips_1,
                clips_2,
                chans_used_for_clips,
                shift_samples_per_chan,
            )

    def confusion_test_two_units(
        self,
        n1_ind,
        n2_ind,
        chan_covariance_mats,
        max_shift=None,
    ):
        shift_temp1, shift_temp2, _, _ = optimal_align_templates(
            self.summaries[n1_ind]["bp_template"],
            self.summaries[n2_ind]["bp_template"],
            self.sort_info["n_channels"],
            max_shift=max_shift,
            align_abs=False,
            zero_pad=True,
        )

        # Hard coded at 10% errors which is roughly useful without being too strict
        confusion_threshold = 0.1
        p_confusion = check_template_pair(
            shift_temp1,
            shift_temp2,
            chan_covariance_mats,
            self.sort_info,
        )
        if p_confusion > confusion_threshold:
            confused = True
        else:
            confused = False

        return confused

    def re_sort_two_units(
        self,
        clips_1,
        clips_2,
        use_weights=True,
        curr_chan_inds=None,
    ):
        if self.sort_info["add_peak_valley"] and curr_chan_inds is None:
            raise ValueError("Must give curr_chan_inds if using peak valley.")

        # Get each clip score from template based PCA space
        clips = np.vstack((clips_1, clips_2))
        orig_neuron_labels = np.ones(clips.shape[0], dtype=np.int64)
        orig_neuron_labels[clips_1.shape[0] :] = 2

        # Projection onto templates, weighted by number of spikes
        t1 = np.median(clips_1, axis=0)
        t2 = np.median(clips_2, axis=0)
        if use_weights:
            t1 *= clips_1.shape[0] / clips.shape[0]
            t2 *= clips_2.shape[0] / clips.shape[0]
        scores = clips @ np.vstack((t1, t2)).T

        scores = np.float64(scores)
        # neuron_labels = isosplit6(scores)
        # neuron_labels = isosplit6(scores, initial_labels=orig_neuron_labels)

        neuron_labels = merge_clusters(
            scores,
            orig_neuron_labels,
            split_only=False,
            merge_only=False,
            p_value_cut_thresh=self.sort_info["p_value_cut_thresh"],
            match_cluster_size=self.sort_info["match_cluster_size"],
            check_splits=self.sort_info["check_splits"],
        )

        curr_labels, n_per_label = np.unique(neuron_labels, return_counts=True)
        if curr_labels.size == 1:
            clips_merged = True
        else:
            clips_merged = False
        return clips_merged

    def merge_test_two_units(
        self,
        clips_1,
        clips_2,
        p_cut,
        method="template_pca",
        split_only=False,
        merge_only=False,
        use_weights=True,
        curr_chan_inds=None,
    ):
        if self.sort_info["add_peak_valley"] and curr_chan_inds is None:
            raise ValueError("Must give curr_chan_inds if using peak valley.")
        clips = np.vstack((clips_1, clips_2))
        neuron_labels = np.ones(clips.shape[0], dtype=np.int64)
        neuron_labels[clips_1.shape[0] :] = 2
        if method.lower() == "pca":
            scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca(
                clips,
                self.sort_info["check_components"],
                self.sort_info["max_components"],
                add_peak_valley=self.sort_info["add_peak_valley"],
                curr_chan_inds=curr_chan_inds,
                n_samples=1e6,
            )
        elif method.lower() == "pca_by_channel":
            scores = spikesorting_fullpursuit.dim_reduce.pca.compute_pca_by_channel(
                clips,
                curr_chan_inds,
                self.sort_info["check_components"],
                self.sort_info["max_components"],
                add_peak_valley=self.sort_info["add_peak_valley"],
                n_samples=1e6,
            )
        elif method.lower() == "template_pca":
            scores = spikesorting_fullpursuit.dim_reduce.pca.compute_template_pca(
                clips,
                neuron_labels,
                curr_chan_inds,
                self.sort_info["check_components"],
                self.sort_info["max_components"],
                add_peak_valley=self.sort_info["add_peak_valley"],
                use_weights=use_weights,
            )
        elif method.lower() == "channel_template_pca":
            scores = (
                spikesorting_fullpursuit.dim_reduce.pca.compute_template_pca_by_channel(
                    clips,
                    neuron_labels,
                    curr_chan_inds,
                    self.sort_info["check_components"],
                    self.sort_info["max_components"],
                    add_peak_valley=self.sort_info["add_peak_valley"],
                    use_weights=use_weights,
                )
            )
        elif method.lower() == "projection":
            # Projection onto templates, weighted by number of spikes
            t1 = np.median(clips_1, axis=0)
            t2 = np.median(clips_2, axis=0)
            if use_weights:
                t1 *= clips_1.shape[0] / clips.shape[0]
                t2 *= clips_2.shape[0] / clips.shape[0]
            scores = clips @ np.vstack((t1, t2)).T
        else:
            raise ValueError(
                "Unknown method", method, "for scores. Must use 'pca' or 'projection'."
            )
        scores = np.float64(scores)
        # neuron_labels = isosplit6(scores)
        # neuron_labels = isosplit6(scores, initial_labels=neuron_labels)

        neuron_labels = merge_clusters(
            scores,
            neuron_labels,
            split_only=split_only,
            merge_only=merge_only,
            p_value_cut_thresh=p_cut,
            match_cluster_size=self.sort_info["match_cluster_size"],
            check_splits=self.sort_info["check_splits"],
        )

        label_is_1 = neuron_labels == 1
        label_is_2 = neuron_labels == 2
        if np.all(label_is_1) or np.all(label_is_2):
            clips_merged = True
        else:
            clips_merged = False
        neuron_labels_1 = neuron_labels[0 : clips_1.shape[0]]
        neuron_labels_2 = neuron_labels[clips_1.shape[0] :]
        return clips_merged, neuron_labels_1, neuron_labels_2

    def merge_templates(self, neuron1_ind, neuron2_ind, shift):
        """Returns weighted average of neuron1 and 2 templates, accounting for
        the shift alignment offset 'shift'. Done in the same way as
        find_nearest_shifted_pair such that neuron2 is shifted in the same way."""
        n1 = self.summaries[neuron1_ind]
        n2 = self.summaries[neuron2_ind]
        merged_template = np.zeros(n1["full_template"].shape[0])
        if shift != 0:
            shift_template_2 = np.zeros(n2["full_template"].shape[0])
            for chan in range(0, self.sort_info["n_channels"]):
                chan_temp_2 = n2["full_template"][
                    chan
                    * self.sort_info["n_samples_per_chan"] : (chan + 1)
                    * self.sort_info["n_samples_per_chan"]
                ]
                if shift > 0:
                    shift_template_2[
                        chan * self.sort_info["n_samples_per_chan"]
                        + shift : (chan + 1) * self.sort_info["n_samples_per_chan"]
                    ] = chan_temp_2[: -1 * shift]
                else:
                    shift_template_2[
                        chan
                        * self.sort_info["n_samples_per_chan"] : (chan + 1)
                        * self.sort_info["n_samples_per_chan"]
                        + shift
                    ] = chan_temp_2[-1 * shift :]
        else:
            shift_template_2 = n2["full_template"]

        n1_weight = n1["spike_indices"].shape[0] / (
            n1["spike_indices"].shape[0] + n2["spike_indices"].shape[0]
        )
        merged_template = (
            n1_weight * n1["full_template"] + (1 - n1_weight) * shift_template_2
        )

        return merged_template

    def sharpen_across_chans(self, chan_covariance_mats=None):
        """
        Decides pairwise whether templates should be combined and treated as the
        same unit. If chan_covariance_mats is given, the test is done using
        the binary pursuit statistics, otherwise a cluster merge test using
        pca is performed.
        """
        inds_to_delete = []
        remaining_inds = [x for x in range(0, len(self.summaries))]
        previously_compared_pairs = []
        templates_to_merge = []
        while len(remaining_inds) > 1:
            (
                best_pair,
                best_shift,
                clips_1,
                clips_2,
                chans_used_for_clips,
                shift_samples_per_chan,
            ) = self.find_nearest_shifted_pair(
                remaining_inds, previously_compared_pairs
            )
            if len(best_pair) == 0:
                break
            if clips_1.shape[0] == 1 or clips_2.shape[0] == 1:
                # Don't mess around with only 1 spike, if they are
                # nearest each other they can merge
                is_merged = True
            elif chan_covariance_mats is not None:
                is_merged = self.confusion_test_two_units(
                    best_pair[0],
                    best_pair[1],
                    chan_covariance_mats,
                    max_shift=None,
                )
            else:
                curr_chan_inds = np.arange(0, shift_samples_per_chan, dtype=np.int64)
                is_merged_chan = self.merge_test_two_units(
                    clips_1,
                    clips_2,
                    self.sort_info["p_value_cut_thresh"],
                    method="pca_by_channel",
                    split_only=False,
                    merge_only=False,
                    use_weights=False,
                    curr_chan_inds=curr_chan_inds,
                )
                is_merged_all = self.merge_test_two_units(
                    clips_1,
                    clips_2,
                    self.sort_info["p_value_cut_thresh"],
                    method="template_pca",
                    split_only=False,
                    merge_only=False,
                    use_weights=False,
                    curr_chan_inds=None,
                )
                if is_merged_chan and is_merged_all:
                    is_merged = True
                else:
                    is_merged = False

            if is_merged:
                # Delete the unit with the fewest spikes
                if (
                    self.summaries[best_pair[0]]["spike_indices"].shape[0]
                    > self.summaries[best_pair[1]]["spike_indices"].shape[0]
                ):
                    inds_to_delete.append(best_pair[1])
                    remaining_inds.remove(best_pair[1])
                    templates_to_merge.append([best_pair[0], best_pair[1], best_shift])
                else:
                    inds_to_delete.append(best_pair[0])
                    remaining_inds.remove(best_pair[0])
                    # Invert shift because merge_templates assumes shift relative
                    # to first unit (best_pair[0])
                    templates_to_merge.append(
                        [best_pair[1], best_pair[0], -1 * best_shift]
                    )
            else:
                # These mutually closest failed so do not repeat either
                remaining_inds.remove(best_pair[0])
                remaining_inds.remove(best_pair[1])
            previously_compared_pairs.append(best_pair)

        for merge_item in templates_to_merge:
            merged_template = self.merge_templates(
                merge_item[0], merge_item[1], merge_item[2]
            )
            # Update new weighted merged template
            self.summaries[merge_item[0]]["full_template"] = merged_template
            # Also update number of spikes so that future merges can have the
            # correct weighting. These are no long in correspondence with the clips
            # and not sorted so it is assumed they will not be used again
            self.summaries[merge_item[0]]["spike_indices"] = np.hstack(
                (
                    self.summaries[merge_item[0]]["spike_indices"],
                    self.summaries[merge_item[1]]["spike_indices"] + merge_item[2],
                )
            )
            # Need these to stay sorted for future clip finding
            self.summaries[merge_item[0]]["spike_indices"].sort()
            # Update to unioned neighborhood
            self.summaries[merge_item[0]]["neighbors"] = np.union1d(
                self.summaries[merge_item[0]]["neighbors"],
                self.summaries[merge_item[1]]["neighbors"],
            )
            self.summaries[merge_item[0]]["high_snr_neighbors"] = np.union1d(
                self.summaries[merge_item[0]]["high_snr_neighbors"],
                self.summaries[merge_item[1]]["high_snr_neighbors"],
            )
        # Delete merged units
        inds_to_delete.sort()
        for d_ind in reversed(inds_to_delete):
            del self.summaries[d_ind]

    def remove_redundant_neurons(self, overlap_ratio_threshold=1):
        """
        Note that this function does not actually delete anything. It removes
        links between segments for redundant units and it adds a flag under
        the key 'deleted_as_redundant' to indicate that a segment unit should
        be deleted. Deleting units in this function would ruin the indices used
        to link neurons together later and is not worth the book keeping trouble.
        Note: overlap_ratio_threshold == np.inf will not delete anything while
        overlap_ratio_threshold == -np.inf will delete everything except 1 unit.
        """
        rn_verbose = False
        # Since we are comparing across channels, we need to consider potentially
        # large alignment differences in the overlap_time
        overlap_time = self.half_clip_inds / self.sort_info["sampling_rate"]
        neurons = self.summaries
        overlap_ratio = np.zeros((len(neurons), len(neurons)))
        expected_ratio = np.zeros((len(neurons), len(neurons)))
        delta_ratio = np.zeros((len(neurons), len(neurons)))
        quality_scores = [n["quality_score"] for n in neurons]
        violation_partners = [set() for x in range(0, len(neurons))]
        for neuron1_ind, neuron1 in enumerate(neurons):
            violation_partners[neuron1_ind].add(neuron1_ind)
            # Loop through all pairs of units and compute overlap and expected
            for neuron2_ind in range(neuron1_ind + 1, len(neurons)):
                neuron2 = neurons[neuron2_ind]
                if neuron1["channel"] == neuron2["channel"]:
                    continue  # If they are on the same channel, do nothing
                exp, act, delta = calc_ccg_overlap_ratios(
                    neuron1["spike_indices"],
                    neuron2["spike_indices"],
                    overlap_time,
                    self.sort_info["sampling_rate"],
                )
                expected_ratio[neuron1_ind, neuron2_ind] = exp
                expected_ratio[neuron2_ind, neuron1_ind] = expected_ratio[
                    neuron1_ind, neuron2_ind
                ]
                overlap_ratio[neuron1_ind, neuron2_ind] = act
                overlap_ratio[neuron2_ind, neuron1_ind] = overlap_ratio[
                    neuron1_ind, neuron2_ind
                ]
                delta_ratio[neuron1_ind, neuron2_ind] = delta
                delta_ratio[neuron2_ind, neuron1_ind] = delta_ratio[
                    neuron1_ind, neuron2_ind
                ]
                if (
                    overlap_ratio[neuron1_ind, neuron2_ind]
                    - expected_ratio[neuron1_ind, neuron2_ind]
                    > overlap_ratio_threshold * delta_ratio[neuron1_ind, neuron2_ind]
                ):
                    # Overlap is higher than chance and at least one of these will be removed
                    violation_partners[neuron1_ind].add(neuron2_ind)
                    violation_partners[neuron2_ind].add(neuron1_ind)

        neurons_remaining_indices = [x for x in range(0, len(neurons))]
        max_accepted = 0.0
        max_expected = 0.0
        while True:
            # Look for our next best pair
            best_ratio = -np.inf
            best_pair = []
            for i in range(0, len(neurons_remaining_indices)):
                for j in range(i + 1, len(neurons_remaining_indices)):
                    neuron_1_index = neurons_remaining_indices[i]
                    neuron_2_index = neurons_remaining_indices[j]
                    if (
                        overlap_ratio[neuron_1_index, neuron_2_index]
                        < overlap_ratio_threshold
                        * delta_ratio[neuron_1_index, neuron_2_index]
                        + expected_ratio[neuron_1_index, neuron_2_index]
                    ):
                        # Overlap not high enough to merit deletion of one
                        # But track our proximity to input threshold
                        if overlap_ratio[neuron_1_index, neuron_2_index] > max_accepted:
                            max_accepted = overlap_ratio[neuron_1_index, neuron_2_index]
                            max_expected = (
                                overlap_ratio_threshold
                                * delta_ratio[neuron_1_index, neuron_2_index]
                                + expected_ratio[neuron_1_index, neuron_2_index]
                            )
                        continue
                    if overlap_ratio[neuron_1_index, neuron_2_index] > best_ratio:
                        best_ratio = overlap_ratio[neuron_1_index, neuron_2_index]
                        best_pair = [neuron_1_index, neuron_2_index]
            if len(best_pair) == 0 or best_ratio == 0:
                # No more pairs exceed ratio threshold
                print(
                    "Maximum accepted ratio was",
                    max_accepted,
                    "at expected threshold",
                    max_expected,
                )
                break

            # We now need to choose one of the pair to delete.
            neuron_1 = neurons[best_pair[0]]
            neuron_2 = neurons[best_pair[1]]
            delete_1 = False
            delete_2 = False

            # We will also consider how good each neuron is relative to the
            # other neurons that it overlaps with. Basically, if one unit is
            # a best remaining copy while the other has better units it overlaps
            # with, we want to preferentially keep the best remaining copy
            # This trimming should work because we choose the most overlapping
            # pairs for each iteration
            combined_violations = violation_partners[best_pair[0]].union(
                violation_partners[best_pair[1]]
            )
            max_other_n1 = neuron_1["quality_score"]
            other_n1 = combined_violations - violation_partners[best_pair[1]]
            for v_ind in other_n1:
                if quality_scores[v_ind] > max_other_n1:
                    max_other_n1 = quality_scores[v_ind]
            max_other_n2 = neuron_2["quality_score"]
            other_n2 = combined_violations - violation_partners[best_pair[0]]
            for v_ind in other_n2:
                if quality_scores[v_ind] > max_other_n2:
                    max_other_n2 = quality_scores[v_ind]
            # Rate each unit on the difference between its quality and the
            # quality of its best remaining violation partner
            # NOTE: diff_score = 0 means this unit is the best remaining
            diff_score_1 = max_other_n1 - neuron_1["quality_score"]
            diff_score_2 = max_other_n2 - neuron_2["quality_score"]

            # Check if both or either had a failed MUA calculation
            if np.isnan(neuron_1["fraction_mua"]) and np.isnan(
                neuron_2["fraction_mua"]
            ):
                # MUA calculation was invalid so just use SNR
                if (
                    neuron_1["snr"] * neuron_1["spike_indices"].shape[0]
                    > neuron_2["snr"] * neuron_2["spike_indices"].shape[0]
                ):
                    delete_2 = True
                else:
                    delete_1 = True
            elif np.isnan(neuron_1["fraction_mua"]) or np.isnan(
                neuron_2["fraction_mua"]
            ):
                # MUA calculation was invalid for one unit so pick the other
                if np.isnan(neuron_1["fraction_mua"]):
                    delete_1 = True
                else:
                    delete_2 = True
            elif diff_score_1 > diff_score_2:
                # Neuron 1 has a better copy somewhere so delete it
                delete_1 = True
                delete_2 = False
            elif diff_score_2 > diff_score_1:
                # Neuron 2 has a better copy somewhere so delete it
                delete_1 = False
                delete_2 = True
            else:
                # Both diff scores == 0 so we have to pick one
                if diff_score_1 != 0 and diff_score_2 != 0:
                    raise RuntimeError(
                        "DIFF SCORES IN REDUNDANT ARE NOT BOTH EQUAL TO ZERO BUT I THOUGHT THEY SHOULD BE!"
                    )
                # First defer to choosing highest quality score
                if neuron_1["quality_score"] > neuron_2["quality_score"]:
                    delete_1 = False
                    delete_2 = True
                else:
                    delete_1 = True
                    delete_2 = False

                # Check if quality score is primarily driven by number of spikes rather than SNR and MUA
                # Spike number is primarily valuable in the case that one unit
                # is truly a subset of another. If one unit is a mixture, we
                # need to avoid relying on spike count
                if (
                    (
                        delete_2
                        and (1 - neuron_2["fraction_mua"]) * neuron_2["snr"]
                        > (1 - neuron_1["fraction_mua"]) * neuron_1["snr"]
                    )
                    or (
                        delete_1
                        and (1 - neuron_1["fraction_mua"]) * neuron_1["snr"]
                        > (1 - neuron_2["fraction_mua"]) * neuron_2["snr"]
                    )
                    or len(violation_partners[best_pair[0]])
                    != len(violation_partners[best_pair[1]])
                ):
                    if rn_verbose:
                        print("Checking for mixture due to lopsided spike counts")
                    if len(violation_partners[best_pair[0]]) < len(
                        violation_partners[best_pair[1]]
                    ):
                        if rn_verbose:
                            print(
                                "Neuron 1 has fewer violation partners. Set default delete neuron 2."
                            )
                        delete_1 = False
                        delete_2 = True
                    elif len(violation_partners[best_pair[1]]) < len(
                        violation_partners[best_pair[0]]
                    ):
                        if rn_verbose:
                            print(
                                "Neuron 2 has fewer violation partners. Set default delete neuron 1."
                            )
                        delete_1 = True
                        delete_2 = False
                    else:
                        if rn_verbose:
                            print("Both have equal violation partners")
                    # We will now check if one unit appears to be a subset of the other
                    # If these units are truly redundant subsets, then the MUA of
                    # their union will be <= max(mua1, mua2)
                    # If instead one unit is largely a mixture containing the
                    # other, then the MUA of their union should greatly increase
                    # Note that the if statement above typically only passes
                    # in the event that one unit has considerably more spikes or
                    # both units are extremely similar. Because rates can vary,
                    # we do not use peak MUA here but rather the rate based MUA
                    # Need to union with compliment so spikes are not double
                    # counted, which will reduce the rate based MUA
                    neuron_1_compliment = ~find_overlapping_spike_bool(
                        neuron_1["spike_indices"],
                        neuron_2["spike_indices"],
                        self.half_clip_inds,
                    )
                    union_spikes = np.hstack(
                        (
                            neuron_1["spike_indices"][neuron_1_compliment],
                            neuron_2["spike_indices"],
                        )
                    )
                    union_spikes.sort()
                    # union_duplicate_tol = self.half_clip_inds
                    union_duplicate_tol = max(
                        neuron_1["duplicate_tol_inds"], neuron_2["duplicate_tol_inds"]
                    )
                    union_fraction_mua_rate = calc_fraction_mua(
                        union_spikes,
                        self.sort_info["sampling_rate"],
                        union_duplicate_tol,
                        self.absolute_refractory_period,
                    )
                    # Need to get fraction MUA by rate, rather than peak,
                    # for comparison here
                    fraction_mua_rate_1 = calc_fraction_mua(
                        neuron_1["spike_indices"],
                        self.sort_info["sampling_rate"],
                        union_duplicate_tol,
                        self.absolute_refractory_period,
                    )
                    fraction_mua_rate_2 = calc_fraction_mua(
                        neuron_2["spike_indices"],
                        self.sort_info["sampling_rate"],
                        union_duplicate_tol,
                        self.absolute_refractory_period,
                    )
                    # We will decide not to consider spike count if this looks like
                    # one unit could be a large mixture. This usually means that
                    # the union MUA goes up substantially. To accomodate noise,
                    # require that it exceeds both the minimum MUA plus the MUA
                    # expected if the units were totally independent, and the
                    # MUA of either unit alone.
                    if union_fraction_mua_rate > min(
                        fraction_mua_rate_1, fraction_mua_rate_2
                    ) + delta_ratio[
                        best_pair[0], best_pair[1]
                    ] and union_fraction_mua_rate > max(
                        fraction_mua_rate_1, fraction_mua_rate_2
                    ):
                        # This is a red flag that one unit is likely a large mixture
                        # and we should ignore spike count
                        if rn_verbose:
                            print("This flagged as a large mixture")
                        if (1 - neuron_2["fraction_mua"]) * neuron_2["snr"] > (
                            1 - neuron_1["fraction_mua"]
                        ) * neuron_1["snr"]:
                            # Neuron 2 has better MUA and SNR so pick it
                            delete_1 = True
                            delete_2 = False
                        else:
                            # Neuron 1 has better MUA and SNR so pick it
                            delete_1 = False
                            delete_2 = True

            if delete_1:
                if rn_verbose:
                    print(
                        "Choosing from neurons with channels",
                        neuron_1["channel"],
                        neuron_2["channel"],
                    )
                if rn_verbose:
                    print(
                        "Deleting neuron 1 with violators",
                        violation_partners[best_pair[0]],
                        "MUA",
                        neuron_1["fraction_mua"],
                        "snr",
                        neuron_1["snr"],
                        "n spikes",
                        neuron_1["spike_indices"].shape[0],
                    )
                if rn_verbose:
                    print(
                        "Keeping neuron 2 with violators",
                        violation_partners[best_pair[1]],
                        "MUA",
                        neuron_2["fraction_mua"],
                        "snr",
                        neuron_2["snr"],
                        "n spikes",
                        neuron_2["spike_indices"].shape[0],
                    )
                neurons_remaining_indices.remove(best_pair[0])
                for vp in violation_partners:
                    vp.discard(best_pair[0])
                # Assign current neuron not to anything since
                # it is designated as trash for deletion
                neurons[best_pair[0]]["deleted_as_redundant"] = True
            if delete_2:
                if rn_verbose:
                    print(
                        "Choosing from neurons with channels",
                        neuron_1["channel"],
                        neuron_2["channel"],
                    )
                if rn_verbose:
                    print(
                        "Keeping neuron 1 with violators",
                        violation_partners[best_pair[0]],
                        "MUA",
                        neuron_1["fraction_mua"],
                        "snr",
                        neuron_1["snr"],
                        "n spikes",
                        neuron_1["spike_indices"].shape[0],
                    )
                if rn_verbose:
                    print(
                        "Deleting neuron 2 with violators",
                        violation_partners[best_pair[1]],
                        "MUA",
                        neuron_2["fraction_mua"],
                        "snr",
                        neuron_2["snr"],
                        "n spikes",
                        neuron_2["spike_indices"].shape[0],
                    )
                neurons_remaining_indices.remove(best_pair[1])
                for vp in violation_partners:
                    vp.discard(best_pair[1])
                # Assign current neuron not to anything since
                # it is designated as trash for deletion
                neurons[best_pair[1]]["deleted_as_redundant"] = True
