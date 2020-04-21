import numpy as np
from scipy import signal
from scipy import stats
from spikesorting_python.src import electrode
from spikesorting_python.src import spikesorting
from spikesorting_python.src.parallel import spikesorting_parallel
from spikesorting_python.src import consolidate
import matplotlib.pyplot as plt



dict_other_templates = {'upsidedown_47a': np.array(
      [-0.1513909 , -0.16631116, -0.17847979, -0.18862691, -0.19872002,
       -0.20880426, -0.21737931, -0.22361187, -0.22874592, -0.23452372,
       -0.24082686, -0.24489726, -0.24255492, -0.22836725, -0.19307666,
       -0.1191242 ,  0.02425   ,  0.28417833,  0.71488001,  1.34970811,
        2.15717331,  3.00861469,  3.70470693,  4.06647289,  4.02591484,
        3.64727615,  3.07093063,  2.43142789,  1.8080954 ,  1.2247536 ,
        0.67450634,  0.14624459, -0.36222091, -0.84091953, -1.2741004 ,
       -1.6452294 , -1.94012256, -2.14839704, -2.26756383, -2.30440089,
       -2.27405578, -2.19505695, -2.0840863 , -1.95470787, -1.81895549,
       -1.68644032, -1.56235016, -1.44811373, -1.3434583 , -1.24678163,
       -1.15489337, -1.06493637, -0.97600263, -0.88764435, -0.79968051,
       -0.71323625, -0.62895673, -0.54701365, -0.46780424, -0.39222975,
       -0.32111773, -0.25396762, -0.19020192, -0.13038235, -0.07453424,
       -0.02132602,  0.02993237,  0.07660115,  0.11522204,  0.14610511,
        0.17122906,  0.19176712,  0.20754389]),
        'upsidedown_47b': np.array(
        [-0.1513909 , -0.16631116, -0.17847979, -0.18862691, -0.19872002,
       -0.20880426, -0.21737931, -0.22361187, -0.22874592, -0.23452372,
       -0.24082686, -0.24489726, -0.24255492, -0.22836725, -0.19307666,
       -0.1191242 ,  0.02425   ,  0.28417833,  0.71488001,  1.34970811,
        2.15717331,  3.00861469,  3.70470693,  4.06647289,  4.02591484,
        3.64727615,  3.07093063,  2.43142789,  1.8080954 ,  1.2247536 ,
        0.67450634,  0.14624459, -0.36222091, -0.84091953, -1.2741004 ,
       -1.6452294 , -1.94012256, -2.14839704, -2.26756383, -2.30440089,
       -2.27405578, -2.19505695, -2.0840863 , -1.95470787, -1.81895549,
       -1.68644032, -1.56235016, -1.44811373, -1.3434583 , -1.24678163,
       -1.15489337, -1.06493637, -0.97600263, -0.88764435, -0.79968051,
       -0.71323625, -0.62895673, -0.54701365, -0.46780424, -0.39222975,
       -0.32111773, -0.25396762, -0.19020192, -0.13038235, -0.07453424,
       -0.02132602,  0.02993237,  0.07660115,  0.11522204,  0.14610511,
        0.17122906,  0.19176712,  0.20754389])}


class TestProbe(electrode.AbstractProbe):

    def __init__(self, sampling_rate, voltage_array, num_channels=None):
        if num_channels is None:
            num_channels = voltage_array.shape[1]
        electrode.AbstractProbe.__init__(self, sampling_rate, num_channels, voltage_array=voltage_array)

    def get_neighbors(self, channel):
        """ Test probe neighbor function simple returns an array of all channel numbers.
            """

        # start = max(channel - 2, 0)
        # stop = min(channel + 3, 4)
        start = 0
        stop = 4

        return np.arange(start, stop)


class TestDataset(object):
    """ Actual IDs are output as the index at (self.neuron_templates.shape[1] // 2)
    """
    def __init__(self, num_channels, duration, random_seed=None,
                 neuron_templates=None, frequency_range=(300, 6000),
                 samples_per_second=40000, amplitude=1, percent_shared_noise=0):
        self.num_channels = num_channels
        self.duration = duration
        self.frequency_range = frequency_range
        self.samples_per_second = samples_per_second
        self.amplitude = amplitude
        if neuron_templates is not None:
            self.neuron_templates = neuron_templates
        else:
            self.neuron_templates = self.get_default_templates()
        self.voltage_array = None
        self.actual_IDs = []
        self.percent_shared_noise = percent_shared_noise
        if self.percent_shared_noise > 1:
            self.percent_shared_noise = 1.
        if self.percent_shared_noise < 0.:
            self.percent_shared_noise = 0.
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.random_state = np.random.get_state()
        np.random.set_state(self.random_state)

    def gen_poisson_spiketrain(self, firing_rate, tau_ref=1.5e-3):
        """gen_poission_spiketrain(rate, refactory period, samples per second [, duration])
        Generate a poisson distributed neuron spike train given the passed firing rate (in Hz).
        The refactory period (in sec) determines the silency period between two adjacent spikes.
        Samples per seconds defines the sampling rate of output spiketrain. """
        number_of_samples = int(np.round(self.duration * self.samples_per_second))
        spiketrain = np.zeros(number_of_samples, dtype=np.bool)
        random_nums = np.random.random(spiketrain.size)

        if tau_ref < 0:
            tau_ref = 0
        elif tau_ref > 0:
            # Adjust the firing rate to take into account the refactory period
            firing_rate = firing_rate * 1.0 / (1.0 - firing_rate * tau_ref)

        # Generate the spiketrain
        last_spike_index = -np.inf
        for i in range(0, spiketrain.size):
            if (i - last_spike_index) / self.samples_per_second > tau_ref:
                spiketrain[i] = random_nums[i] < firing_rate / self.samples_per_second
                if spiketrain[i]:
                    last_spike_index = i
        return spiketrain

    def get_default_templates(self):
        templates = np.array([
            [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.001, -0.0, -0.0, 0.0, 0.0, 0.001, 0.002, 0.005, 0.01, 0.019, 0.034, 0.057, 0.1, 0.166, 0.222, 0.167, -0.088, -0.49, -0.855, -1.0, -0.862, -0.526, -0.151, 0.136, 0.293, 0.339, 0.312, 0.251, 0.187, 0.138, 0.109, 0.093, 0.084, 0.076, 0.069, 0.063, 0.058, 0.056, 0.054, 0.052, 0.051, 0.05, 0.049, 0.049, 0.048, 0.048, 0.047, 0.047, 0.046, 0.046, 0.045, 0.044, 0.043, 0.042, 0.041, 0.04, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.03, 0.029, 0.027, 0.026, 0.025],
            [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.001, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005, 0.005, 0.006, 0.007, 0.007, 0.008, 0.009, 0.011, 0.013, 0.017, 0.024, 0.033, 0.045, 0.062, 0.084, 0.091, 0.033, -0.166, -0.513, -0.858, -1.0, -0.868, -0.555, -0.212, 0.057, 0.218, 0.285, 0.293, 0.274, 0.248, 0.224, 0.204, 0.187, 0.173, 0.16, 0.149, 0.14, 0.132, 0.124, 0.117, 0.11, 0.103, 0.097, 0.091, 0.086, 0.081, 0.076, 0.072, 0.067, 0.064, 0.06, 0.056, 0.053, 0.05, 0.046, 0.043, 0.04, 0.038, 0.035, 0.033, 0.031, 0.028, 0.026, 0.023, 0.021, 0.019, 0.017, 0.015, 0.013, 0.012],
            [-0.108, -0.11, -0.111, -0.112, -0.112, -0.11, -0.107, -0.105, -0.104, -0.101, -0.099, -0.097, -0.095, -0.092, -0.088, -0.083, -0.077, -0.071, -0.065, -0.062, -0.058, -0.053, -0.045, -0.037, -0.03, -0.022, -0.013, -0.003, 0.01, 0.027, 0.045, 0.059, 0.071, 0.081, 0.092, 0.106, 0.12, 0.133, 0.145, 0.158, 0.171, 0.19, 0.215, 0.245, 0.276, 0.305, 0.351, 0.453, 0.642, 0.877, 1.0, 0.883, 0.647, 0.45, 0.342, 0.295, 0.269, 0.242, 0.214, 0.189, 0.171, 0.158, 0.146, 0.134, 0.12, 0.107, 0.094, 0.083, 0.072, 0.06, 0.049, 0.036, 0.022, 0.008, -0.006, -0.019, -0.027, -0.033, -0.04, -0.048, -0.054, -0.059, -0.063, -0.069, -0.073, -0.079, -0.086, -0.092, -0.095, -0.096, -0.098, -0.101, -0.105, -0.105, -0.105, -0.105, -0.105, -0.107, -0.109, -0.111]
            ])
        templates /= np.amax(np.abs(templates), axis=1)[:, None]
        return templates

    def gen_bandlimited_noise(self):
        """gen_bandlimited_noise([frequency_range, samples_per_second, duration, amplitude])
        Generates a timeseries of bandlimited Gaussian noise. This function uses the inverse FFT to generate
        noise in a given bandwidth. Within this bandwidth, the amplitude of each frequency is approximately
        constant (i.e., white), but the phase is random. The amplitude of the output signal is scaled so that
        99% of the values fall within the amplitude criteron (3 standard deviations of zero mean)."""
        freqs = np.abs(np.fft.fftfreq(self.samples_per_second * self.duration, 1/self.samples_per_second))
        f = np.zeros(self.samples_per_second * self.duration)
        idx = np.where(np.logical_and(freqs>=self.frequency_range[0], freqs<=self.frequency_range[1]))[0]
        f[idx] = 1
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        bandlimited_noise = np.fft.ifft(f).real
        bandlimited_noise = bandlimited_noise * self.amplitude / (3 * np.std(bandlimited_noise))
        return bandlimited_noise

    def gen_noise_voltage_array(self):
        noise_voltage_array = np.zeros((self.num_channels, self.samples_per_second*self.duration))
        if self.percent_shared_noise > 0:
            # Make a shared array
            shared_noise = self.percent_shared_noise * self.gen_bandlimited_noise()
            for i in range (0, self.num_channels):
                noise_voltage_array[i, :] = shared_noise
        for i in range (0, self.num_channels):
            noise_voltage_array[i, :] += (1-self.percent_shared_noise) * self.gen_bandlimited_noise()
        return noise_voltage_array

    def gen_test_dataset(self, firing_rates, template_inds, chan_scaling_factors, refractory_wins=1.5e-3):
        """
        """
        if len(firing_rates) != len(template_inds) or len(firing_rates) != len(chan_scaling_factors):
            raise ValueError("Input rates, templates, and scaling factors must all be the same length!")

        refractory_wins = np.array(refractory_wins)
        if refractory_wins.size == 1:
            refractory_wins = np.repeat(refractory_wins, len(firing_rates))
        # Reset neuron actual IDs for each neuron
        self.actual_IDs = [[] for neur in range(0, len(firing_rates))]
        self.actual_templates = [np.empty((0, self.neuron_templates.shape[1])) for neur in range(0, len(firing_rates))]
        voltage_array = self.gen_noise_voltage_array()
        for neuron in range(0, len(firing_rates)):
            # Generate one spike train for each neuron
            spiketrain = self.gen_poisson_spiketrain(firing_rate=firing_rates[neuron], tau_ref=refractory_wins[neuron])
            self.actual_IDs[neuron] = np.where(spiketrain)[0]
            for chan in range(0, self.num_channels):
                # Apply spike train to every channel this neuron is present on
                convolve_kernel = chan_scaling_factors[neuron][chan] * self.neuron_templates[template_inds[neuron], :] * self.amplitude
                self.actual_templates[neuron] = np.vstack((self.actual_templates[neuron], convolve_kernel))
                if chan_scaling_factors[neuron][chan] <= 0:
                    continue
                voltage_array[chan, :] += signal.fftconvolve(spiketrain, convolve_kernel, mode='same')
        self.voltage_array = voltage_array

    def drift_template_preview(self, template_inds, drift_funs, index):
        if len(template_inds) != len(drift_funs):
            raise ValueError("Input templates and drift functions must all be the same length!")
        if len(drift_funs[0](0)) != self.num_channels:
            raise ValueError("Input drift functions must return a scaling factor for each channel. Expected {0} scaling factors, got {1}.".format(self.num_channels, len(drift_funs[0](0))))
        preview_templates = [np.empty((0, self.neuron_templates.shape[1])) for neur in range(0, len(template_inds))]
        for neuron in range(0, len(template_inds)):
            chan_scaling_factors = drift_funs[neuron](index)
            for chan in range(0, self.num_channels):
                preview_templates[neuron] = np.vstack((preview_templates[neuron],
                    chan_scaling_factors[chan] * self.neuron_templates[template_inds[neuron], :] * self.amplitude))
        w_color = ['b', 'r', 'g']
        for n in range(0, len(preview_templates)):
            use_color = w_color.pop(0)
            _ = plt.plot(preview_templates[n].flatten(), color=use_color)
        ag = plt.gcf()
        ag.set_size_inches(15, 12)
        plt.show()
        return

    def gen_test_dataset_with_drift(self, firing_rates, template_inds, drift_funs,
                refractory_wins=1.5e-3, scaled_spike_thresh=None):
        """drift_funs must be a LIST of functions.
        """
        if len(firing_rates) != len(template_inds) or len(firing_rates) != len(drift_funs):
            raise ValueError("Input rates, templates, and drift functions must all be the same length!")
        if len(drift_funs[0](0)) != self.num_channels:
            raise ValueError("Input drift functions must return a scaling factor for each channel. Expected {0} scaling factors, got {1}.".format(self.num_channels, len(drift_funs[0](0))))

        if scaled_spike_thresh is None:
            scaled_spike_thresh = 2 * self.amplitude
        self.scaled_spike_thresh = scaled_spike_thresh

        refractory_wins = np.array(refractory_wins)
        if refractory_wins.size == 1:
            refractory_wins = np.repeat(refractory_wins, len(firing_rates))
        # Reset neuron actual IDs for each neuron
        self.actual_IDs = [[] for neur in range(0, len(firing_rates))]
        self.actual_templates = [np.empty((0, self.neuron_templates.shape[1])) for neur in range(0, len(firing_rates))]
        half_temp_width = (self.neuron_templates.shape[1] // 2)
        voltage_array = self.gen_noise_voltage_array()
        for neuron in range(0, len(firing_rates)):
            # Store templates used, use scaling at time zero
            chan_scaling_factors = drift_funs[neuron](0)
            for chan in range(0, self.num_channels):
                self.actual_templates[neuron] = np.vstack((self.actual_templates[neuron],
                    chan_scaling_factors[chan] * self.neuron_templates[template_inds[neuron], :] * self.amplitude))
            # Generate one spike train for each neuron
            spiketrain = self.gen_poisson_spiketrain(firing_rate=firing_rates[neuron], tau_ref=refractory_wins[neuron])
            spiketrain[0:half_temp_width+2] = False # Ensure spike times will not overlap beginning
            spiketrain[-(half_temp_width+2):] = False # or overlap end
            self.actual_IDs[neuron] = np.nonzero(spiketrain)[0] - half_temp_width
            remove_IDs = np.zeros(self.actual_IDs[neuron].size, dtype=np.bool)
            for i, spk_ind in enumerate(self.actual_IDs[neuron]):
                chan_scaling_factors = drift_funs[neuron](spk_ind)
                if np.all(chan_scaling_factors < scaled_spike_thresh):
                    remove_IDs[i] = True
                    # continue
                for chan in range(0, self.num_channels):
                    # Apply spike train to every channel this neuron is present on
                    scaled_template = chan_scaling_factors[chan] * self.neuron_templates[template_inds[neuron], :] * self.amplitude
                    voltage_array[chan, spk_ind:(spk_ind+self.neuron_templates.shape[1])] += scaled_template
            print("Removing", np.count_nonzero(remove_IDs), "neuron", neuron, "spikes for scale factors less than", scaled_spike_thresh)
            self.actual_IDs[neuron] = self.actual_IDs[neuron][~remove_IDs]
            self.actual_IDs[neuron] += half_temp_width # Re-center the spike times
        self.voltage_array = voltage_array

    def assess_performance():
        pass

    def sort_test_dataset(self, kwargs):

        spike_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                              'p_value_cut_thresh': 0.01, 'check_components': None,
                              'max_components': 10,
                              'min_firing_rate': 1, 'do_binary_pursuit': False,
                              'add_peak_valley': False, 'do_ZCA_transform': True,
                              'do_branch_PCA': True, 'use_GPU': True,
                              'max_gpu_memory': None,
                              'use_rand_init': True, 'cleanup_neurons': False,
                              'verbose': True}
        for key in kwargs:
            spike_sort_kwargs[key] = kwargs[key]

        self.Probe = TestProbe(self.samples_per_second, self.voltage_array, self.num_channels)
        sort_data, work_items, sort_info = spikesorting.spike_sort(self.Probe, **spike_sort_kwargs)

        return sort_data, work_items, sort_info

    def sort_test_dataset_parallel(self, kwargs):

        spike_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                           'filter_band': self.frequency_range,
                           'p_value_cut_thresh': 0.01, 'check_components': None,
                           'max_components': 10,
                           'min_firing_rate': 1, 'do_binary_pursuit': False,
                           'add_peak_valley': False, 'do_branch_PCA': True,
                           'use_GPU': True, 'max_gpu_memory': None,
                           'save_1_cpu': True, 'use_rand_init': True,
                           'cleanup_neurons': False,
                           'verbose': True,
                           'test_flag': True, 'log_dir': None,
                           'do_ZCA_transform': True}
        for key in kwargs:
            spike_sort_kwargs[key] = kwargs[key]

        self.Probe = TestProbe(self.samples_per_second, self.voltage_array, self.num_channels)
        sort_data, work_items, sort_info = spikesorting_parallel.spike_sort_parallel(self.Probe, **spike_sort_kwargs)

        return sort_data, work_items, sort_info


    def compare_single_vs_parallel(self, kwargs):

        single_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                              'p_value_cut_thresh': 0.01, 'check_components': None,
                              'max_components': 10,
                              'min_firing_rate': 1, 'do_binary_pursuit': False,
                              'add_peak_valley': False, 'do_ZCA_transform': True,
                              'do_branch_PCA': True, 'use_GPU': True,
                              'max_gpu_memory': None,
                              'use_rand_init': True,
                              'cleanup_neurons': False,
                              'verbose': True}
        for key in kwargs:
            single_sort_kwargs[key] = kwargs[key]
        par_sort_kwargs = {'sigma': 4., 'clip_width': [-6e-4, 8e-4],
                           'filter_band': self.frequency_range,
                           'p_value_cut_thresh': 0.01, 'check_components': None,
                           'max_components': 10,
                           'min_firing_rate': 1, 'do_binary_pursuit': False,
                           'add_peak_valley': False, 'do_branch_PCA': True,
                           'use_GPU': True, 'max_gpu_memory': None,
                           'save_1_cpu': True, 'use_rand_init': True,
                           'cleanup_neurons': False,
                           'verbose': True,
                           'test_flag': True, 'log_dir': None,
                           'do_ZCA_transform': True}
        for key in single_sort_kwargs.keys():
            par_sort_kwargs[key] = single_sort_kwargs[key]

        # Enforce test_flag else this will almost surely fail
        par_sort_kwargs['test_flag'] = True
        # Setup electrode TestProbe
        self.Probe = TestProbe(self.samples_per_second, self.voltage_array, self.num_channels)

        # Set and save random generator state
        self.random_state = np.random.get_state()
        first_state = self.random_state
        np.random.set_state(first_state)
        sort_data, work_items, sort_info = spikesorting.spike_sort(self.Probe, **single_sort_kwargs)

        # Make default work summary to check and organize data
        single_wis = consolidate.WorkItemSummary(sort_data, work_items, sort_info)


        np.random.set_state(first_state)
        sort_data, work_items, sort_info = spikesorting_parallel.spike_sort_parallel(self.Probe, **par_sort_kwargs)
        self.random_state = first_state

        parallel_wis = consolidate.WorkItemSummary(sort_data, work_items, sort_info)

        for key in single_wis.sort_info.keys():
            assert np.all(single_wis.sort_info[key] == parallel_wis.sort_info[key])
        for key in parallel_wis.sort_info.keys():
            if key in ['log_dir', 'test_flag', 'save_1_cpu']:
                continue
            assert np.all(single_wis.sort_info[key] == parallel_wis.sort_info[key])
        print("All sort info is the SAME.")

        assert len(single_wis.sort_data) == len(parallel_wis.sort_data), "Number of sort_data elements are different"

        # WorkItemSummary rearranges these as a list of channels by segments
        for chan in range(0, len(single_wis.sort_data)):
            # Iterate over zipped segments
            for s, p in zip(single_wis.sort_data[chan], parallel_wis.sort_data[chan]):
                for data_ind in range(0, len(s)):
                    try:
                        assert np.allclose(s[data_ind], p[data_ind], equal_nan=True), "Data item {0} {1} on chan {2} !!! DIFFERENT !!!".format(s[4], data_ind, chan)
                    except:
                        print("Data item {0} {1} on chan {2} !!! DIFFERENT (probably in size) !!!".format(s[4], data_ind, chan))
                        raise
            print("All seg data on chan", chan, "is the SAME.")

        assert len(single_wis.work_items) == len(parallel_wis.work_items), "Number of work_items elements are different"

        # WorkItemSummary rearranges these as a list of channels by segments
        for chan in range(0, len(single_wis.sort_data)):
            # Iterate over zipped segments
            for s, p in zip(single_wis.work_items[chan], parallel_wis.work_items[chan]):
                for key in s.keys():
                    assert np.all(s[key] == p[key]), "Data item {0} {1} on chan {2} !!! DIFFERENT !!!".format(s['ID'], key, chan)
                for key in p.keys():
                    assert np.all(s[key] == p[key]), "Data item {0} {1} on chan {2} !!! DIFFERENT !!!".format(s['ID'], key, chan)
            print("All seg work items on chan", chan, "are the SAME.")

        return single_wis, parallel_wis
