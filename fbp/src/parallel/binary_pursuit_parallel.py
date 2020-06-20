import numpy as np
import os
import platform as sys_platform
import re
import time
from fbp.src.sort import reorder_labels
from fbp.src import segment
from fbp.src.parallel import segment_parallel
from fbp.src.consolidate import find_overlapping_spike_bool
from scipy.signal import fftconvolve



def get_zero_phase_kernel(x, x_center):
    """ Zero pads the 1D kernel x, so that it is aligned with the current element
        of x located at x_center.  This ensures that convolution with the kernel
        x will be zero phase with respect to x_center.
    """

    kernel_offset = x.size - 2 * x_center - 1 # -1 To center ON the x_center index
    kernel_size = np.abs(kernel_offset) + x.size
    if kernel_size // 2 == 0: # Kernel must be odd
        kernel_size -= 1
        kernel_offset -= 1
    kernel = np.zeros(kernel_size)
    if kernel_offset > 0:
        kernel_slice = slice(kernel_offset, kernel.size)
    elif kernel_offset < 0:
        kernel_slice = slice(0, kernel.size + kernel_offset)
    else:
        kernel_slice = slice(0, kernel.size)
    kernel[kernel_slice] = x

    return kernel


def binary_pursuit(templates, voltage, template_labels, sampling_rate, v_dtype,
                   clip_width, template_samples_per_chan, thresh_sigma=1.645,
                   kernels_path=None, max_gpu_memory=None):
    """
    	binary_pursuit_opencl(voltage, crossings, labels, clips)

    Uses OpenCL on a GPU to identify spikes that overlap with already identified
    spikes. This also will identify spikes that are less than the threshold.
    The algorithm is based, loosely, on work by Johnathan Pillow (Princeton)
    published in Plos One (2013): A Model-Based Spike Sorting Algorithm for
    Removing Correlation Artifacts in Multi-Neuron Recordings.

    Computes the mean squared error (MSE) for every spike at every instance in
    time in the voltage trace and asks whether the addition of a particular spike
    template at each point in time would reduce the overall MSE. This function essentially
    separates overlapping spikes by finding the spike times for each template that
    minimize the squared error.

    The output of this function is a new set of crossing times, labels, and clips.
    The clips have all other spikes removed. The clips and binary pursuit are all
    done in np.float32 precision. Output clips are cast to templates.dtype
    at the end for output.

    A couple of notes regarding the OpenCL implementation of the binary pursuit algorithm.
     1. The labels passed into the both the residual computation and the binary pursuit
        OpenCL kernels *MUST* refer to their row in the templates array, in standard C
        (base 0) notation. That is, the first row of templates should be assigned a label of
        0, the second row should have a label of 1 and so on.
     2. The spike indices passed into the kernels must refer to the index of the onset of the
        template. If the index refers to an alignment point in the middle of the clip (e.g.,
        0.3ms after the onset of the clip), the spike indices must be shifted back to refer
        to the start of the clip before calling either kernel.
     4. The templates passed into the OpenCL kernels must be a vector not a matrix. Initially,
        we compute the templates for each label as MxN vector (M being the number of units, N
        being the number of points in each template). Before calling the CL kernels, this must
        be converted to 1x(M*N) vector where the first N points refer to the first template.

    """
    ############################################################################
    # Must reserve all references to pyopencl to be inside this function.
    # Otherwise importing by main calling function (spikesorting_parallel)
    # instantiates pyopencl in the host process and blocks the children
    import pyopencl as cl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    ############################################################################
    # Ease of use variables
    n_chans = np.uint32(voltage.shape[0])
    n_samples = voltage.shape[1]
    chan_win, clip_width = segment_parallel.time_window_to_samples(clip_width, sampling_rate)

    # Templates must be float32
    templates = np.float32(templates)
    # Reshape our templates so that instead of being an MxN array, this
    # becomes a 1x(M*N) vector. The first template should be in the first N points
    templates_vector = templates.reshape(templates.size).astype(np.float32)

    # Load OpenCL code from a file (stored as a string)
    if kernels_path is None:
        current_path = os.path.realpath(__file__)
        if sys_platform.system() == 'Windows':
            kernels_path = current_path.split('\\')
            # slice at -2 if parallel folder is same level as kernels folder
            kernels_path = [x + '\\' for x in kernels_path[0:-2]]
            kernels_path = ''.join(kernels_path) + 'kernels\\binary_pursuit.cl'
        else:
            kernels_path = current_path.split('/')
            # slice at -2 if parallel folder is same level as kernels folder
            kernels_path = [x + '/' for x in kernels_path[0:-2]]
            kernels_path = ''.join(kernels_path) + 'kernels/binary_pursuit.cl'
    with open(kernels_path, 'r') as fp:
        kernels = fp.read()

    # Search for a platform with a GPU device, and pick GPU with most global memory
    platforms = cl.get_platforms()
    device = None
    max_global_mem = 0
    for platform in platforms:
        devices = platform.get_devices(cl.device_type.GPU)
        if len(devices) > 0:
            for d in range(0, len(devices)):
                if devices[d].get_info(cl.device_info.GLOBAL_MEM_SIZE) > max_global_mem:
                    device = devices[d]
    if device is None:
        raise RuntimeError("No GPU available")

    # Create a context and build our program
    ctx = cl.Context([device])
    mf = cl.mem_flags
    with cl.CommandQueue(ctx) as queue:
        prg = cl.Program(ctx, kernels)
        prg.build() # David has a bunch of PYOPENCL_BUILD_OPTIONS used here

        # Adjust event_indices so that they refer to the start of the clip to be
        # subtracted.
        clip_init_samples = int(np.abs(chan_win[0]))
        template_labels = np.uint32(template_labels)

        # Determine the segment size that we need to use to fit into the
        # memory on this GPU device.
        gpu_memory_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) # Total memory size in bytes
        # Save 50% or 1GB, whichever is less for the operating system to use
        # assuming this is not a headless system.
        gpu_memory_size = max(gpu_memory_size * 0.5, gpu_memory_size - (1024 * 1024 * 1024))
        if max_gpu_memory is not None:
            gpu_memory_size = min(gpu_memory_size, max_gpu_memory)
        # Windows 10 only allows 80% of the memory on the graphics card to be used,
        # the rest is reserved for WDDMv2. See discussion here:
        # https://www.nvidia.com/en-us/geforce/forums/geforce-graphics-cards/5/269554/windows-10wddm-grabbing-gpu-vram/
        regex_version = re.search('[0-9][0-9][.]', sys_platform.version())
        if (sys_platform.system() == 'Windows') and (float(regex_version.group()) >= 10.):
            gpu_memory_size = min(device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * 0.75,
                                  gpu_memory_size)
        assert gpu_memory_size > 0

        # Estimate the number of bytes that 1 second of data take up
        # This is skipping the bias and template squared error buffers which are small
        constant_memory_usage = templates_vector.nbytes + template_labels.nbytes
        # Usage for voltage
        memory_usage_per_second = (n_chans * sampling_rate * np.dtype(np.float32).itemsize)
        # Usage for new spike info storage buffers depends on samples divided by template size
        memory_usage_per_second += (sampling_rate / template_samples_per_chan) * \
                                    (np.dtype(np.uint32).itemsize + # additional indices
                                     np.dtype(np.uint32).itemsize + # additional labels
                                     np.dtype(np.uint32).itemsize + # best indices
                                     np.dtype(np.uint32).itemsize + # best labels
                                     np.dtype(np.float32).itemsize) # best likelihoods

        # Estimate how much data we can look at in each data 'chunk'
        num_seconds_per_chunk = (gpu_memory_size - constant_memory_usage) \
                                / (memory_usage_per_second)
        # Do not exceed the length of a buffer in bytes that can be addressed
        # for the single largest data buffer, the voltage
        # Note: there is also a max allocation size,
        # device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        max_addressable_seconds = (((1 << device.get_info(cl.device_info.ADDRESS_BITS)) - 1)
                                    / (np.dtype(np.float32).itemsize * sampling_rate
                                     * n_chans))
        num_seconds_per_chunk = min(num_seconds_per_chunk, np.floor(max_addressable_seconds))
        # Convert from seconds to indices, the actual currency of this function's computations
        num_indices_per_chunk = int(np.floor(num_seconds_per_chunk * sampling_rate))

        if num_indices_per_chunk < 4 * template_samples_per_chan:
            raise ValueError("Cannot fit enough data on GPU to run binary pursuit. Decrease neighborhoods and/or clip width.")

        # Get all of our kernels
        compute_residual_kernel = prg.compute_residual
        compute_template_maximum_likelihood_kernel = prg.compute_template_maximum_likelihood
        binary_pursuit_kernel = prg.binary_pursuit
        get_adjusted_clips_kernel = prg.get_adjusted_clips

        # Set local work size for both kernels to the max for device
        resid_local_work_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        pursuit_local_work_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

        # Windows 10 also seems to be fussy about the number of compute units occupied...
        regex_version = re.search('[0-9][0-9][.]', sys_platform.version())
        if (sys_platform.system() == 'Windows') and (float(regex_version.group()) >= 10.):
            # resid_local_work_size = compute_residual_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
            # pursuit_local_work_size = binary_pursuit_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
            max_enqueue_resid = np.uint32(resid_local_work_size * np.floor(device.max_compute_units * 0.5))
            max_enqueue_pursuit = np.uint32(pursuit_local_work_size * np.floor(device.max_compute_units * 0.5))
        else:
            max_enqueue_resid = np.uint32(resid_local_work_size * (device.max_compute_units))
            max_enqueue_pursuit = np.uint32(pursuit_local_work_size * (device.max_compute_units))
        max_enqueue_resid = max(resid_local_work_size, resid_local_work_size * (max_enqueue_resid // resid_local_work_size))
        max_enqueue_pursuit = max(pursuit_local_work_size, pursuit_local_work_size * (max_enqueue_pursuit // pursuit_local_work_size))

        # Set up our final outputs
        num_additional_spikes = np.zeros(1, dtype=np.uint32)

        # Set-up any buffers/lists that are not dependent on chunks
        template_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=templates_vector)
        fft_kernels = []
        secret_spike_indices = []
        secret_spike_labels = []
        secret_spike_bool = []
        adjusted_spike_clips = []

        # Compute our template sum squared error (see note below).
        # This is a num_templates x num_neighbors vector
        template_sum_squared = np.zeros(templates.shape[0] * n_chans, dtype=np.float32)
        for n in range(0, templates.shape[0]):
            for chan in range(0, n_chans):
                t_win = [chan*template_samples_per_chan, chan*template_samples_per_chan + template_samples_per_chan]
                template_sum_squared[(n*n_chans) + chan] = np.float32(-0.5 * np.dot(templates[n, t_win[0]:t_win[1]], templates[n, t_win[0]:t_win[1]]))
                # Also get the FFT kernels used for spike bias below while we're at it
                fft_kernels.append(get_zero_phase_kernel(templates[n, t_win[0]:t_win[1]], clip_init_samples))
        template_sum_squared_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=template_sum_squared)

        # Compute the template bias terms over all voltage data (fixed for each chunk)
        spike_biases = np.zeros(templates.shape[0] * n_chans, dtype=np.float32)
        # Compute bias separately for each neuron
        for n in range(0, templates.shape[0]):
            neighbor_bias = np.zeros(n_samples, dtype=np.float32)
            for chan in range(0, n_chans):
                if np.all(fft_kernels[n*n_chans + chan] == 0):
                    # Unit is not defined over this channel so skip
                    continue
                # Moved bias out here so voltage is not yet a 1D vector, so no
                # need to do all this stride indexing
                # cv_win = [chan * n_samples,
                #           chan * n_samples + n_samples]
                # neighbor_bias += np.float32(fftconvolve(
                #             voltage[cv_win[0]:cv_win[1]],
                #             fft_kernels[n*n_chans + chan],
                #             mode='same'))
                neighbor_bias += np.float32(fftconvolve(
                                                voltage[chan, :],
                                                fft_kernels[n*n_chans + chan],
                                                mode='same'))
            # Use MAD to estimate STD of the noise and set bias at
            # thresh_sigma standard deviations. The typical extremely large
            # n value for neighbor_bias makes this calculation converge to
            # normal distribution
            # Assumes zero-centered (which median usually isn't)
            MAD = np.median(np.abs(neighbor_bias))
            std_noise = MAD / 0.6745 # Convert MAD to normal dist STD
            total_bias = np.float32(thresh_sigma*std_noise)
            bias_per_chan = total_bias / n_chans
            # Distribute bias evenly over channels with data
            for chan in range(0, n_chans):
                # CAN'T DO THIS NOW BUT MAY IF WE GET SMART ABOUT NEIGHBORHOODS...
                # if np.all(fft_kernels[n*n_chans + chan] == 0):
                #     # Unit is not defined over this channel so skip
                #     spike_biases[(n*n_chans) + chan] = 0
                #     continue
                spike_biases[(n*n_chans) + chan] = bias_per_chan

        # Delete stuff no longer needed for this chunk
        # del residual_voltage
        del neighbor_bias

        # Determine our chunk onset indices, making sure that each new start
        # index overlaps the previous chunk by 3 template widths so that no
        # spikes are missed by binary pursuit
        # Note: Any spikes in the last template width of data are not checked
        chunk_onsets = []
        curr_onset = 0
        while ((curr_onset) < (n_samples - template_samples_per_chan)):
            chunk_onsets.append(curr_onset)
            curr_onset += num_indices_per_chunk - 3 * template_samples_per_chan
        print("Using ", len(chunk_onsets), " chunks for binary pursuit.", flush=True)

        # Loop over chunks
        for chunk_number, start_index in enumerate(chunk_onsets):
            stop_index = np.uint32(min(n_samples, start_index + num_indices_per_chunk))
            print("Starting chunk number", chunk_number, "from", start_index, "to", stop_index, flush=True)
            chunk_voltage = np.float32(voltage[:, start_index:stop_index])
            chunk_voltage_length = np.uint32(stop_index - start_index)
            # Reshape voltage over channels into a single 1D vector
            chunk_voltage = chunk_voltage.reshape(chunk_voltage.size)

            # We will find all spikes in binary pursuit so set chunk crossings to zero
            chunk_crossings = np.array([], dtype=np.uint32)
            chunk_labels = np.array([], dtype=np.uint32)

            # - Set up the compute_residual kernel -
            # Create our buffers on the graphics cards.
            # Essentially all we are doing is copying each of our arrays to the graphics card.
            voltage_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=chunk_voltage)

            # Set arguments that are the same every iteration
            compute_residual_kernel.set_arg(0, voltage_buffer) # location where chunk voltage is stored
            compute_residual_kernel.set_arg(1, chunk_voltage_length) # length of chunk voltage array
            compute_residual_kernel.set_arg(2, n_chans) # Number of neighboring channels
            compute_residual_kernel.set_arg(3, template_buffer) # location where our templates are stored
            compute_residual_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of neurons (`rows` in templates)
            compute_residual_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each channel of templates

            next_wait_event = None

            # Determine how many independent workers will work on binary pursuit
            # (Includes both template ML and binary pursuit kernels)
            num_template_widths = int(np.ceil(chunk_voltage_length / template_samples_per_chan))
            total_work_size_pursuit = pursuit_local_work_size * int(np.ceil(num_template_widths / pursuit_local_work_size))

            # Construct our buffers
            spike_biases_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=spike_biases)
            num_additional_spikes_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(1, dtype=np.uint32)) # NOTE: Must be :rw for atomic to work
            additional_spike_indices_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            additional_spike_labels_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            best_spike_likelihoods_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.float32))
            best_spike_labels_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            best_spike_indices_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            next_check_window_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            work_ids_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.arange(0, num_template_widths, 1, dtype=np.uint32))

            # This is separate storage for transferring data from next check, NOT A HOST (maybe could be)
            next_check_window = np.zeros(num_template_widths, dtype=np.uint32)

            # Set input arguments for template maximum likelihood kernel
            compute_template_maximum_likelihood_kernel.set_arg(0, voltage_buffer) # Voltage buffer created previously
            compute_template_maximum_likelihood_kernel.set_arg(1, chunk_voltage_length) # Length of chunk voltage
            compute_template_maximum_likelihood_kernel.set_arg(2, n_chans) # number of neighboring channels
            compute_template_maximum_likelihood_kernel.set_arg(3, template_buffer) # Our template buffer, created previously
            compute_template_maximum_likelihood_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of unique neurons, M
            compute_template_maximum_likelihood_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            compute_template_maximum_likelihood_kernel.set_arg(6, np.uint32(0)) # Template number
            compute_template_maximum_likelihood_kernel.set_arg(7, template_sum_squared_buffer) # Sum of squared templated
            compute_template_maximum_likelihood_kernel.set_arg(8, spike_biases_buffer) # Bias
            compute_template_maximum_likelihood_kernel.set_arg(9, best_spike_indices_buffer) # Storage for peak likelihood index
            compute_template_maximum_likelihood_kernel.set_arg(10, best_spike_labels_buffer) # Storage for peak likelihood label
            compute_template_maximum_likelihood_kernel.set_arg(11, best_spike_likelihoods_buffer) # Storage for peak likelihood value
            compute_template_maximum_likelihood_kernel.set_arg(12, work_ids_buffer) # Storage for peak likelihood value
            compute_template_maximum_likelihood_kernel.set_arg(13, np.uint32(num_template_widths))

            # Construct a local buffer (unsigned int * local_work_size)
            local_buffer = cl.LocalMemory(4 * pursuit_local_work_size)

            # Set input arguments for binary pursuit kernel
            binary_pursuit_kernel.set_arg(0, voltage_buffer) # Voltage buffer created previously
            binary_pursuit_kernel.set_arg(1, chunk_voltage_length) # Length of chunk voltage
            binary_pursuit_kernel.set_arg(2, n_chans) # number of neighboring channels
            binary_pursuit_kernel.set_arg(3, template_buffer) # Our template buffer, created previously
            binary_pursuit_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of unique neurons, M
            binary_pursuit_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            binary_pursuit_kernel.set_arg(6, template_sum_squared_buffer) # Sum of squared templated
            binary_pursuit_kernel.set_arg(7, spike_biases_buffer) # Bias
            binary_pursuit_kernel.set_arg(8, best_spike_indices_buffer) # Storage for peak likelihood index
            binary_pursuit_kernel.set_arg(9, best_spike_labels_buffer) # Storage for peak likelihood label
            binary_pursuit_kernel.set_arg(10, best_spike_likelihoods_buffer) # Storage for peak likelihood value
            binary_pursuit_kernel.set_arg(11, local_buffer) # Local buffer
            binary_pursuit_kernel.set_arg(12, num_additional_spikes_buffer) # Output, total number of additional spikes
            binary_pursuit_kernel.set_arg(13, additional_spike_indices_buffer) # Additional spike indices
            binary_pursuit_kernel.set_arg(14, additional_spike_labels_buffer) # Additional spike labels
            binary_pursuit_kernel.set_arg(15, np.uint32(num_template_widths)) # Additional spike labels
            binary_pursuit_kernel.set_arg(16, next_check_window_buffer) # Additional spike labels
            binary_pursuit_kernel.set_arg(17, work_ids_buffer) # Additional spike labels
            binary_pursuit_kernel.set_arg(18, np.uint32(num_template_widths))

            # Run the kernel until num_additional_spikes is zero
            chunk_total_additional_spikes = 0
            chunk_total_additional_spike_indices = []
            chunk_total_additional_spike_labels = []
            n_loops = 0
            print("pursuit total size is", total_work_size_pursuit, "local size is", pursuit_local_work_size, "with max enqueue", max_enqueue_pursuit, "chose n to enqueue", min(total_work_size_pursuit, max_enqueue_pursuit))
            while True:
                n_loops += 1
                if n_loops % 10 == 0:
                    print("Starting loop", n_loops, "for this chunk")
                n_to_enqueue = min(total_work_size_pursuit, max_enqueue_pursuit)
                for template_index in range(0, templates.shape[0]):
                    for enqueue_step in np.arange(0, total_work_size_pursuit, max_enqueue_pursuit, dtype=np.uint32):
                        # time.sleep(.1) # Giving OS a second here seems to help from timeout crashes ('Out of Resources Error')
                        compute_template_maximum_likelihood_kernel.set_arg(6, np.uint32(template_index)) # Template number
                        temp_ml_event = cl.enqueue_nd_range_kernel(queue,
                                              compute_template_maximum_likelihood_kernel,
                                              (n_to_enqueue, ), (pursuit_local_work_size, ),
                                              global_work_offset=(enqueue_step, ),
                                              wait_for=next_wait_event)
                        queue.finish()
                        next_wait_event = [temp_ml_event]

                n_to_enqueue = min(total_work_size_pursuit, max_enqueue_pursuit)
                for enqueue_step in np.arange(0, total_work_size_pursuit, max_enqueue_pursuit, dtype=np.uint32):
                    # time.sleep(.1)
                    pursuit_event = cl.enqueue_nd_range_kernel(queue,
                                          binary_pursuit_kernel,
                                          (n_to_enqueue, ), (pursuit_local_work_size, ),
                                          global_work_offset=(enqueue_step, ),
                                          wait_for=next_wait_event)
                    queue.finish()
                    next_wait_event = [pursuit_event]

                cl.enqueue_copy(queue, num_additional_spikes, num_additional_spikes_buffer, wait_for=next_wait_event)
                # print("Added", num_additional_spikes[0], "secret spikes", flush=True)

                if (num_additional_spikes[0] == 0):
                    break # Converged, no spikes added in last pass

                # Read out the data from next_check_window_buffer
                # Already waited for pursuit to finish above
                next_wait_event = [cl.enqueue_copy(queue, next_check_window, next_check_window_buffer, wait_for=None)]
                print("Next round has", np.count_nonzero(next_check_window))
                # Use next_check_window data to determine work_ids_buffer for next pass
                new_work_ids = np.uint32(np.nonzero(next_check_window)[0])
                # Free old work_ids and set new buffer
                work_ids_buffer.release()
                work_ids_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=new_work_ids)
                compute_template_maximum_likelihood_kernel.set_arg(12, work_ids_buffer)
                compute_template_maximum_likelihood_kernel.set_arg(13, np.uint32(new_work_ids.shape[0]))
                binary_pursuit_kernel.set_arg(17, work_ids_buffer)
                binary_pursuit_kernel.set_arg(18, np.uint32(new_work_ids.shape[0]))
                # Reset number of kernels to run for next pass
                total_work_size_pursuit = pursuit_local_work_size * int(np.ceil(new_work_ids.shape[0] / pursuit_local_work_size))
                # Reset next_check_window for next pass and copy to GPU
                next_check_window[:] = 0
                next_wait_event = [cl.enqueue_copy(queue, next_check_window_buffer, next_check_window, wait_for=next_wait_event)]

                # Read out and save all the new spikes we just found for this chunk
                additional_spike_indices = np.zeros(num_additional_spikes[0], dtype=np.uint32)
                additional_spike_labels = np.zeros(num_additional_spikes[0], dtype=np.uint32)
                next_wait_event = []
                # Already waited for pursuit to finish above
                next_wait_event.append(cl.enqueue_copy(queue, additional_spike_indices, additional_spike_indices_buffer, wait_for=None))
                next_wait_event.append(cl.enqueue_copy(queue, additional_spike_labels, additional_spike_labels_buffer, wait_for=None))
                chunk_total_additional_spike_indices.append(additional_spike_indices)
                chunk_total_additional_spike_labels.append(additional_spike_labels)

                # We have added additional spikes this pass, so we need to subtract them off
                # by calling the compute_residuals kernel. Most of the arguments are the
                # same as when we called it previously (e.g., voltage is the same buffer on the
                # GPU).
                compute_residual_kernel.set_arg(6, additional_spike_indices_buffer)
                compute_residual_kernel.set_arg(7, additional_spike_labels_buffer)
                compute_residual_kernel.set_arg(8, num_additional_spikes[0])

                total_work_size_resid = resid_local_work_size * int(np.ceil(
                                            (num_additional_spikes[0]) / resid_local_work_size))

                n_to_enqueue = min(total_work_size_resid, max_enqueue_resid)
                for enqueue_step in np.arange(0, total_work_size_resid, max_enqueue_resid, dtype=np.uint32):
                    # time.sleep(.1)
                    residual_event = cl.enqueue_nd_range_kernel(queue,
                                           compute_residual_kernel,
                                           (n_to_enqueue, ), (resid_local_work_size, ),
                                           global_work_offset=(enqueue_step, ),
                                           wait_for=next_wait_event)
                    queue.finish()
                    next_wait_event = [residual_event]
                # Ensure that num_additional_spikes is equal to zero for the next pass
                cl.enqueue_copy(queue, num_additional_spikes_buffer, np.zeros(1, dtype=np.uint32), wait_for=None)
                chunk_total_additional_spikes += num_additional_spikes[0]
                # if n_loops >= 2:
                #     print("BREAKING OUT")
                #     break
                num_additional_spikes[0] = 0

            additional_spike_indices_buffer.release()
            additional_spike_labels_buffer.release()
            spike_biases_buffer.release()
            best_spike_likelihoods_buffer.release()
            best_spike_labels_buffer.release()
            best_spike_indices_buffer.release()

            # Read out the adjusted spikes here before releasing
            # the residual voltage. Only do this if there are spikes to get clips of
            if (chunk_total_additional_spikes + chunk_crossings.shape[0]) > 0:
                if chunk_total_additional_spikes == 0:
                    all_chunk_crossings = chunk_crossings
                    all_chunk_labels = chunk_labels
                elif chunk_crossings.shape[0] == 0:
                    all_chunk_crossings = np.hstack(chunk_total_additional_spike_indices)
                    all_chunk_labels = np.hstack(chunk_total_additional_spike_labels)
                else:
                    all_chunk_crossings = np.hstack((np.hstack(chunk_total_additional_spike_indices), chunk_crossings))
                    all_chunk_labels = np.hstack((np.hstack(chunk_total_additional_spike_labels), chunk_labels))

                print("Found", chunk_total_additional_spikes, "secret spikes this chunk. Getting adjusted clips for", all_chunk_crossings.shape[0], "total spikes this chunk", flush=True)

                all_adjusted_clips = np.zeros((chunk_total_additional_spikes + chunk_crossings.shape[0]) * templates.shape[1], dtype=np.float32)
                all_chunk_crossings_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=all_chunk_crossings)
                all_chunk_labels_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=all_chunk_labels)
                all_adjusted_clips_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=all_adjusted_clips)

                get_adjusted_clips_kernel.set_arg(0, voltage_buffer)
                get_adjusted_clips_kernel.set_arg(1, chunk_voltage_length)
                get_adjusted_clips_kernel.set_arg(2, n_chans)
                get_adjusted_clips_kernel.set_arg(3, template_buffer)
                get_adjusted_clips_kernel.set_arg(4, np.uint32(templates.shape[0]))
                get_adjusted_clips_kernel.set_arg(5, np.uint32(template_samples_per_chan))
                get_adjusted_clips_kernel.set_arg(6, all_chunk_crossings_buffer)
                get_adjusted_clips_kernel.set_arg(7, all_chunk_labels_buffer)
                get_adjusted_clips_kernel.set_arg(8, np.uint32(all_chunk_crossings.shape[0]))
                get_adjusted_clips_kernel.set_arg(9, all_adjusted_clips_buffer)

                # Run the kernel using same sizes as for resid kernel
                total_work_size_clips = resid_local_work_size * int(np.ceil(
                                            (all_chunk_crossings.shape[0]) / resid_local_work_size))
                clip_events = []
                n_to_enqueue = min(total_work_size_clips, max_enqueue_resid)
                print("Getting adjusted clips", flush=True)
                for enqueue_step in np.arange(0, total_work_size_clips, max_enqueue_resid, dtype=np.uint32):
                    # time.sleep(.1)
                    clip_event = cl.enqueue_nd_range_kernel(queue,
                                           get_adjusted_clips_kernel,
                                           (n_to_enqueue, ), (resid_local_work_size, ),
                                           global_work_offset=(enqueue_step, ),
                                           wait_for=next_wait_event)
                    queue.finish()
                    next_wait_event = [clip_event]
                cl.enqueue_copy(queue, all_adjusted_clips, all_adjusted_clips_buffer, wait_for=clip_events)
                all_adjusted_clips = np.reshape(all_adjusted_clips, (all_chunk_crossings.shape[0], templates.shape[1]))

                # Save all chunk data and free spike dependent memory
                secret_spike_indices.append(all_chunk_crossings + start_index)
                secret_spike_labels.append(all_chunk_labels)
                adjusted_spike_clips.append(all_adjusted_clips)
                all_chunk_bool = np.zeros(all_chunk_crossings.shape[0], dtype=np.bool)
                all_chunk_bool[0:chunk_total_additional_spikes] = True
                secret_spike_bool.append(all_chunk_bool)

                all_chunk_crossings_buffer.release()
                all_chunk_labels_buffer.release()
                all_adjusted_clips_buffer.release()

            # Release buffers before going on to next chunk
            voltage_buffer.release()
            num_additional_spikes_buffer.release()

        template_buffer.release()
        template_sum_squared_buffer.release()

    if len(secret_spike_indices) > 0:
        event_indices = np.int64(np.hstack(secret_spike_indices))
        neuron_labels = np.int64(np.hstack(secret_spike_labels))
        adjusted_clips = (np.vstack(adjusted_spike_clips)).astype(v_dtype)
        binary_pursuit_spike_bool = np.hstack(secret_spike_bool)
        # Realign events with center of spike
        event_indices += clip_init_samples
        print("Found a total of", np.count_nonzero(binary_pursuit_spike_bool), "secret spikes", flush=True)
    else:
        # No spikes found and findall == True
        event_indices, neuron_labels, binary_pursuit_spike_bool, adjusted_clips = [], [], [], []
        print("Found a total of ZERO secret spikes", flush=True)

    return event_indices, neuron_labels, binary_pursuit_spike_bool, adjusted_clips