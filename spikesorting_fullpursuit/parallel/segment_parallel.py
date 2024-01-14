import numpy as np


def keep_valid_inds(keep_data_list: list, valid_inds):
    """

    Args:
        Keep_data_list: List of numpy arrays, such as event indicies or neuron labels
        valid_inds: 1D numpy array of booleans, which specify which elements of 
            keep_data_list are valid

    Returns: Either a tuple of numpy arrays or a single numpy array
        that contains only the valid elements of keep_data_list

    """

    out_data = []
    for data in keep_data_list:
        out_data.append(data[valid_inds])
    return tuple(x for x in out_data) if len(keep_data_list) > 1 else out_data[0]


def memmap_to_mem(memmap, dtype=None, order=None):
    """ 
    Helpful function that takes a numpy memmap as input and copies it to a
    numpy array in memory as output. 
    """
    if not isinstance(memmap, np.memmap):
        raise ValueError("Input object is not instance of numpy.memmap")
    if dtype is None:
        dtype = memmap.dtype
    if order is None:
        order = "F" if memmap.flags['F_CONTIGUOUS'] else "C"
    mem = np.empty(memmap.shape, dtype=dtype, order=order)
    np.copyto(mem, memmap)

    return mem


def get_zero_phase_kernel(x, x_center):
    """
    Zero pads the 1D kernel x, so that it is aligned with the current element
    of x located at x_center.  This ensures that convolution with the kernel
    x will be zero phase with respect to x_center.
    """

    kernel_offset = (x.size
                     - 2 * x_center
                     - 1)  # -1 To center ON the x_center index
    kernel_size = np.abs(kernel_offset) + x.size
    if kernel_size // 2 == 0:  # Kernel must be odd
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


