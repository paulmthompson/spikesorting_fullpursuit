import numpy as np
from scipy.fftpack import dct, fft, ifft


def initial_cluster_farthest(
        data,
        median_cluster_size,
        choose_percentile=0.95,
        n_random=0):
    """
    Create distance based cluster labels along the rows of data.

    Returns a vector containing the labels for each data point.

    Data are iteratively clustered based on Euclidean distance until the median
    number of points in each cluster is <= median_cluster_size or each point is
    the only member of its own cluster. For each iteration, a new cluster center
    is chosen. First, the distance of each point from its nearest cluster center
    is computed. Second, from these distances, the point lying at the 99th
    percentile is chosen to be a new cluster center.  Finally, all points closer
    to this center than their current cluster center are assigned to the new
    cluster and the process is repeated.  This is similar to the k-means++
    algorithm except that it is deterministic, always choosing at the 99th
    percentile.

    Parameters
    ----------
    data : numpy ndarray
        Each row of data will be treated as an observation and each column as a
        dimension over which distance will be computed.  Must be two dimensional.
    median_cluster_size : {int, float, ndarray etc.}
        Must be a single, scalar value regardless of type. New cluster centers
        will be added until the median number of points from data that are
        nearest a cluster center is less than or equal to this number (see
        Notes below).

    Returns
    -------
    labels : numpy ndarray of dtype int64
        A new array holding the numerical labels indicating the membership of
        each point input in data. Array is the same size as data.shape[0].
    """
    if data.ndim <= 1 or data.size == 1:
        # Only 1 spike so return 1 label!
        return np.zeros(1, dtype=np.int64)

    # Begin with a single cluster (all data belong to the same cluster)
    labels = np.zeros((data.shape[0]), dtype=np.int64)
    label_counts = labels.size
    current_num_centers = 0
    if labels.size <= median_cluster_size:
        return labels
    if median_cluster_size <= 2:
        labels = np.arange(0, labels.size, dtype=np.int64)
        return labels
    centers = np.mean(data, axis=0)
    distances = np.sum((data - centers)**2, axis=1)
    if np.all(np.all(distances == 0)):
        # All scores are the same, so return all same label
        return labels

    if n_random > 0:
        if n_random >= labels.size:
            return np.arange(0, labels.size, dtype=np.int64)
        n_random = np.ceil(n_random).astype(np.int64)
        for nl in range(0, n_random):
            rand_ind = np.random.choice(data.shape[0], 1,
                            p=(distances/np.sum(distances)), replace=False)
            current_num_centers += 1
            new_center = data[rand_ind, :]
            centers = np.vstack((centers, new_center))
            temp_distance = np.sum((data - new_center)**2, axis=1)
            select = temp_distance < distances
            labels[select] = current_num_centers
            distances[select] = temp_distance[select]
            _, label_counts = np.unique(labels, return_counts=True)
            if current_num_centers == labels.size:
                break
    pre_centers = current_num_centers

    _, label_counts = np.unique(labels, return_counts=True)
    # Convert percentile to an index
    n_percentile = np.ceil((labels.size-1) * (1 - choose_percentile)).astype(np.int64)
    while np.median(label_counts) > median_cluster_size and current_num_centers < labels.size:
        current_num_centers += 1
        # Partition the negative distances (ascending partition)
        new_index = np.argpartition(-distances, n_percentile)[n_percentile]
        # Choose data at percentile index as the center of the next cluster
        new_center = data[new_index, :]
        centers = np.vstack((centers, new_center))
        temp_distance = np.sum((data - new_center)**2, axis=1)
        # Add any points closer to new center than their previous center
        select = temp_distance < distances
        labels[select] = current_num_centers
        distances[select] = temp_distance[select]
        _, label_counts = np.unique(labels, return_counts=True)

    return labels


def reorder_labels(labels):
    """
    Rename labels from 0 to n-1, where n is the number of unique labels.

    Returns None. Input labels is altered in place.

    Following sorting, the labels for individual spikes can be any number up to
    the maximum number of clusters used for initial sorting (worst case
    scenario, this could be 0:M-1, where M is the number of spikes). This
    function reorders the labels so that they nicely go from 0:num_clusters.

    Parameters
    ----------
    labels : numpy ndarray
        A one dimensional array of numerical labels.

    Returns
    -------
    None :
        The array labels is changed in place.
    """

    if labels.size == 0:
        return
    unique_labels = np.unique(labels)
    new_label = 0
    for old_label in unique_labels:
        labels[labels == old_label] = new_label
        new_label += 1

    return None


def kde_builtin(data, n):
    """
    ----------------------------------------------------------------------------
    THIS IS THE ORIGINAL PYTHON IMPLEMENTATION THAT IS SUPER SLOW AND USED ONLY
    FOR TESTING AGAINST THE COMPILED VERSION IN sort_cython.pyx
    ----------------------------------------------------------------------------
    Kernel density estimate (KDE) with automatic bandwidth selection.

    Returns an array of the KDE and the bandwidth used to compute it.

    This code was adapted to Python from the original MatLab script distributed
    by Zdravko Botev (see Notes below).
    "Reliable and extremely fast kernel density estimator for one-dimensional
    data. Gaussian kernel is assumed and the bandwidth is chosen automatically.
    Unlike many other implementations, this one is immune to problems caused by
    multimodal densities with widely separated modes (see example). The
    estimation does not deteriorate for multimodal densities, because we never
    assume a parametric model for the data."

    Parameters
    ----------
    data : numpy ndarray
        a vector of data from which the density estimate is constructed
    n : float
        the number of mesh points used in the uniform discretization of the
        input data over the interval [MIN, MAX] (min and max is determined from
        the range of values input in data). n has to be a power of two. if n is
        not a power of two, then n is rounded up to the next power of two,
        i.e., n is set to n = 2 ** np.ceil(np.log2(n)).

    Returns
    -------
    density : numpy ndarray
        Column vector of length 'n' with the values of the density estimate at
        the xmesh grid points.
    xmesh : numpy ndarray
        The grid over which the density estimate was computed.
    bandwidth : float
        The optimal bandwidth (Gaussian kernel assumed).

    Example
    -------
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.hstack((np.random.randn(100), np.random.randn(100)*2+35,
                    np.random.randn(100)+55))
    density, xmesh, bandwidth = sort.kde(data, 2**14)
    counts, xvals = np.histogram(data, bins=100)
    plt.bar(xvals[0:-1], counts, width=1, align='edge', color=[.5, .5, .5])
    plt.plot(xmesh, density * (np.amax(counts) / np.amax(density)), color='r')
    plt.xlim(-5, 65)
    plt.show()

    Notes
    -----
    New comments for this Python translation are in triple quotes below.  The
    original comments in the distributed MatLab implementation are indicated
    with hashtag style.

    MatLab code downloaded from:
    https://www.mathworks.com/matlabcentral/fileexchange/
                                14034-kernel-density-estimator
    with references and author information here:
    https://web.maths.unsw.edu.au/~zdravkobotev/
    and here:
    Kernel density estimation via diffusion
    Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
    Annals of Statistics, Volume 38, Number 5, pages 2916-2957.

    I removed the original 'MIN' and 'MAX' inputs and instead always set them to
    the default values.  Originally these inputs were optional, and defined as:
    MIN, MAX  - defines the interval [MIN,MAX] on which the density estimate is
                constructed the default values of MIN and MAX are:
                MIN=min(data)-Range/10 and MAX=max(data)+Range/10, where
                Range=max(data)-min(data)

    I removed the original 'cdf' output and only output the 'density' pdf.
    Original cdf output definition was:
        cdf  - column vector of length 'n' with the values of the cdf

    I removed all plotting functionality of the original function.
    """

    def dct1d(data):
        """ I changed this to use the scipy discrete cosine transform instead.
        The only difference is the scipy needs to be set to 1 in the first
        element.  I kept the original function below for reference.
        """
        a = dct(data)
        """ The original implementation below returns 1 for first element, so
        # enforce that here """
        a[0] = 1.
        return a
        """
        --------------------------------------------------------------------
                ORIGINAL FUNCTION BELOW
        --------------------------------------------------------------------
        """
        # computes the discrete cosine transform of the column vector data
        # Compute weights to multiply DFT coefficients
        data_copy = np.copy(data)
        weight = np.hstack((1, 2*(np.exp(-1 * 1j * np.arange(1, data_copy.size) * np.pi / (2 * data_copy.size)))))
        # Re-order the elements of the columns of x
        data_copy = np.hstack((data_copy[0::2], data_copy[-1:0:-2]))
        #Multiply FFT by weights:
        return np.real(weight * fft(data_copy))

    def idct1d(data):
        # computes the inverse discrete cosine transform
        # Compute weights
        weights = data.size * np.exp(1j * np.arange(0, data.size) * np.pi / (2 * data.size))
        # Compute x tilde using equation (5.93) in Jain
        data = np.real(ifft(weights * data))
        # Re-order elements of each column according to equations (5.93) and
        # (5.94) in Jain
        out = np.zeros(data.size)
        out_midslice = int(data.size / 2)
        out[0::2] = data[0:out_midslice]
        out[1::2] = data[-1:out_midslice-1:-1]
        #   Reference:
        #      A. K. Jain, "Fundamentals of Digital Image
        #      Processing", pp. 150-153.

        return out

    def fixed_point(t, N, I, a2):
        # this implements the function t-zeta*gamma^[l](t)
        l = 7
        # f_fac = np.sum(I ** l * a2 * np.exp(-I * np.pi ** 2 * t))
        # This line removes I ** l and keeps things in range of float64
        f_fac = np.sum(np.exp(np.log(I) * l + np.log(a2) - I * np.pi ** 2 * t))
        if f_fac < 1e-6 or N == 0:
            # Prevent zero division, which converges to negative infinity
            return -np.inf
        f = 2 * np.pi ** (2*l) * f_fac
        for s in range(l - 1, 1, -1): # s=l-1:-1:2
            K0 = np.prod(np.arange(1, 2*s, 2)) / np.sqrt(2*np.pi)
            const = (1 + (1/2) ** (s + 1/2)) / 3
            time = (2 * const * K0 / N / f) ** (2 / (3 + 2*s))
            # f_fac = np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
            # This line removes I ** s and keeps things in range of float64
            f_fac = np.sum(np.exp(np.log(I) * s + np.log(a2) - I * np.pi ** 2 * time))
            if f_fac < 1e-6:
                # Prevent zero division, which converges to negative infinity
                f = -1.0
                break
            f = 2 * np.pi ** (2*s) * f_fac

        if f > 0.0:
            return t - (2 * N * np.sqrt(np.pi) * f) ** (-2/5)
        else:
            return -np.inf

    def fixed_point_abs(t, N, I, a2):
        """ I added this for the case where no root is found and we seek the
        minimum absolute value in the main optimization while loop below.  It
        is identical to 'fixed_point' above but returns the absolute value.
        """
        f_t = fixed_point(t, N, I, a2)
        # Get absolute value
        if f_t >= 0.0:
            return f_t
        else:
            return -1.0 * f_t

    def bound_grad_desc_fixed_point_abs(N, I, a2, lower, upper, xtol, ytol):
        """ I added this for the case where no root is found and we seek the
        minimum absolute value in the main optimization while loop below.  It
        is identical to 'fixed_point' above but returns the absolute value.
        """
        alpha = 1e-4 # Choose learning rate
        max_iter = 1000
        t_star = lower
        f_min = np.inf
        t = lower

        # Choose starting point as lowest over 10 intervals
        dt = (upper - lower) / 10
        if dt < xtol:
            dt = xtol
        while t <= upper:
            f_t = fixed_point_abs(t, N, I, a2)
            if f_t < f_min:
                t_star = t
                f_min = f_t
            t += dt
        if np.isinf(f_min):
            return 0.0
        # reset t and f_t to lowest point to start search
        t = t_star
        f_t = f_min
        n_iters = 0
        while True:
            f_dt_pl = fixed_point_abs(t + dt, N, I, a2)
            f_dt_mn = fixed_point_abs(t - dt, N, I, a2)
            d_f_t_dt = (f_dt_pl - f_dt_mn) / (2*dt)
            if np.isinf(d_f_t_dt):
                t_star = t
                break

            # Update t according to gradient d_f_t_dt
            next_t = t - alpha * d_f_t_dt
            # If next_t is beyond bounds, choose point halfway
            if next_t >= upper:
                next_t = (upper - t)/2 + t
            if next_t <= lower:
                next_t = (t - lower)/2 + lower
            f_t = fixed_point_abs(next_t, N, I, a2)

            # Get absolute value of change in f_t and t
            f_step = f_t - f_min
            if f_step < 0.0:
                f_step *= -1.0
            t_step = t - next_t
            if t_step < 0.0:
                t_step *= -1.0

            if (f_step < ytol) or (t_step < xtol):
                # So little change we declare ourselves done
                t_star = t
                break
            t = next_t
            dt = t_step
            f_min = f_t
            n_iters += 1
            if n_iters > max_iter:
                t_star = t
                break

        # if do_print: print("SOLUTION CONVERGED IN ", n_iters, "ITERS to", t_star, upper)
        return t_star

    n = 2 ** np.ceil(np.log2(n)) # round up n to the next power of 2
    # define the interval [MIN, MAX]
    MIN = np.amin(data)
    MAX = np.amax(data)
    # Range = maximum - minimum
    # MIN = minimum# - Range / 20 # was divided by 2
    # MAX = maximum# + Range / 20

    density = np.array([1])
    xmesh = np.array([MAX])
    bandwidth = 0
    if MIN == MAX:
        return density, xmesh, bandwidth

    # set up the grid over which the density estimate is computed
    R = MAX - MIN
    dx = R / (n - 1)
    xmesh = MIN + np.arange(0, R+dx, dx)
    N = np.unique(data).size
    # bin the data uniformly using the grid defined above
    """ ADD np.inf as the final bin edge to get MatLab histc like behavior """
    initial_data = np.histogram(data, bins=np.hstack((xmesh, np.inf)))[0] / N
    initial_data = initial_data / np.sum(initial_data)
    a = dct1d(initial_data) # discrete cosine transform of initial data

    # now compute the optimal bandwidth^2 using the referenced method
    I = np.arange(1, n, dtype=np.float64) ** 2 # Do I as float64 so it doesn't overflow in fixed_point
    a2 = (a[1:] / 2) ** 2
    N_tol = 50 * (N <= 50) + 1050 * (N >= 1050) + N * (np.logical_and(N < 1050, N>50))
    tol = 10.0 ** -12.0 + 0.01 * (N_tol - 50.0) / 1000.0
    fixed_point_0 = fixed_point(0, N, I, a2)
    fmin_val = np.inf
    f_0 = fixed_point_0
    tol_0 = 0
    """ This is the main optimization loop to solve for t_star """
    while True:
        f_tol = fixed_point(tol, N, I, a2)
        """ Attempt to find a zero crossing in the fixed_point function moving
        stepwise from fixed_point_0 """
        if np.sign(f_0) != np.sign(f_tol):
            # use fzero to solve the equation t=zeta*gamma^[5](t)
            """ I am using fsolve here rather than MatLab 'fzero' """
            t_star = bound_grad_desc_fixed_point_abs(N, I, a2, tol_0, tol, 1e-6, 1e-6)
            break
        else:
            tol_0 = tol
            tol = min(tol * 2, 1.0) # double search interval
            f_0 = f_tol
        if tol == 1.0: # if all else fails
            """ Failed to find zero crossing so find absolute minimum value """
            t_star = bound_grad_desc_fixed_point_abs(N, I, a2, 0, 1.0, 1e-6, 1e-6)
            break
    # smooth the discrete cosine transform of initial data using t_star
    a_t = a * np.exp(-1 * np.arange(0, n) ** 2 * np.pi ** 2 * t_star / 2)
    # now apply the inverse discrete cosine transform
    density = idct1d(a_t) / R
    # take the rescaling of the data into account
    bandwidth = np.sqrt(t_star) * R
    """ I set this to zero instead of a small epsilon value. """
    density[density < 0] = 0; # remove negatives due to round-off error

    return density, xmesh, bandwidth
