import numpy as np
from scipy import signal, linalg


def get_full_zca_matrix(data, rowvar=True):
    """
    Computes ZCA matrix for data. rowvar=False means that COLUMNS represent
    variables and ROWS represent observations.  Else the opposite is used.
    ZCA whitening is done with this matrix by calling:
        zca_filtered_data = np.dot(zca_matrix, data).

    The ZCA procedure was taken (and reformatted into 2 lines) from:
    https://github.com/zellyn/deeplearning-class-2011/blob/master/ufldl/pca_2d/pca_2d.py
    """
    if data.ndim == 1:
        return 1.0
    elif data.shape[0] == 1:
        return 1.0

    sigma = np.cov(data, rowvar=rowvar)
    U, S, _ = linalg.svd(sigma)
    zca_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-9)) @ U.T

    # Scale each of the rows of Z
    # This ensures that the output of the matrix multiplication (Z * channel voltages)
    # remains in the same range (doesn't clip ints)
    zca_matrix = zca_matrix / np.diag(zca_matrix)[:, None]

    return zca_matrix


def get_noise_sampled_zca_matrix(
    voltage_data,
    thresholds,
    sigma,
    thresh_cushion,
    n_samples=1e6,
):
    """
    The ZCA procedure was taken (and reformatted into 2 lines) from:
    https://github.com/zellyn/deeplearning-class-2011/blob/master/ufldl/pca_2d/pca_2d.py

    Args:
        voltage_data: (n_channels, n_samples) array of voltage data
        thresholds: (n_channels, ) array of thresholds for each channel
        sigma: (n_channels, ) array of sigma values for each channel
        thresh_cushion: number of samples to add to each side of threshold crossing
        n_samples: number of samples to use for computing ZCA matrix

    Returns:
        zca_matrix: (n_channels, n_channels) matrix for ZCA whitening
    """
    if voltage_data.ndim == 1:
        return 1.0
    zca_thresholds = np.copy(thresholds)
    # Use threshold for ZCA just under spike threshold
    zca_thresholds *= 0.90
    # convert cushion to zero centered window
    thresh_cushion = thresh_cushion * 2 + 1
    volt_thresh_bool = np.zeros(voltage_data.shape, dtype="bool")
    for chan_v in range(0, volt_thresh_bool.shape[0]):
        volt_thresh_bool[chan_v, :] = np.rint(
            signal.fftconvolve(
                np.abs(voltage_data[chan_v, :]) > zca_thresholds[chan_v],
                np.ones(thresh_cushion),
                mode="same",
            )
        ).astype("bool")
    sigma = np.empty((voltage_data.shape[0], voltage_data.shape[0]))
    for i in range(0, voltage_data.shape[0]):
        # Compute i variance for diagonal elements of sigma
        valid_samples = voltage_data[i, :][~volt_thresh_bool[i, :]]
        if valid_samples.size == 0:
            raise ValueError(
                "No data points under threshold of {0} to compute ZCA. Check threshold sigma, clip width, and artifact detection.".format(
                    thresholds[i]
                )
            )
        if n_samples > valid_samples.size:
            out_samples = valid_samples
        else:
            out_samples = np.random.choice(valid_samples, int(n_samples), replace=True)
        if out_samples.size < 1000:
            print(
                "WARNING: LESS THAN 1000 SAMPLES FOUND FOR ZCA CALCULATION ON CHANNEL",
                i,
            )
        sigma[i, i] = np.var(out_samples, ddof=1)
        ij_samples = np.full((2, voltage_data.shape[0]), np.nan)
        for j in range(i + 1, voltage_data.shape[0]):
            row_inds = np.array([i, j], dtype=np.int64)
            valid_samples = np.nonzero(~np.any(volt_thresh_bool[row_inds, :], axis=0))[
                0
            ]
            if n_samples > valid_samples.size:
                out_samples = valid_samples
            else:
                out_samples = np.random.choice(
                    valid_samples, int(n_samples), replace=True
                )

            sigma[i, j] = np.dot(
                voltage_data[i, out_samples], voltage_data[j, out_samples]
            ) / (out_samples.size - 1)
            sigma[j, i] = sigma[i, j]
            ij_samples[0, j] = out_samples.size
            ij_samples[1, j] = valid_samples.size

    U, S, _ = linalg.svd(sigma)
    zca_matrix = U @ np.diag(1.0 / np.sqrt(S + 1e-9)) @ U.T
    # Scale each of the rows of Z
    # This ensures that the output of the matrix multiplication (Z * channel voltages)
    # remains in the same range (doesn't clip ints)
    zca_matrix = zca_matrix / np.diag(zca_matrix)[:, None]
    # print("TURNED OFF UNDO ZCA SCALING LINE 81 PREPROCESSING")

    return zca_matrix
