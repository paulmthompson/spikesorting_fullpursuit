import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal

# setting path
import sys

sys.path.append("..")

from generate_voltage_traces import generate_voltage_traces
from spikesorting_fullpursuit.processing import zca
from spikesorting_fullpursuit.threshold.threshold import (
    single_thresholds,
)


def plot_power(
    voltages,
):
    T = 1 / 40000.0
    N = len(voltages[0])

    xf = np.linspace(0.0, 1.0 / (2.0 * T), len(voltages[0]) // 2)
    yf = scipy.fftpack.fft(voltages[0, :])

    plt.plot(xf, 2.0 / N * np.abs(yf[: N // 2]), color="black")
    plt.show()


def plot_cross_correlation(voltages):
    N = len(voltages[0])

    cross_cor = scipy.signal.correlate(voltages[0, :], voltages[1, :])
    lags = scipy.signal.correlation_lags(N, N)
    # cross_cor /= np.max(cross_cor)
    cross_cor /= N

    plt.plot(lags, cross_cor, color="black")
    plt.xlim([-100, 100])
    plt.show()


voltages, timestamps = generate_voltage_traces()

voltages = np.float32(voltages)

raw_voltages = np.frombuffer(voltages, dtype=np.float32).reshape((4, len(voltages[0])))

sigma = 4.0
zca_cushion = 100

for i in range(0, 4):
    plt.plot(raw_voltages[i, 0:3000] + i * 5, color="black")
plt.show()

plot_cross_correlation(raw_voltages)

plot_power(raw_voltages)

thresholds = single_thresholds(raw_voltages, sigma)

zca_matrix = zca.get_noise_sampled_zca_matrix(
    raw_voltages,
    thresholds,
    sigma,
    zca_cushion,
    n_samples=1e6,
)

zca_matrix = zca.get_full_zca_matrix(raw_voltages, rowvar=True)

# Set seg_voltage to ZCA transformed voltage
# @ makes new copy
seg_voltage = (zca_matrix @ raw_voltages).astype(np.float32)

for i in range(0, 4):
    plt.plot(seg_voltage[i, 0:3000] + i * 5, color="black")
plt.show()

plot_cross_correlation(seg_voltage)

plot_power(seg_voltage)
