import numpy as np

# setting path
import sys
sys.path.append("..")

from generate_voltage_traces import generate_voltage_traces


def test_create_voltage():
    voltages, timestamps = generate_voltage_traces()

    assert len(voltages) == 4
    assert len(voltages[0]) == 30 * 40000


def test_create_voltage_matrix():
    voltages, timestamps = generate_voltage_traces()

    voltages = np.float32(voltages)

    raw_voltages = np.frombuffer(voltages, dtype=np.float32).reshape((4,len(voltages[0])))

    assert np.size(raw_voltages, 0) == 4
    assert np.size(raw_voltages, 1) == 30 * 40000


