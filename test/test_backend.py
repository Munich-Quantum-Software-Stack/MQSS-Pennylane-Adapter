import pennylane as qml

import pytest
from .config import *
from src.mqp.pennylane_provider.device import LRZDevice


# tests should start with test_
# def test_
@pytest.mark.parametrize("params", [[1, 2]])
def test_simulate(params):
    result = my_quantum_function(1, 2)
    print(result)

    assert type(result) == float


# @pytest.mark.parametrize("params", [[1, 2]])
# def test_dummy(params):

#     assert len(params) != 0

#     qc = my_quantum_function(1, 2)

#     job = backend.run(qc, shots=1024)

#     assert False


# TODO1: Dynamically get number of wires
# DONE TODO2: Dynamically set the custom device (LRZ-backend)

dev = LRZDevice()
dev_simulator = qml.device("default.qubit", wires=5)


@qml.qnode(dev)
def my_quantum_function(x, y):
    """
    The function `my_quantum_function` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit.

    :param x: The parameter `x` in the `my_quantum_function` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `my_quantum_function` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `my_quantum_function` returns the expected value of the Pauli Z operator
    acting on the second qubit (qubit 1) after applying the quantum operations RZ(x) on qubit 0, a CNOT
    gate between qubits 0 and 1, and RY(y) on qubit 1.
    """
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    return qml.expval(qml.PauliZ(1))
    # return qml.probs(range(0, 2))


@qml.qnode(dev_simulator)
def my_quantum_function_simulator(x, y):
    """
    The function `my_quantum_function` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit. Implemented to be done on Pennylane simulator

    :param x: The parameter `x` in the `my_quantum_function` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `my_quantum_function` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `my_quantum_function` returns the expected value of the Pauli Z operator
    acting on the second qubit (qubit 1) after applying the quantum operations RZ(x) on qubit 0, a CNOT
    gate between qubits 0 and 1, and RY(y) on qubit 1.
    """
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    # return qml.expval(qml.PauliZ(1))
    return qml.probs(range(0, 2))


def convert_counts_json_to_array(counts):
    """Given counts object as a json, convert it to a list of floats

    Args:
        counts (list[float]): Counts(z-axis) from the QC
    """
    pass


def compare_runs(counts, probs, method="hellinger"):
    """Compare the runs done on LRZ backend with ideal simulations in d

    Args:
        counts (list[float]): Counts(z-axis) from the QC
        probs (list[float]): Probabily distribution from classical simulation
        method (str):
            'hellinger': Hellinger distance
            'fidelity': Exact fidelity calculation, requires state tomography from QC
    """
    pass
