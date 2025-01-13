import pennylane as qml

import pytest
from .config import *
from src.mqp.pennylane_provider.device import LRZDevice
from pennylane import numpy as np

dev = LRZDevice(wires=2)
dev_simulator = qml.device("default.qubit", wires=2)


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
    qml.CNOT(wires=[1, 0])
    qml.RX(x, wires=1)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))
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
    qml.CNOT(wires=[1, 0])
    qml.RX(x, wires=1)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))
    # return qml.expval(qml.PauliZ(1))
    # return qml.probs(range(0, 2))

''' TEMPORARILY COMMENTED
@pytest.mark.parametrize(
    "params", [[np.pi / 3, np.pi / 17], [np.pi * 13 / 12, np.pi / 8]]
)

def test_compare_runs(params, method="hellinger"):
    """Compare the runs done on LRZ backend with ideal simulations in d

    Args:
        counts (list[float]): Counts(z-axis) from the QC
        probs (list[float]): Probabily distribution from classical simulation
        method (str):
            'hellinger': Hellinger distance
            'fidelity': Exact fidelity calculation, requires state tomography from QC
    """
    result = my_quantum_function(*params)
    result_simulator = my_quantum_function_simulator(*params)
    assert abs(result - result_simulator) <= 1e-1 '''

@qml.qnode(dev, diff_method="parameter-shift")
def basic_circuit(x):
    qml.RX(x, wires=0)
    return qml.probs(wires=0)

@qml.qnode(dev_simulator, diff_method="parameter-shift")
def basic_circuit_simulator(x):
    qml.RX(x, wires=0)
    return qml.probs(wires=0)

@pytest.mark.parametrize(
    "params", [[np.pi / 3]]
)
#,[np.pi / 2] add another parameter if reqd
def test_gradient_calculations(params,method="hellinger"): 
    result = basic_circuit(*params)
    print('LRZ Backend :', result)

    result_simulator = basic_circuit_simulator(*params)
    print('Simulator : ', result_simulator)
    
    diff = abs(result.numpy() - result_simulator) #result converted to numpy array from tenser
    print(f'Difference: {diff}')

    assert np.all(diff <= 1e-1), f"Differences exceeded tolerance: {diff}"
    
    