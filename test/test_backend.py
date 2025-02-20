import pennylane as qml
import pytest
from .config import *
from src.mqp.pennylane_provider.device import LRZDevice
from pennylane import numpy as np

dev = LRZDevice(wires=3)
dev_simulator = qml.device("default.qubit", wires=3)

def circuit(x, y, PauliWord):
    """
    The function `my_quantum_function` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of the Pauli Word and converts measurement basis to Z specified 

    :param x: angle for the rotation gate RZ on the qubit at wire 0
    :param y: angle for the rotation gate RY on the qubit at wire 1
    :param Pauli_Word: Set of Pauli operators to be measured for each qubit. If Identity is applied then it's not required to mention in the argument.
    :ConvertToPauliZ: Converts to PauliZ basis

    :return: Returns the expectation value of the Pauli Word given
    """
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RX(x, wires=1)
    qml.RX(y, wires=2)
    qml.CNOT(wires=[1, 2])
    return qml.expval(PauliWord)

@pytest.mark.parametrize(
    "params", [ [np.pi * 13 / 12, np.pi / 8]]  
)

def test_compare_expectation_values_Hamiltonians(params,method="hellinger"):

    Devices = { dev_simulator:['Simulator: ',] , dev : ['LRZ Device: '] }

    #Test simple Pauliword
    Pauliword = qml.PauliY(1)@qml.PauliX(2)

    # Test Sprod 
    H1 = 5*qml.PauliZ(1)

    #Test Hamiltonian as a simple sum
    H2 = 0.5 * (qml.PauliX(1) @ qml.PauliZ(2)) + 5 * (qml.PauliZ(0) @ qml.PauliY(1)) + -4 * (qml.PauliY(2))

    #Test pennylane Hamiltonian
    coeffs = [0.2, -0.543]
    obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    H3 = qml.ops.LinearCombination(coeffs, obs)

    Hamiltonians = [Pauliword,H1,H2,H3]

    for Hamiltonian in Hamiltonians:
        for device in Devices:
            qnode = qml.QNode(circuit, device)
            result = qnode (*params, Hamiltonian )
            Devices[device].append(result)

    success = True
    tolerance = 15e-1
    
    for i in range(1, len(Devices[dev][1:])+1):
        diff = abs(Devices[dev][i] - Devices[dev_simulator][i])
        print(f"{i}) {Hamiltonians[i-1]} \n LRZ device: {Devices[dev][i]} \n Simulator: {Devices[dev_simulator][i]}\n Difference: {diff}\n")
        if diff > tolerance:
            print(f"Comparison failed at index {i}: Difference {diff} exceeds tolerance.")
            success = False
    
    # Assert that all comparisons were successful
    assert success, "Comparison failed for some results!"
