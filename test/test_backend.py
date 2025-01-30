import pennylane as qml

import pytest
from .config import *
from src.mqp.pennylane_provider.device import LRZDevice
from pennylane import numpy as np

dev = LRZDevice(wires=2)
dev_simulator = qml.device("default.qubit", wires=2)

def apply_basis_change(PauliWord):
    """
    Applies Hadamard gate if PauliX is present and S† (adjoint S) + Hadamard if PauliY is present.

    :param PauliWord: A tensor product of Pauli operators (e.g., qml.PauliX(0) @ qml.PauliY(1))
    """
    # Modify each Pauli operator in the tensor product
    new_terms = []
    for operator in PauliWord:
        qubit = operator.wires[0]

        if isinstance(operator, qml.PauliX): 
            qml.Hadamard(qubit)  

        if isinstance(operator, qml.PauliY):
            qml.adjoint(qml.S)(qubit)  #  S† (adjoint S)
            qml.Hadamard(qubit)  

        new_terms.append(qml.PauliZ(qubit))

    # Return the updated tensor product of Pauli operators
    return qml.prod(*new_terms)


def circuit(x, y, PauliWord, Convert_to_Z_basis):
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

    if Convert_to_Z_basis:
        PauliWord=apply_basis_change(PauliWord)

    print(PauliWord) 
    return qml.expval(PauliWord)


@pytest.mark.parametrize(
    "params", [ [np.pi * 13 / 12, np.pi / 8]]
)
def test_compare_runs(params, method="hellinger"):
    import pennylane as qml

    """Compare the runs done on LRZ backend with ideal simulations in d

    Args:
        counts (list[float]): Counts(z-axis) from the QC
        probs (list[float]): Probabily distribution from classical simulation
        method (str):
            'hellinger': Hellinger distance
            'fidelity': Exact fidelity calculation, requires state tomography from QC
    """
    #Let our hamiltonian be H= PauliX(0) @ PauliZ(1) + PauliZ(0) @ PauliY(1) + PauliY(0)
    Hamiltonian = [qml.PauliX(0) @ qml.PauliZ(1) , qml.PauliZ(0) @ qml.PauliY(1) , qml.PauliY(1)]
    #Hamiltonian = [qml.PauliY(1)@qml.PauliZ(0)]
    #Devices dictionary stores the device to be tested, string to be printed during 
    Devices = { dev : ['LRZ Device: '] } #UNCOMMENT WHEN QEXA IS BACK
    #Devices = {dev_simulator:['Simulator: ',[]]}

    #This loop runs for each device defined in devices and the results are stored in the value of the dictionary Devices
    for device, result_from_device in Devices.items():
        
        #This loop first converts X and Y to the Z basis and in the second iteration it measures in the given basis
        for Convert_to_Z_basis in [True, False]:
            print("\nMeasuring in Z basis") if Convert_to_Z_basis else print("Measuring in the given basis")
            result = 0

            #This loop sums up the expectation value of each pauliword in the defined Hamiltonian
            for PauliWord in Hamiltonian:
                qnode = qml.QNode(circuit, device)
                result += qnode (*params, PauliWord, Convert_to_Z_basis)

                qml_to_qasm_circuit = qnode.qtape.to_openqasm()
                print(qml_to_qasm_circuit)

            print(result_from_device[0], result)
            result_from_device.append(result)

    '''
    #Uncomment when WHEN QEXA IS OPERATING 
    # Compare pairwise results
    for i in range(1,len(result[1:])): #ignore first term which is a string indicating device
        assert abs(result[i] - result_simulator[i]) <= 1e-1, f"Comparison failed at index {i}: {result[i]} vs {result_simulator[i]}"

    '''
'''
RESULTS: inspect qasm circuit
Measuring in Z basis
Z(1) @ Z(0)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(3.4033920413889427) q[0];
cx q[0],q[1];
ry(0.39269908169872414) q[1];
cx q[1],q[0];
rx(3.4033920413889427) q[1];
sdg q[1];
h q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];

Simulator:  0.25881904510252063
Measuring in the given basis
Y(1) @ Z(0)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(3.4033920413889427) q[0];
cx q[0],q[1];
ry(0.39269908169872414) q[1];
cx q[1],q[0];
rx(3.4033920413889427) q[1];
z q[1];
s q[1];
h q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];

Simulator:  0.25881904510252063
PASSED
'''