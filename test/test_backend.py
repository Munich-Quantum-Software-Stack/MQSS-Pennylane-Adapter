import pennylane as qml

import pytest
from mqss.pennylane_adapter.config import MQSS_TOKEN, MQSS_BACKENDS
from mqss.pennylane_adapter.device import MQSSPennylaneDevice
from pennylane import numpy as np

dev = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)
dev_simulator = qml.device("default.qubit", wires=2)
dev_hamiltonian = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)
dev_hamiltonian_simulator = qml.device("default.qubit", wires=2)
dev_autograd = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)


def arbitrary_quantum_circuit(x: float, y: float) -> None:
    """
    Defines an arbitrary mock quantum circuit for testing purposes, without a measurement operation

    :param x: The parameter `x` in the `quantum_function_expval` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `quantum_function_expval` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    """
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RX(x, wires=1)


@qml.qnode(dev)
def quantum_function_expval(x: float, y: float) -> float:
    """
    The function `quantum_function_expval` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit.

    :param x: The parameter `x` in the `quantum_function_expval` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `quantum_function_expval` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `quantum_function_expval` returns the expected value of a given operator
    """
    arbitrary_quantum_circuit(x, y)
    return qml.expval(qml.PauliX(0) @ qml.PauliY(1))


@qml.qnode(dev_autograd, interface="autograd", diff_method="parameter-shift")
def quantum_function_autograd(x: float, y: float) -> float:
    """
    The function `quantum_function_expval` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit.

    :param x: The parameter `x` in the `quantum_function_expval` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `quantum_function_expval` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `quantum_function_expval` returns the expected value of a given operator
    """
    arbitrary_quantum_circuit(x, y)
    return qml.expval(qml.PauliX(0) @ qml.PauliY(1))


@qml.qnode(dev_simulator)
def quantum_function_expval_simulator(x: float, y: float) -> float:
    """
    The function `quantum_function_expval` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit. Implemented to be done on Pennylane simulator

    :param x: The parameter `x` in the `quantum_function_expval` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `quantum_function_expval` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `quantum_function_expval` returns the expected value of a given operator
    """
    arbitrary_quantum_circuit(x, y)
    return qml.expval(qml.PauliX(0) @ qml.PauliY(1))


@qml.qnode(dev_hamiltonian)
def quantum_function_hamiltonian_expval(
    x: float, y: float, H: qml.Hamiltonian
) -> float:
    """
    The function `quantum_function_expval` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit.

    :param x: The parameter `x` in the `quantum_function_expval` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `quantum_function_expval` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `quantum_function_expval` returns the expected value of a given operator
    :H: Pennylane Hamiltonian object
    """
    arbitrary_quantum_circuit(x, y)

    return qml.expval(H)


@qml.qnode(dev_hamiltonian_simulator)
def quantum_function_hamiltonian_expval_simulator(
    x: float, y: float, H: qml.Hamiltonian
) -> float:
    """
    The function `quantum_function_expval` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit.

    :param x: The parameter `x` in the `quantum_function_expval` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `quantum_function_expval` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `quantum_function_expval` returns the expected value of a given operator
    :H: Pennylane Hamiltonian object
    """
    arbitrary_quantum_circuit(x, y)
    return qml.expval(H)


@pytest.mark.parametrize(
    "params", [[np.pi / 3, np.pi / 17], [np.pi * 13 / 12, np.pi / 8]]
)
def test_compare_runs(params: list[float], method: str = "hellinger") -> bool:
    """Compare the runs done on LRZ backend with ideal simulations.

    Args:
        params (list[float]): List of parameters to the quantum circuit
        method (str):
            'hellinger': Hellinger distance
            'fidelity': Exact fidelity calculation, requires state tomography from QC
    """
    result_simulator = quantum_function_expval_simulator(*params)
    result = quantum_function_expval(*params)
    assert abs(result - result_simulator) <= 1e-1


@pytest.mark.parametrize("params", [[np.pi / 5, np.pi]])
def test_compare_generated_circuits(params: list[float]) -> bool:
    """Compare the runs done on LRZ backend with ideal simulations.

    Args:

        params (list[float]): List of parameters to the quantum circuit

    """
    _ = quantum_function_expval_simulator(*params)
    _ = quantum_function_expval(*params)

    assert (
        quantum_function_expval.qtape.operations
        == quantum_function_expval_simulator.qtape.operations
    )


@pytest.mark.parametrize("params", [[np.pi / 5, np.pi]])
def _test_autograd(params: list[float]) -> bool:
    """Compare the runs done on LRZ backend with ideal simulations in d

    Args:

        params (list[float]): List of parameters to the quantum circuit

    """

    results = qml.gradients.param_shift(quantum_function_autograd)(*params)
    print(results)
    assert (
        quantum_function_expval.qtape.operations
        == quantum_function_expval_simulator.qtape.operations
    )


@pytest.mark.parametrize("coeffs", [[1.5, -0.92]])
@pytest.mark.parametrize(
    "obs",
    [
        [
            qml.PauliX(0) @ qml.PauliY(1),
            qml.PauliY(0) @ qml.PauliZ(1),
        ]
    ],
)
@pytest.mark.parametrize("params", [[np.pi / 5, np.pi]])
def test_hamiltonian_measurements(
    params: list[float],
    coeffs: list[float],
    obs: list[qml.ops.qubit.non_parametric_ops],
):
    """Run a quantum circuit with a hamiltonian expectation value

    Args:
        coeffs (list[float]): _description_
        obs (list[qml.ops.qubit.non_parametric_ops]): _description_
    """

    hamiltonian = qml.Hamiltonian(coeffs, obs)

    try:
        result = quantum_function_hamiltonian_expval(*params, hamiltonian)
        result_simulator = quantum_function_hamiltonian_expval_simulator(
            *params, hamiltonian
        )

    except Exception as e:
        print(
            f"There was an error while measuring the expectation value of the hamiltonian, with the following error: {e}"
        )
    assert abs(result - result_simulator) <= 1e-1
