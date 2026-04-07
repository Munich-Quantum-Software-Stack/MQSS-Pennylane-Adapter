import pennylane as qml

import pytest
from src.mqss.pennylane_adapter.config import MQSS_TOKEN, MQSS_BACKENDS
from src.mqss.pennylane_adapter.device import MQSSPennylaneDevice
from pennylane import numpy as np
from .pennylane_adapter_tests_base import TestPennylaneAdapter

dev = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)
dev_simulator = qml.device("default.qubit", wires=2)
dev_hamiltonian = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)
dev_hamiltonian_simulator = qml.device("default.qubit", wires=2)
dev_hamiltonian_simulator2 = qml.device("default.qubit", wires=2)
dev_autograd = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)
dev_probs = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)


def GHZ_circuit(num_wires: int) -> None:
    """Defines a GHZ state preparation circuit on the specified number of wires."""

    qml.Hadamard(wires=0)
    for i in range(num_wires - 1):
        qml.CNOT(wires=[0, i])


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


@qml.qnode(dev_probs)
def quantum_function_probs(x: float, y: float) -> np.ndarray:
    """
    The function `quantum_function_expval` applies an arbitrary quantum function to qubits and returns
    the probabilities of the computational basis states.
    """
    arbitrary_quantum_circuit(x, y)
    return qml.probs(wires=[0, 1])


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


@qml.qnode(dev_hamiltonian_simulator2)
def quantum_function_hamiltonian_expval_simulator2(
    x: float, y: float, H: qml.Hamiltonian
) -> float:
    arbitrary_quantum_circuit(x, y)
    return qml.probs(wires=range(2))


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


@pytest.mark.live
class TestPennylaneLiveJobs(TestPennylaneAdapter):

    @pytest.mark.parametrize("params", [[np.pi / 5, np.pi]])
    def _test_compare_generated_circuits(self, params: list[float]) -> bool:
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
    def _test_autograd(self, params: list[float]) -> bool:
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

    def test_expectation_value_measurements(
        self, obs: list[qml.ops.qubit.non_parametric_ops], params: list[float]
    ):
        """Run a quantum circuit with an expectation value measurement and compare the results with the simulator."""

        result = quantum_function_hamiltonian_expval(*params, obs)
        result_simulator = quantum_function_hamiltonian_expval_simulator(*params, obs)

        assert result is not None
        assert abs(result - result_simulator) <= 3e-1

    def test_hamiltonian_measurements(
        self,
        hamiltonian_data: tuple[list[float], list[qml.ops.qubit.non_parametric_ops]],
        params: list[float],
    ):
        """Run a quantum circuit with a hamiltonian expectation value

        Args:
            coeffs (list[float]): _description_
            obs (list[qml.ops.qubit.non_parametric_ops]): _description_
        """
        coeffs, obs = hamiltonian_data
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        try:
            result = quantum_function_hamiltonian_expval(*params, hamiltonian)
            result_simulator = quantum_function_hamiltonian_expval_simulator(
                *params, hamiltonian
            )
            result_simulator2 = quantum_function_hamiltonian_expval_simulator2(
                *params, hamiltonian
            )
        except Exception as e:
            print(
                f"There was an error while measuring the expectation value of the hamiltonian, with the following error: {e}"
            )
            raise e

        assert result is not None
        print(result, "result")
        print(result_simulator, "result_simulator")
        print(result_simulator2, "result_simulator2")
        assert abs(result - result_simulator) <= 3e-1

    def test_probs(self, params: list[float]):
        """Test that we can get probabilities back from the device"""
        x, y = params
        result = quantum_function_probs(x, y)
        num_qubits = quantum_function_probs.qtape.num_wires
        assert result is not None
        assert len(result[0]) == (2**num_qubits)
        assert abs(sum(result[0]) - 1) <= 1e-6
        assert all(0 <= p <= 1 for p in result[0])
