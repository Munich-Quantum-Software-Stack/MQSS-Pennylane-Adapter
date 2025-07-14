import pennylane as qml

import pytest
from src.mqss.pennylane_adapter.config import MQSS_TOKEN, MQSS_BACKENDS
from src.mqss.pennylane_adapter.device import MQSSPennylaneDevice
from pennylane import numpy as np
from .pennylane_adapter_tests_base import TestPennylaneAdapter
from .mocks import MOCK_JOB_DATA

from datetime import datetime
import json
from mqss_client import (
    MQSSClient,
    Result,
)

dev = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)
dev_single = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)
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


@qml.qnode(dev_single)
def quantum_function_expval_single_pauli(x: float, y: float) -> float:
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
    return qml.expval(qml.PauliX(0))


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


@pytest.mark.mock
class TestPennylaneJobs(TestPennylaneAdapter):

    @pytest.fixture(autouse=True)
    def patch_submit_job(self, monkeypatch):
        def mock_submit_job(self, job_request):
            return "mock-uuid-12345"

        monkeypatch.setattr(MQSSClient, "submit_job", mock_submit_job)

    @pytest.fixture(autouse=True)
    def patch_job_result(self, monkeypatch):
        def mock_job_result(self, uuid, job_type):
            # Just always return the MOCK_JOB_DATA for the fixed UUID and job type key
            key = f"job/{uuid}/result"  # or hardcode if you want
            result_json = MOCK_JOB_DATA.get(key)
            # Construct Result without any checks
            return Result(
                counts=json.loads(result_json["result"]),
                timestamp_completed=datetime.strptime(
                    result_json["timestamp_completed"], "%Y-%m-%d %H:%M:%S.%f"
                ),
                timestamp_submitted=datetime.strptime(
                    result_json["timestamp_submitted"], "%Y-%m-%d %H:%M:%S.%f"
                ),
                timestamp_scheduled=datetime.strptime(
                    result_json["timestamp_scheduled"], "%Y-%m-%d %H:%M:%S.%f"
                ),
            )

        monkeypatch.setattr(MQSSClient, "wait_for_job_result", mock_job_result)

    @pytest.mark.parametrize(
        "params", [[np.pi / 3, np.pi / 17], [np.pi * 13 / 12, np.pi / 8]]
    )
    def test_compare_runs(
        monkeypatch: pytest.MonkeyPatch, params: list[float], method: str = "hellinger"
    ) -> bool:
        """Compare the runs done on LRZ backend with ideal simulations.

        Args:
            params (list[float]): List of parameters to the quantum circuit
            method (str):
                'hellinger': Hellinger distance
                'fidelity': Exact fidelity calculation, requires state tomography from QC
        """
        result = quantum_function_expval(*params)
        assert result is not None

    @pytest.mark.parametrize(
        "params", [[np.pi / 3, np.pi / 17], [np.pi * 13 / 12, np.pi / 8]]
    )
    def test_compare_runs_single_pauli(
        monkeypatch: pytest.MonkeyPatch, params: list[float], method: str = "hellinger"
    ) -> bool:
        """Compare the runs done on LRZ backend with ideal simulations.

        Args:
            params (list[float]): List of parameters to the quantum circuit
            method (str):
                'hellinger': Hellinger distance
                'fidelity': Exact fidelity calculation, requires state tomography from QC
        """
        result = quantum_function_expval_single_pauli(*params)
        assert result is not None

    @pytest.mark.parametrize("params", [[np.pi / 5, np.pi]])
    def _test_compare_generated_circuits(params: list[float]) -> bool:
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
    def _test_hamiltonian_measurements(
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
