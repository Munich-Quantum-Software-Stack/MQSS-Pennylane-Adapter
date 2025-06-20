import pytest

from mqss_client import (
    MQSSClient,
    Result,
)
import pennylane as qml
from .config import CURRENT_RESOURCES
from .mocks import MOCK_JOB_DATA

from datetime import datetime
import json


@pytest.fixture
def resource_name():
    """Fixture to provide a valid resource name for tests"""
    return list(CURRENT_RESOURCES.keys())[0]


class TestPennylaneAdapter:
    """Base class for Pennylane Adapter tests."""

    @pytest.fixture
    def arbitrary_quantum_circuit_builder(self):
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

        return arbitrary_quantum_circuit

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
