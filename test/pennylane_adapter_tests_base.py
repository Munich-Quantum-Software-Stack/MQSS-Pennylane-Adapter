import pytest


import pennylane as qml
from .config import CURRENT_RESOURCES


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
