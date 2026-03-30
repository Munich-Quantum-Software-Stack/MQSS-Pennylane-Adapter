import pytest
import pennylane as qml
from pennylane import numpy as np


@pytest.fixture(
    params=[
        [np.pi / 5, np.pi],
        [np.pi / 3, np.pi / 17],
        [np.pi * 13 / 12, np.pi / 8],
    ]
)
def params(request):
    return request.param


@pytest.fixture
def coeffs():
    """Shared Hamiltonian coefficients for testing."""
    return [1.5, -0.92]


@pytest.fixture(
    params=[
        [
            qml.PauliX(0) @ qml.PauliY(1),
            qml.PauliY(0) @ qml.PauliZ(1),
        ],
        [qml.PauliZ(0) @ qml.PauliX(1)],
        [qml.PauliZ(0)],
    ]
)
def obs(request):
    """Shared observables for Hamiltonian testing."""
    return request.param


@pytest.fixture(
    params=[
        (
            [1.5, -1],
            [
                qml.PauliZ(0) @ qml.PauliZ(1),
                qml.PauliY(0) @ qml.PauliZ(1),
            ],
        ),
        (
            [1.5],
            [qml.PauliZ(0) @ qml.PauliX(1)],
        ),
        (
            [1.5],
            [qml.PauliZ(0)],
        ),
    ]
)
def hamiltonian_data(request):
    return request.param
