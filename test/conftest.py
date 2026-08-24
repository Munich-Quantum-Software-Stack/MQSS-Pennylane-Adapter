import pennylane as qml
import pytest
from pennylane import numpy as np


@pytest.fixture(
    params=[
        [2 * np.pi / 7, np.pi/8],
        [0,0],
    ]
)
def params(request):
    return request.param


@pytest.fixture(
    params=[
        qml.PauliZ(0) @ qml.PauliX(1),
        qml.PauliZ(0),
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
        # (
        #     [1.5],
        #     [qml.PauliZ(0)],
        # ),
        # (
        #     [1.2, 0.3],
        #     [qml.PauliX(0) @ qml.PauliX(1), qml.PauliX(0)],
        # ),
        (
            [0.8, -0.2, -0.2],
            [
                qml.PauliZ(0) @ qml.PauliZ(1),
                qml.PauliX(0),
                qml.PauliX(1),
            ],
        ),
    ]
)
def hamiltonian_data(request):
    return request.param


@pytest.fixture(
    params=[
        [qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(2), qml.PauliZ(3)],
        [qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(2) @ qml.PauliZ(3)],
        [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliZ(2), qml.PauliZ(3)],
        [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliZ(2) @ qml.PauliZ(3)],
        [qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliZ(2) @ qml.PauliZ(3)],
    ]
)
def list_obs(request):
    return request.param
