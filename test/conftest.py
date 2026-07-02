import pytest
import pennylane as qml
from pennylane import numpy as np


@pytest.fixture(params=[np.array([0.1, 0.2], requires_grad=True)])
def grad_params(request):
    return request.param


@pytest.fixture(
    params=[
        [np.pi / 5, np.pi],
        [np.pi / 3, np.pi / 17],
        [np.pi * 13 / 12, np.pi / 8],
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
        (
            [1.5],
            [qml.PauliZ(0)],
        ),
        (
            [1.2, 0.3],
            [qml.PauliX(0) @ qml.PauliX(1), qml.PauliX(0)],
        ),
        (
            [-0.5, -0.2, -0.2],
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
