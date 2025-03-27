# Getting Started

This repository implements a custom PennyLane backend called MQSSPennylaneDevice, which is able to send quantum jobs to LRZ's infrastructure using the [PennyLane](https://pennylane.ai) frontend. 
The users would be able to use various PennyLane functions ([executing circuits](https://docs.pennylane.ai/en/stable/introduction/operations.html), [different measurement types](https://docs.pennylane.ai/en/stable/introduction/measurements.html). For future releases: circuit optimization and Quantum Machine Learning) while running their jobs on [LRZ's Quantum Hardware](https://portal.quantum.lrz.de).

## 🛠️ Installation
To install the package, simply run 
```bash
pip install mqss-pennylane-adapter
```

## 🚀 Usage
MQSS PennyLane Provider has support for most of the native PennyLane features. For instance, you can define a quantum circuit using PennyLane quantum gates, and decorate the method with the MQSSPennylaneDevice object. Parametric gates can also be used. 
```python
import pennylane as qml
from pennylane import numpy as np
from mqss.pennylane_adapter.device import MQSSPennylaneDevice

dev = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)

@qml.qnode(dev)
def quantum_function_expval(x, y):
    """
    The function `quantum_function_expval` applies quantum operations RZ, CNOT, and RY to qubits and returns
    the expectation value of PauliZ on the second qubit.

    :param x: The parameter `x` in the `quantum_function_expval` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` in the `quantum_function_expval` function is used as the angle parameter for
    the rotation gate `RY(y, wires=1)`. This gate applies a rotation around the y-axis of the Bloch
    sphere by an angle `y` to the qubit on wire
    :return: The function `quantum_function_expval` returns the expected value of the given operator
    """
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RX(x, wires=1)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))
params = [np.pi / 3, np.pi / 17]
result = quantum_function_expval(*params)
```
Furthermore, you can define a Hamiltonian object within PennyLane, and calculate the expectation value with respect to that Hamiltonian. For these cases, Pennylane Provider simply creates a batch job for each term in the Hamiltonian, to calculate the expectation value.

```python
import pennylane as qml
from pennylane import numpy as np
from mqss.pennylane_adapter.device import MQSSPennylaneDevice
dev_hamiltonian = MQSSPennylaneDevice(wires=2, token='<MQSS_TOKEN>', backends='<MQSS_BACKENDS>')

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

J = 0.5  # Interaction strength
h = 0.2  # Transverse field strength
coeffs = [-J, -h, -h]  # TFIM with 2 sites
obs = [
    qml.PauliZ(0) @ qml.PauliZ(1),  # Ising interaction between sites 0 and 1
    qml.PauliX(0),
    qml.PauliX(1),
]

hamiltonian = qml.Hamiltonian(coeffs, obs)
result = quantum_function_hamiltonian_expval(*params, hamiltonian)
```

If you are just interested in accessing the counts, you can also use

```python
import pennylane as qml
from pennylane import numpy as np
from mqss.pennylane_adapter.device import MQSSPennylaneDevice
n_wires = 5
dev_hamiltonian = MQSSPennylaneDevice(wires=n_wires, token='<MQSS_TOKEN>', backends='<MQSS_BACKENDS>')
@qml.qnode(dev)
def circuit(
    x: float, y: float
) -> float:
    
    arbitrary_quantum_circuit(x, y)

    return qml.probs(range=(0, n_wires))

```

## 🛠️ Upcoming Features
 - Autograd support with parameter-shift
 - Grouping of commuting terms in the Hamiltonians to reduce the number of circuits in the batch

## 🤝 Contributing
Feel free to open [issues](https://github.com/Munich-Quantum-Software-Stack/MQSS-Pennylane-Adapter/issues) or submit [pull requests](https://github.com/Munich-Quantum-Software-Stack/MQSS-Pennylane-Adapter/pulls) to improve this project!
