<a href="https://gitmoji.dev">
  <img
    src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square"
    alt="Gitmoji"
  />
</a>

# pennylane-provider

This projects implements a custom Pennylane backend called MQSSPennylaneDevice, which is able to send quantum jobs to LRZ's infrastructure using the pennylane frontend. The users would be able to use all full-fletched pennylane functions (optimization, QML etc.) while running their jobs on LRZ's quantum hardware.


## Code Snippet

```python
import pennylane as qml
from pennylane import numpy as np
from src.mqp.pennylane_provider.device import MQSSPennylaneDevice

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
    :return: The function `quantum_function_expval` returns the expected value of the Pauli X operator acting on the first qubit (qubit 0) and Z operator
    on the second qubit (qubit 1)
    gate between qubits 0 and 1, and RY(y) on qubit 1.
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