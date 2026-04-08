# Getting Started

This repository implements a custom PennyLane backend called MQSSPennylaneDevice, which is able to send quantum jobs to LRZ's infrastructure using the [PennyLane](https://pennylane.ai) frontend. 
The users would be able to use various PennyLane functions, such as [executing circuits](https://docs.pennylane.ai/en/stable/introduction/operations.html) and [different measurement types](https://docs.pennylane.ai/en/stable/introduction/measurements.html), while running their jobs on [LRZ's Quantum Hardware](https://portal.quantum.lrz.de). Support for circuit optimization and Quantum Machine Learning is planned for future releases.

## 🛠️ Installation
To install the package, simply run 
```bash
pip install mqss-pennylane-adapter
```

## 🚀 Job Submission with different measurement types
MQSS PennyLane Provider has support for most of the native PennyLane features. For instance, you can define a quantum circuit using PennyLane quantum gates, and decorate the method with the MQSSPennylaneDevice object. Parametric gates can also be used. 
```python
import pennylane as qml
from pennylane import numpy as np
from mqss.pennylane_adapter.device import MQSSPennylaneDevice

dev = MQSSPennylaneDevice(wires=2, token=MQSS_TOKEN, backends=MQSS_BACKENDS)

@qml.qnode(dev, shots = 1024)
def quantum_function_expval(x, y):
    """
    Defines an arbitrary mock quantum circuit for testing purposes, with an expectation value measurement

    :param x: The parameter `x` represents the angle for the rotation gate
    `RZ` applied on the qubit at wire 0
    :param y: The parameter `y` represents the angle parameter for
    the rotation gate `RY` at wire 1.
    """
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RX(x, wires=1)
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1)) # Here, qubits 0 and 1 are being measured, in bases X and Z, respectively. 

params = [np.pi / 3, np.pi / 17]
result = quantum_function_expval(*params)
```
Furthermore, you can define a Hamiltonian object within PennyLane, and calculate the expectation value with respect to that Hamiltonian. For these cases, Pennylane Provider simply creates a batch job for each term in the Hamiltonian, to calculate the expectation value.

```python
import pennylane as qml
from pennylane import numpy as np
from mqss.pennylane_adapter.device import MQSSPennylaneDevice
dev_hamiltonian = MQSSPennylaneDevice(wires=2, token='<MQSS_TOKEN>', backends='<MQSS_BACKENDS>')

@qml.qnode(dev_hamiltonian, shots = 1024)
def quantum_function_hamiltonian_expval(
    x: float, y: float, H: qml.Hamiltonian
) -> float:
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    
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

If you are just interested in accessing the counts (or probabilities), you can also use

```python
import pennylane as qml
from pennylane import numpy as np
from mqss.pennylane_adapter.device import MQSSPennylaneDevice
n_wires = 5
dev = MQSSPennylaneDevice(wires=n_wires, token='<MQSS_TOKEN>', backends='<MQSS_BACKENDS>')
@qml.qnode(dev, shots = 1024)
def circuit(
    x: float, y: float
) -> float:
    
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    return qml.probs(qml.probs(wires=[0, 1]))
```

With the latest release (1.2.0), it is also possible to run high level PennyLane native quantum operations using the MQSS backends, such as `FermionicDoubleExcitation`:
```python
def build_h2_problem():
    symbols = ["H", "H"]
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.735],
        ],
        requires_grad=False,
    )
    mol = qml.qchem.Molecule(
        symbols, geometry, charge=0, mult=1, basis_name="sto-3g", unit="Angstrom"
    )
    H, n_qubits = qml.qchem.molecular_hamiltonian(mol, method="dhf")
    n_electrons = 2
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    n_params = len(singles) + len(doubles)
    return H, n_qubits, hf_state, s_wires, d_wires, n_params


def make_ansatz(hf_state, s_wires, d_wires, n_qubits):
    def ansatz(theta):
        qml.UCCSD(
            weights=theta,
            wires=range(n_qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=hf_state,
        )

    return ansatz

H, n_qubits, hf_state, s_wires, d_wires, n_params = build_h2_problem()
ansatz = make_ansatz(hf_state, s_wires, d_wires, n_qubits)

@qml.qnode(dev)
def energy_backend(theta):
    ansatz(theta)
    return qml.expval(H)

theta_test = np.zeros(n_params)
print("backend energy:", energy_backend(theta_test))
```


## 🧪 Inspecting the Quantum Circuit

It is possible to use PennyLane’s `specs` method to inspect details about a quantum circuit:

```python
_ = quantum_function_expval(*params)

resources = qml.specs(quantum_function_expval)(*params).resources
simulator_resources = qml.specs(quantum_function_expval_simulator)(
    *params
).resources

print(resources)
```

**Output:**
```text
Total wire allocations: 4
Total gates: 4
Circuit depth: 4

Gate types:
  BasisState: 1
  FermionicDoubleExcitation: 1
  FermionicSingleExcitation: 2

Measurements:
  expval(Sum(num_wires=4, num_terms=15)): 1
```

## 🛠️ Upcoming Features
 - Autograd support with parameter-shift
 - Grouping of commuting terms in the Hamiltonians to reduce the number of circuits in the batch

## 🤝 Contributing
Feel free to open [issues](https://github.com/Munich-Quantum-Software-Stack/MQSS-Pennylane-Adapter/issues) or submit [pull requests](https://github.com/Munich-Quantum-Software-Stack/MQSS-Pennylane-Adapter/pulls) to improve this project!
