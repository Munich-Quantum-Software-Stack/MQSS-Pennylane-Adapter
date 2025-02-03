from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.operation import Operator
from pennylane import numpy as np
import numpy.typing as npt
from pennylane.typing import TensorLike, PostprocessingFn, Result, ResultBatch
from typing import Optional, Union
import pennylane as qml

# The line `from .config import *` is importing all the variables and functions defined in the
# `config` module within the same package or directory as the current script. The `.` before `config`
# indicates that the module is in the same directory as the current script.
from src.mqp.pennylane_provider.provider import MQPProvider
from test.config import *  # TODO: Find a better solution
from src.mqp.pennylane_provider.utils import *

operations = frozenset(
    {"PauliX", "PauliY", "PauliZ", "Hadamard", "CNOT", "CZ", "RX", "RY", "RZ"}
)


def supports_operation(op: Operator) -> bool:
    """This function used by preprocessing determines what operations
    are natively supported by the device.

    While in theory ``simulate`` can support any operation with a matrix, we limit the target
    gate set for improved testing and reference purposes.

    """
    print()
    return getattr(op, "name", None) in operations


class LRZDevice(Device):
    """My Documentation."""

    def __init__(self, wires=None, shots=None, seed=None):
        super().__init__(wires=wires, shots=shots)
        # TODO1: Add transformations if necessary

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: "ExecutionConfig" = DefaultExecutionConfig, # type: ignore
        shots=1024,
    ) -> TensorLike:

        provider = MQPProvider(token=TEST_API_TOKEN, url=TEST_API_URL)
        #backend = provider.get_backend(TEST_API_BACKENDS)
        backend = provider.get_backend('Q20')

        # TODO: Add asserts to qasm
        for tape in circuits:
            assert all(supports_operation(op) for op in tape.operations)
            # pass
        job = None

        # Might not be required, batch jobs are handled in middleware level
        # if isinstance(circuits, qml.tape.QuantumScript):
        #     job = backend.run(circuits)
        # else:
        #     job = [backend.run(c) for c in circuits]
        #if type(circuits[0].measurements[0]).__name__ == "ExpectationMP": #if it's H then call fn that sends multiple circuits
        #    circuits = self.append_measurement_gates(circuits)
        job = backend.run(circuits, shots=shots)

        result = job.result()
        counts = self.fetch_counts(result, shots)

        measurement = self.calculate_measurement_type(counts, circuits, shots)
        return measurement

    def modify_circuit(self, circuit: QuantumScriptOrBatch) -> QuantumScriptOrBatch:
        """Given a quantum circuit, return a

        Args:
            circuit (QuantumScriptOrBatch): Pennylane circuit

        Returns:
            QuantumScriptOrBatch: Pennylane circuit
        """
        pass

    def calculate_measurement_type(
        self, counts: TensorLike, circuits: QuantumScriptOrBatch, shots: int
    ) -> Union[list[TensorLike], list[float]]:
        """Given a measurement type (e.g. probs, exp. val.), return the measurement result.

        Args:
            counts (list): List of sampled measurements
            circuits (qml.Qnode): Pennylane circuit
            shots (int): Number of shots

        """
        if type(circuits[0].measurements[0]).__name__ == "ProbabilityMP":
            return [np.sqrt(counts[0] / shots)]
        elif type(circuits[0].measurements[0]).__name__ == "ExpectationMP":
            print(1,circuits[0])
            print(2,circuits[0]._measurements[0])
            print(3,circuits[0]._measurements[0].obs)

            obs = circuits[0]._measurements[0].obs

            # If obs is a single Pauli operator, wrap it in a list
            if isinstance(obs, (qml.PauliX, qml.PauliY, qml.PauliZ)):
                print('hi')
                obs = [obs]

            measured_qubits = [op.wires.labels[0] for op in obs]

            expectation = 0.0
            for idx, value in enumerate(counts[0]):
                try:
                    bitstring = int2bit(idx, len(circuits[0].wires))
                    for bdx, bit in enumerate(bitstring):
                        weighted_count = value
                        if bdx in measured_qubits:
                            if bit == "1":
                                weighted_count *= -1
                            expectation += weighted_count
                    expectation /= shots

                except ValueError:
                    raise ValueError(
                        "Number of wires must be defined for expectation value calculation"
                    )
            return [expectation]

    def append_measurement_gates(
        self,
        circuits: QuantumScriptOrBatch,
    ) -> QuantumScriptOrBatch:
        """Append gates for basis change for measurements in non-Z basis.

        Args:
            circuits (QuantumScriptOrBatch): Pennylane circuit

        Returns:
            QuantumScriptOrBatch: Pennylane circuit
        """
        new_terms = []

        obs = circuits[0]._measurements[0].obs

        # If obs is a single Pauli operator, wrap it in a list
        if isinstance(obs, (qml.PauliX, qml.PauliY, qml.PauliZ)):
            print('hi')
            obs = [obs]
    
        for op in obs:
            qubit = op.wires.labels[0]
            basis = op.basis
            if basis == "X":
                hadamard = qml.H(qubit)
                circuits[0]._ops.append(hadamard)
            elif basis == "Y":
                sdg = qml.adjoint(qml.S(qubit))
                hadamard = qml.H(qubit)
                circuits[0]._ops.append(sdg)
                circuits[0]._ops.append(hadamard)
            new_terms.append(qml.PauliZ(qubit))
        return circuits #, qml.prod(*new_terms) # Return the updated tensor product of Pauli operators 

    def fetch_counts(
        self, result: Union[Result, ResultBatch], shots: int
    ) -> list[Result]:
        """Given a dictionary representing the measurements, return the probability distribution as an array

        Args:
            result (qiskit.Result): Results of the quantum job
            shots (int): Number of shots

        Returns:
            probs (np.ndarray): The probability distribution of the measurements
        """
        counts = result.get_counts()
        example_key = next(iter(counts.keys()))
        qubit_length = len(example_key)
        probs = np.zeros(2**qubit_length, dtype=np.float64)
        for count in zip(counts.keys(), counts.values()):
            key, value = count
            probs[bit2int(key)] = value

        return [probs]
