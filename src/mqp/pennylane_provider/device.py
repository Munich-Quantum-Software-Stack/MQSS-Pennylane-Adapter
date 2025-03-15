from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScriptOrBatch

from pennylane import numpy as np
from pennylane.typing import TensorLike, Result, ResultBatch
from typing import Union
import pennylane as qml


from src.mqp.pennylane_provider.provider import MQSSPennylaneProvider
from test.config import *
from src.mqp.pennylane_provider.utils import *


class MQSSPennylaneDevice(Device):
    """My Documentation."""

    def __init__(self, wires=None, shots=None, seed=None):
        super().__init__(wires=wires, shots=shots)
        # TODO1: Add transformations if necessary

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: "ExecutionConfig" = DefaultExecutionConfig,
        shots=1024,
    ) -> TensorLike:

        provider = MQSSPennylaneProvider(token=TEST_API_TOKEN, url=TEST_API_URL)
        backend = provider.get_backend(TEST_API_BACKENDS)
        print(
            circuits[0].to_openqasm(rotations=False)
        )  # provide rotations False to see what happens
        for tape in circuits:
            assert all(supports_operation(op) for op in tape.operations)

        # Create a batch for a Hamiltonian calculation
        if (tape.measurements[0]) is not None:
            if len(tape.measurements[0].obs) > 1:
                for obs in circuits[0]._measurements[0].obs:
                    modified_circuit = self.append_measurement_gates(circuits[0], obs)
                    pass

        job = None

        # Might not be required, batch jobs are handled in middleware level
        # if isinstance(circuits, qml.tape.QuantumScript):
        #     job = backend.run(circuits)
        # else:
        #     job = [backend.run(c) for c in circuits]
        # if type(circuits[0].measurements[0]).__name__ == "ExpectationMP":
        #     circuits = self.append_measurement_gates(circuits)
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
            measured_qubits = [
                op.wires.labels[0] for op in circuits[0]._measurements[0].obs
            ]
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
        self, circuits: QuantumScriptOrBatch, obs: qml.ops
    ) -> QuantumScriptOrBatch:
        """Append gates for basis change for measurements in non-Z basis.

        Args:
            circuits (QuantumScriptOrBatch): Pennylane circuit

        Returns:
            QuantumScriptOrBatch: Pennylane circuit
        """
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
        return circuits

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
