from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScriptOrBatch

from pennylane import numpy as np
from pennylane.typing import TensorLike, Result, ResultBatch
from typing import Union
import copy
import pennylane as qml


from src.mqp.pennylane_provider.provider import MQSSPennylaneProvider
from src.config import MQSS_URL
from src.mqp.pennylane_provider.utils import (
    int2bit,
    bit2int,
    supports_operation,
)


class MQSSPennylaneDevice(Device):
    """My Documentation."""

    def __init__(
        self,
        token: str,
        backends: str,
        wires=None,
        shots=None,
        seed=None,
        supports_derivatives=False,
    ):
        super().__init__(wires=wires, shots=shots)
        self.TOKEN = token
        self.BACKENDS = backends

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: DefaultExecutionConfig,
        shots=1024,
    ) -> TensorLike:

        provider = MQSSPennylaneProvider(token=self.TOKEN, url=MQSS_URL)
        backend = provider.get_backend(self.BACKENDS)
        is_hamiltonian = False
        for tape in circuits:
            try:
                self.validate_tape_operations(tape)
            except ValueError as e:
                print(
                    f"Skipping tape due to error in validating operations, original exception: {e}"
                )

        # TODO2:What happens if there is no measurement instruction, or probs?

        if isinstance(tape.measurements[0], qml.measurements.ExpectationMP):
            if isinstance(tape.measurements[0].obs, qml.ops.op_math.LinearCombination):
                is_hamiltonian = True
                circuits = self.create_batch_circuits_for_hamiltonians(
                    circuits[0], is_hamiltonian
                )
                is_hamiltonian = True
            elif isinstance(tape.measurements[0].obs, qml.ops.op_math.Prod):
                circuits = self.create_batch_circuits_for_hamiltonians(
                    circuits[0], is_hamiltonian
                )

        job = backend.run(circuits, shots=shots)

        result = job.result()
        counts = self.fetch_counts(result, shots)

        measurement = self.calculate_measurement_type(
            counts, circuits, shots, is_hamiltonian
        )
        return [measurement]

    def create_batch_circuits_for_hamiltonians(
        self, tape: QuantumScriptOrBatch, is_hamiltonian: bool
    ) -> list[QuantumScriptOrBatch]:
        """Creates a batched job where there is a Hamiltonian expectation value calculation as measurement

        Args:
            tape (QuantumScriptOrBatch): Original quantum circuit
            is_hamiltonian (bool): Indicates if there is a hamiltonian expectation value calculation

        Returns:
            list[QuantumScriptOrBatch]: Batch of circuits for each term in the Hamiltonian

        """
        batched_circuits = []
        if is_hamiltonian:
            observables = tape._measurements[0].obs
        else:
            observables = [tape._measurements[0].obs]
        for obs in observables:
            modified_circuit = self.append_measurement_gates(
                copy.deepcopy(tape), obs, is_hamiltonian
            )
            modified_circuit._measurements = [qml.expval(obs)]
            batched_circuits.append(modified_circuit)

        return batched_circuits

    def validate_tape_operations(self, tape: QuantumScriptOrBatch):
        """Validate if the operations in the tape are all supported

        Args:
            tape (QuantumScriptOrBatch): Pennylane circuit
        Raises:
            ValueError: If an unsupported operation is found.

        """
        if not all(supports_operation(op) for op in tape.operations):
            raise ValueError(f"Unsupported operation found in tape: {tape}")

    def calculate_measurement_type(
        self,
        counts: TensorLike,
        circuits: QuantumScriptOrBatch,
        shots: int,
        is_hamiltonian: bool,
    ) -> Union[list[TensorLike], list[float]]:
        """Given a measurement type (e.g. probs, exp. val.), return the measurement result.

        Args:
            counts (list): List of sampled measurements
            circuits (qml.Qnode): Pennylane circuit
            shots (int): Number of shots
            is_hamiltonian (bool): Indicates if there is a hamiltonian expectation value calculation

        """
        if type(circuits[0].measurements[0]).__name__ == "ProbabilityMP":
            return [np.sqrt(counts[0] / shots)]
        elif type(circuits[0].measurements[0]).__name__ == "ExpectationMP":
            final_expectation = 0
            for cdx, count in enumerate(counts):
                if is_hamiltonian:
                    measured_qubits = [
                        op.wires.labels[0]
                        for op in circuits[cdx]._measurements[0].obs.base
                    ]
                else:
                    measured_qubits = [
                        op.wires.labels[0] for op in circuits[cdx]._measurements[0].obs
                    ]

                num_qubits = len(circuits[cdx].wires)
                expectation = self.get_expectation_value(
                    count, measured_qubits, num_qubits, shots
                )
                if is_hamiltonian:
                    final_expectation += (
                        expectation * circuits[0]._measurements[0].obs.scalar
                    )
                else:
                    final_expectation += expectation
            return final_expectation

    def get_expectation_value(
        self,
        count: list[float],
        measured_qubits: list[int],
        num_qubits: int,
        shots: int,
    ):
        """Calculate the expectation value from the counts

        Args:
            counts (list[float]): List of sampled measurements
            measured_qubits (list[int]): The qubits involved in measurement process
            num_qubits (int): Number of circuits of the given circuit
            shots (int): Number of shots

        Raises:
            ValueError: Raised in case of the number of wires missing
        """
        expectation = 0.0
        for idx, value in enumerate(count):
            try:
                bitstring = int2bit(idx, num_qubits)
                for bdx, bit in enumerate(bitstring):
                    weighted_count = value
                    if bdx in measured_qubits:
                        if bit == "1":
                            weighted_count *= -1
                        expectation += weighted_count
                expectation /= shots

            except ValueError as e:
                raise ValueError(
                    f"Number of wires must be defined for expectation value calculation, original error: {e}"
                )
        return expectation

    def append_measurement_gates(
        self, circuits: QuantumScriptOrBatch, obs: qml.ops, is_hamiltonian: bool
    ) -> QuantumScriptOrBatch:
        """Append gates for basis change for measurements in non-Z basis.

        Args:
            circuits (QuantumScriptOrBatch): Pennylane circuit
            obs (qml.ops): The measured observable
            is_hamiltonian (bool): Indicates if there is a hamiltonian expectation value calculation
        Returns:
            QuantumScriptOrBatch: Pennylane circuit
        """
        try:
            if is_hamiltonian:
                observables = obs.base
            else:
                observables = obs
            for op in observables if (observables.num_wires > 1) else [observables]:
                qubit = op.wires.labels[0]
                basis = op.basis
                if basis == "X":
                    hadamard = qml.H(qubit)
                    circuits._ops.append(hadamard)
                elif basis == "Y":
                    hadamard = qml.H(qubit)
                    sdg = qml.adjoint(qml.S(qubit))
                    circuits._ops.append(hadamard)
                    circuits._ops.append(sdg)

        except TypeError as e:
            raise TypeError(
                f" Excepted an iterable object, but got {type(obs)}. Original error: {e}"
            )

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
        probs_list = []
        if not isinstance(counts, list):
            counts = [counts]
        for count in counts:
            example_key = next(iter(count.keys()))
            qubit_length = len(example_key)
            probs = np.zeros(2**qubit_length, dtype=np.float64)
            for count in zip(count.keys(), count.values()):
                key, value = count
                probs[bit2int(key)] = value
            probs_list.append(probs)
        return probs_list
