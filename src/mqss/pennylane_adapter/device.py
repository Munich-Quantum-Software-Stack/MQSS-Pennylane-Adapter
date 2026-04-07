from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScriptOrBatch

from pennylane import numpy as np
from pennylane.typing import TensorLike
from typing import Union, Tuple
import copy
import pennylane as qml
from mqss.pennylane_adapter.adapter import MQSSPennylaneAdapter
from mqss.pennylane_adapter.config import MQSS_URL
from .utils import int2bit, bit2int, supports_operation, operations

from enum import Enum, auto


class MeasurementType(Enum):
    PROBS = auto()
    EXPVAL = auto()
    EXPVAL_HAMILTONIAN = auto()
    SAMPLE = auto()
    STATE = auto()
    UNKNOWN = auto()


class MQSSPennylaneDevice(Device):
    """Implements a Custom Pennylane Device that uses MQSS as a backend.

    Attributes
    ----------
    TOKEN : str
        Munich Quantum Portal (MQP) Token
    BACKENDS : str
        Munich Quantum Portal (MQP) Backend
    Methods
    -------
    __init__(self):
        Constructor
    execute(self, circuits, execution_config, shots):
        Sends the Pennylane circuit to the specified MQSS backend.

    """

    def __init__(
        self,
        token: str,
        backends: str,
        wires=None,
        shots=1024,
        seed=None,
        supports_derivatives=False,
    ):
        """Construct an MQSSPennylaneDevice Object

        Args:
            token (str): Munich Quantum Portal (MQP) token
            backends (str): MQP backend
            wires (int, optional): Number of wires in the circuit Defaults to None.
            shots (int, optional): Number of shots, for expectation values leave it as None. Defaults to 1024.
            seed (int, optional): Defaults to None.
            supports_derivatives (bool, optional): Boolean flag for autograd support. Defaults to False.
        """
        super().__init__(wires=wires)

        self.TOKEN = token
        self.BACKENDS = backends
        self.measurement_type: MeasurementType = MeasurementType.UNKNOWN
        self.batch_circuits: bool = False

    def determine_measurement_type(
        self, circuit: QuantumScriptOrBatch
    ) -> MeasurementType:
        """Determines the measurement type of the given tape.

        Args:
            tape (QuantumScriptOrBatch): Pennylane circuit

        Returns:
            MeasurementType: The type of measurement in the tape
        """
        if not circuit.measurements:
            return MeasurementType.UNKNOWN
        measurement = circuit.measurements[0]
        if isinstance(measurement, qml.measurements.ProbabilityMP):
            return MeasurementType.PROBS

        elif isinstance(measurement, qml.measurements.ExpectationMP):
            if isinstance(measurement.obs, qml.ops.op_math.LinearCombination):
                return MeasurementType.EXPVAL_HAMILTONIAN
            else:
                return MeasurementType.EXPVAL

        elif isinstance(measurement, qml.measurements.SampleMP):
            return MeasurementType.SAMPLE
        elif isinstance(measurement, qml.measurements.StateMP):
            return MeasurementType.STATE
        else:
            return MeasurementType.UNKNOWN

    def execute(
        self,
        circuits: Tuple[QuantumScriptOrBatch],
        execution_config: ExecutionConfig,
    ) -> TensorLike:
        """Sends the Pennylane circuit to the specified MQSS backend.

        Args:
            circuits (QuantumScriptOrBatch): Pennylane circuit
            execution_config (ExecutionConfig): Additional config for the circuit if necessary
            shots (int, optional): Number of shots. Defaults to 1024.

        Returns:
            TensorLike: Measurement results
        """
        shots = (
            circuits[0].shots.total_shots
            if circuits[0].shots.total_shots is not None
            else 1024
        )
        self.batch_circuits = False
        if isinstance(circuits, list):
            self.batch_circuits = True

        circuit = circuits[0]
        self.measurement_type = self.determine_measurement_type(circuit)
        adapter = MQSSPennylaneAdapter(token=self.TOKEN, url=MQSS_URL)
        backend = adapter.get_backend(self.BACKENDS)
        is_hamiltonian = False

        circuit, _ = qml.transforms.decompose(
            circuit,
            gate_set=operations,
        )
        circuit = circuit[0]

        if (
            self.measurement_type == MeasurementType.EXPVAL_HAMILTONIAN
            or self.measurement_type == MeasurementType.EXPVAL
        ):
            if isinstance(
                circuit.measurements[0].obs, qml.ops.op_math.LinearCombination
            ):
                is_hamiltonian = True
            circuit = self.create_batch_circuits_for_hamiltonians(
                circuit, is_hamiltonian
            )

        job = backend.run(circuit, shots=shots)
        result = job.result()

        measurement = self.calculate_measurement_type(
            result, circuit, shots, is_hamiltonian
        )
        return measurement

    def create_batch_circuits_for_hamiltonians(
        self, circuit: QuantumScriptOrBatch, is_hamiltonian: bool
    ) -> Union[list[QuantumScriptOrBatch], QuantumScriptOrBatch]:
        """Creates a batched job where there is a Hamiltonian expectation value calculation as measurement

        Args:
            tape (QuantumScriptOrBatch): Original quantum circuit
            is_hamiltonian (bool): Indicates if there is a hamiltonian expectation value calculation

        Returns:
            list[QuantumScriptOrBatch]: Batch of circuits for each term in the Hamiltonian

        """
        if (
            self.measurement_type != MeasurementType.EXPVAL_HAMILTONIAN
            and self.measurement_type != MeasurementType.EXPVAL
        ):
            return circuit

        batched_circuits = []
        if is_hamiltonian:
            observables = circuit.measurements[0].obs
        else:
            observables = [circuit.measurements[0].obs]
        for obs in observables:
            modified_circuit = self.append_measurement_gates(
                copy.deepcopy(circuit), obs, is_hamiltonian
            )
            # modified_circuit.measurements = [qml.expval(obs)]
            batched_circuits.append(modified_circuit)
        if len(batched_circuits) > 1:
            self.batch_circuits = True

        else:
            batched_circuits = batched_circuits[0]

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
        result,
        circuits: QuantumScriptOrBatch,
        shots: int,
        is_hamiltonian: bool,
    ) -> Union[list[TensorLike], Union[TensorLike, list[float]]]:
        """Given a measurement type (e.g. probs, exp. val.), return the measurement result.

        Args:
            counts (list): List of sampled measurements
            circuits (qml.Qnode): Pennylane circuit
            shots (int): Number of shots
            is_hamiltonian (bool): Indicates if there is a hamiltonian expectation value calculation

        """

        counts = self.fetch_counts(result.get_counts(), shots)
        if self.measurement_type == MeasurementType.PROBS:

            measurement = []
            for count in counts:
                measurement.append(count / shots)
            return measurement
        elif (
            self.measurement_type == MeasurementType.EXPVAL
            or self.measurement_type == MeasurementType.EXPVAL_HAMILTONIAN
        ):

            final_expectation = 0
            for cdx, count in enumerate(counts):
                if self.batch_circuits:
                    measurement = circuits[0].measurements[0]
                else:
                    measurement = circuits.measurements[0]
                observable = getattr(measurement, "obs", None)

                if is_hamiltonian:
                    if observable is None:
                        measured_qubits = list(measurement.wires.labels)
                    else:
                        base_observable = getattr(observable, "base", observable)
                        if hasattr(base_observable, "operands"):
                            obs_terms = base_observable.operands
                        else:
                            obs_terms = [base_observable]

                        measured_qubits = [op.wires.labels for op in obs_terms]
                else:
                    if observable is None:
                        measured_qubits = list(measurement.wires.labels)
                    else:
                        if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
                            measured_qubits = [tuple([observable.wires.labels[0]])]
                        elif hasattr(observable, "operands"):
                            measured_qubits = [observable.wires.labels]
                        else:
                            measured_qubits = [observable.wires.labels[0]]

                if self.batch_circuits:

                    num_qubits = len(circuits[0].wires)
                else:
                    num_qubits = len(circuits.wires)

                expectation = self.get_expectation_value(
                    count, measured_qubits[cdx], num_qubits, shots
                )
                if is_hamiltonian:
                    final_expectation += expectation * observable.coeffs[cdx]
                else:
                    final_expectation += expectation
            return [final_expectation]
        elif self.measurement_type == MeasurementType.SAMPLE:
            raise NotImplementedError
        elif self.measurement_type == MeasurementType.STATE:
            raise NotImplementedError
        else:
            raise ValueError("Unknown measurement type")

    def get_expectation_value(
        self,
        count: list[float],
        measured_qubits: tuple[int],
        num_qubits: int,
        shots: int,
    ):
        """Calculate the expectation value from the counts

        Args:
            counts (list[float]): List of sampled measurements
            measured_qubits (tuple[int]): The qubits involved in measurement process
            num_qubits (int): Number of circuits of the given circuit
            shots (int): Number of shots

        Raises:
            ValueError: Raised in case of the number of wires missing
        """
        expectation = 0.0
        for idx, value in enumerate(count):

            try:
                weighted_count = 1
                bitstring = int2bit(idx, num_qubits)
                for bdx, bit in enumerate(bitstring):
                    if bdx in measured_qubits:
                        if bit == "1":
                            weighted_count *= -1
                expectation += weighted_count * value

            except ValueError as e:
                raise ValueError(
                    f"Number of wires must be defined for expectation value calculation, original error: {e}"
                )
        expectation /= shots

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
        self, counts: dict[str, int] | list[dict[str, int]], shots: int
    ) -> list[dict[str, int] | list[dict[str, int]]]:
        """Given a dictionary representing the measurements, return the probability distribution as an array

        Args:
            result (qiskit.Result): Results of the quantum job
            shots (int): Number of shots

        Returns:
            probs (np.ndarray): The probability distribution of the measurements
        """

        probs_list = []
        if not isinstance(counts, list):
            counts = [counts]
        for count in counts:
            reversed_count = self.reverse_bit_order(count)
            example_key = next(iter(reversed_count.keys()))
            qubit_length = len(example_key)
            probs = np.zeros(2**qubit_length, dtype=np.float64)
            for count_id in zip(reversed_count.keys(), reversed_count.values()):
                key, value = count_id
                probs[bit2int(key)] = value
            probs_list.append(probs)
        return probs_list

    def reverse_bit_order(self, counts: dict) -> dict:
        """Reverse the order of bits in a bitstring.
        Pennylane uses big-endian ending which contradicts with Qiskit's little-endian ordering, so we need to reverse the bit order when converting between the two formats.
        This function takes a bitstring as input and returns the reversed bitstring.


        Args:
            counts (dict): The counts dictionary whose input bitstrings is to be reversed.

        Returns:
            dict: The reversed bitstring.
        """
        reversed_counts = {}
        for count in zip(counts.keys(), counts.values()):
            key, value = count
            reversed_key = key[::-1]
            reversed_counts[reversed_key] = value
        return reversed_counts
