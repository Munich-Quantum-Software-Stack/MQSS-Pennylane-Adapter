from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.operation import Operator
from pennylane import numpy as np
import numpy.typing as npt
from pennylane.typing import TensorLike, PostprocessingFn, Result, ResultBatch
from typing import Optional, Union
import pennylane as qml
from pennylane.ops.op_math import LinearCombination, SProd, Sum

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
        backend = provider.get_backend(TEST_API_BACKENDS)

        # Might not be required, batch jobs are handled in middleware level
        # if isinstance(circuits, qml.tape.QuantumScript):
        #     job = backend.run(circuits)
        # else:
        #     job = [backend.run(c) for c in circuits]
        
   
        def run_job(circuits,shots):
            
            job = backend.run(circuits, shots=shots)
            result = job.result()
            counts = self.fetch_counts(result, shots)
            measurement = self.calculate_measurement_type(counts, circuits, shots)
            return measurement
        
        def handle_hamiltonian(circuit, observable, shots):

            total_result = 0.0 #qml.numpy.tensor(0.0, requires_grad = True)

            if isinstance(observable, (SProd)): # Eg: 5 * qml.PauliX(0) just one term with a scalar
                coeffs = [observable.scalar]
                ops = [observable.base]

            else:
                coeffs, ops = observable.terms()

        
            for coeff, obs in zip(coeffs, ops):
                # Create a new circuit with a single observable term
                original_circuit = circuit[0]

                # Create a new QuantumScript with the same operations but new measurements
                new_circuit = QuantumScript(
                    list(original_circuit.operations),  # Copy operations
                    [qml.expval(obs)],  # Replace measurement
                    shots=original_circuit.shots  # Preserve shots
                )

                modified_tuple = (new_circuit,) #rewrap in tuple
                expval = run_job(modified_tuple, shots)

                if isinstance(expval, list):
                    expval = expval[0]  # Assuming you want the first element in the list

                # Convert expval to float if it's a PennyLane tensor
                expval_float = expval.item() if isinstance(expval, qml.numpy.tensor) else float(expval)
                total_result += coeff * expval_float
                
            return [total_result]
      
        # Ensure all operations are supported
        # TODO: Add asserts to qasm
        for tape in circuits:
            assert all(supports_operation(op) for op in tape.operations)

        observable = circuits[0]._measurements[0].obs
        if isinstance(observable, (LinearCombination, SProd, qml.Hamiltonian, Sum)):
            return handle_hamiltonian(circuits, observable, shots)
        
        result = run_job(circuits, shots)
        return result # result is a list for both hamiltonian and this
        

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
            obs = circuits[0]._measurements[0].obs
            
            # If obs is a single Pauli operator, wrap it in a list
            if isinstance(obs, (qml.PauliX, qml.PauliY, qml.PauliZ)):
                obs = [obs]

            measured_qubits = []
            for op in obs:
                if not isinstance(op, qml.Identity):  #
                    measured_qubits.append(op.wires.labels[0])
    
            expectation = 0.0
            for idx, value in enumerate(counts[0]):
                try:
                    bitstring = int2bit(idx, len(circuits[0].wires)) #eg 000,001 for 3 qubits
                    weighted_count = value

                    for bdx, bit in enumerate(bitstring[::-1]):
                        if bdx in measured_qubits:
                            if bit == "1":
                                weighted_count *= -1
                    
                    expectation += weighted_count
                    
                except ValueError:
                    raise ValueError(
                        "Number of wires must be defined for expectation value calculation"
                    )

            expectation /= shots  
            return [expectation] 

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
