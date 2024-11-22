from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.operation import Operator

# The line `from .config import *` is importing all the variables and functions defined in the
# `config` module within the same package or directory as the current script. The `.` before `config`
# indicates that the module is in the same directory as the current script.
from src.mqp.pennylane_provider.provider import MQPProvider
from test.config import *  # TODO: Find a better solution

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
        execution_config: "ExecutionConfig" = DefaultExecutionConfig,
        shots=1024,
    ):
        # TODO: Check what is execution_config and DefaultExecutionConfig
        provider = MQPProvider(token=TEST_API_TOKEN, url=TEST_API_URL)

        backend = provider.get_backend(TEST_API_BACKENDS)
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
        print("0")
        job = backend.run(circuits, shots=shots)
        # TODO: Make sure what the return type should be equal to.
        print(job)
        print("1")
        result = job.result()

        print("2")
        print(result)
        return job
        # return result.get_counts()
