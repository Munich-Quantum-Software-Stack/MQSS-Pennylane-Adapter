from pennylane.operation import Operator

int2bit = lambda x, N: str(bin(x)[2:].zfill(N))
bit2int = lambda b: int("".join(str(bs) for bs in b), base=2)

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
