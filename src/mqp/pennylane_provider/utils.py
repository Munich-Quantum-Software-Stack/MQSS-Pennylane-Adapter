from pennylane.operation import Operator


def int2bit(x: int, N: int):
    """Converts an integer to a binary

    Args:
        x (int): Input number
        N (int): Number of digits

    Returns:
        (str): Binary representation of the input number given as a string
    """
    return str(bin(x)[2:].zfill(N))


operations = frozenset(
    {"PauliX", "PauliY", "PauliZ", "Hadamard", "CNOT", "CZ", "RX", "RY", "RZ"}
)


def bit2int(b: str):
    """Given a binary as a string, find its integer representation

    Args:
        b (str): Binary in string

    Returns:
        (int): Integer representation
    """
    return int("".join(str(bs) for bs in b), base=2)


def supports_operation(op: Operator) -> bool:
    """This function used by preprocessing determines what operations
    are natively supported by the device.

    While in theory ``simulate`` can support any operation with a matrix, we limit the target
    gate set for improved testing and reference purposes.

    """
    return getattr(op, "name", None) in operations
