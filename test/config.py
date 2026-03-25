from mqss_client.resource_info import ResourceInfo

CURRENT_RESOURCES = {
    "Q5": ResourceInfo(
        name="Q5",
        qubits=5,
        online=True,
        connectivity=None,
        instructions=None,
    ),
    "Q20": ResourceInfo(
        name="Q20",
        qubits=20,
        online=True,
        connectivity=None,
        instructions=None,
    ),
    "QExa20": ResourceInfo(
        name="QExa20", qubits=20, online=True, connectivity=None, instructions=None
    ),
    "AQT20": ResourceInfo(
        name="AQT20", qubits=12, online=True, connectivity=None, instructions=None
    ),
    "WMI3": ResourceInfo(
        name="WMI3", qubits=3, online=True, connectivity=None, instructions=None
    ),
    "QLM": ResourceInfo(
        name="QLM", qubits=38, online=True, connectivity=None, instructions=None
    ),
    "EQE1": ResourceInfo(
        name="EQE1", qubits=54, online=True, connectivity=None, instructions=None
    ),
}

QASM_FILE = "test/example.qasm"


def get_qasm() -> str:
    with open(QASM_FILE, "r") as f:
        return f.read()
