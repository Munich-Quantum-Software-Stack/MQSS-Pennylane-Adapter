from .job import MQPJob
from typing import List, Optional, Union
from qiskit.transpiler import CouplingMap, Target
from pennylane.tape import QuantumTape, QuantumScript

from qiskit.providers import BackendV2, Options  # type: ignore

from mqss_client import MQSSClient, CircuitJobRequest, ResourceInfo  # type: ignore
from qiskit.circuit import QuantumCircuit
from .mqss_resources import get_coupling_map, get_target


class MQSSPennylaneBackend(BackendV2):
    """MQP Pennylane Backend class responsible for handling requests and responses from the MQSS Backend."""

    def __init__(
        self,
        name: str,
        client: MQSSClient,
        resource_info: Optional[ResourceInfo] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.client = client
        _resource_info = resource_info or (
            self.client.get_resource_info(self.name) if name else None
        )
        self._coupling_map = get_coupling_map(_resource_info)
        self._target = get_target(_resource_info)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024, qubit_mapping=None, calibration_set_id=None, no_modify=False
        )

    @property
    def coupling_map(self) -> CouplingMap:
        """Returns the coupling map of the selected backend

        Returns:
            CouplingMap: CouplingMap
        """
        return self._coupling_map

    @property
    def target(self) -> Target:
        if self._target is None:
            raise NotImplementedError(f"Target for {self.name} is not available.")
        return self._target

    @property
    def max_circuits(self) -> Optional[int]:
        return None

    def run(
        self,
        run_input: Union[QuantumCircuit, List[QuantumCircuit]],
        shots: int = 1024,
        no_modify: bool = False,
        queued: bool = False,
        **options,
    ) -> MQPJob:
        """Sends the quantum circuit(s) to the selected backend.

        Args:
            run_input (Union[QuantumCircuit, List[QuantumCircuit]]): Pennylane circuit
            shots (int, optional): Number of shots. Defaults to 1024.
            no_modify (bool, optional): Flag to bypass MQSS transpilation. no_modify=True means the transpilation will be bypassed if possible. Defaults to False.

        Returns:
            MQPJob: Returns the MQPJob object
        """
        if isinstance(run_input, QuantumTape) or isinstance(run_input, QuantumScript):
            _circuits = str([run_input.to_openqasm(rotations=False)])
        else:
            _circuits = str([qc.to_openqasm(rotations=False) for qc in run_input])
        _circuit_format = "qasm"

        job_request = CircuitJobRequest(
            circuits=_circuits,
            circuit_format=_circuit_format,
            resource_name=self.name,
            shots=shots,
            no_modify=no_modify,
            queued=queued,
        )
        job_id = self.client.submit_job(job_request)
        return MQPJob(self.client, job_id, job_request)
