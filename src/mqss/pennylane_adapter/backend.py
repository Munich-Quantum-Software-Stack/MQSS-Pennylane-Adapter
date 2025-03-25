from .job import MQPJob
from typing import List, Optional, Union
from qiskit.transpiler import CouplingMap, Target
from pennylane.tape import QuantumTape, QuantumScript

from qiskit.providers import BackendV2, Options  # type: ignore

from mqp_client import MQPClient, ResourceInfo  # type: ignore
from qiskit.circuit import QuantumCircuit
from .mqp_resources import get_coupling_map, get_target


class MQSSPennylaneBackend(BackendV2):
    """MQP Pennylane Backend class"""

    def __init__(
        self,
        name: str,
        client: MQPClient,
        resource_info: Optional[ResourceInfo] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.client = client
        _resource_info = resource_info or self.client.resource_info(self.name)

        # assert _resource_info is not None
        self._coupling_map = get_coupling_map(_resource_info)
        self._target = get_target(_resource_info)

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=1024, qubit_mapping=None, calibration_set_id=None, no_modify=False
        )

    @property
    def coupling_map(self) -> CouplingMap:
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
        **options,
    ) -> MQPJob:

        # Convert Pennylane QuantumTape(s) to QASM for the MQSS-Client
        if isinstance(run_input, QuantumTape) or isinstance(run_input, QuantumScript):
            _circuits = str([run_input.to_openqasm(rotations=False)])
        else:
            _circuits = str([qc.to_openqasm(rotations=False) for qc in run_input])
        _circuit_format = "qasm"

        job_id = self.client.submit_job(
            resource_name=self.name,
            circuit=_circuits,
            circuit_format=_circuit_format,
            shots=shots,
            no_modify=no_modify,
        )
        return MQPJob(self.client, job_id)
