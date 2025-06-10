from typing import Optional

from qiskit.providers import ProviderV1  # type: ignore

from mqss_client import MQSSClient  # type: ignore

from .backend import MQSSPennylaneBackend


class MQSSPennylaneAdapter(ProviderV1):
    """MQSS Pennylane Adapter Class"""

    def __init__(self, token: str, url: Optional[str] = None) -> None:
        if url:
            self._client = MQSSClient(base_url=url, token=token)
        else:
            self._client = MQSSClient(token=token)

    def get_backend(self, name=None, **kwargs) -> MQSSPennylaneBackend:
        return MQSSPennylaneBackend(name, self._client, **kwargs)

    def backends(self, name=None, **kwargs):
        resources = self._client.resources()
        if resources is None:
            return []
        return [
            MQSSPennylaneBackend(name, self._client, resources[name])
            for name in resources
        ]
