"""MQPJob Module"""

from qiskit.providers import JobStatus as QiskitJobStatus  # type: ignore
from qiskit.providers import JobV1  # type: ignore
from qiskit.result import Counts, Result  # type: ignore

from mqss_client import JobStatus, MQSSClient, CircuitJobRequest  # type: ignore


class MQPJob(JobV1):
    """MQPJob Class"""

    def __init__(self, client: MQSSClient, job_id: str, job_request: CircuitJobRequest,**kwargs) -> None:
        super().__init__(None, job_id, **kwargs)
        self.client = client
        self.job_request = job_request

    def submit(self):
        return NotImplementedError("Submit jobs via the MQSSClient")

    def cancel(self):
        self.client.cancel_job(self.job_id(), self.job_request)

    def status(self):
        """Returns the job status

        Raises:
            RuntimeWarning: Unknown job status

        Returns:
            status: QiskitJobStatus
        """
        mqss_status = self.client.job_status(self.job_id(), self.job_request)
        if mqss_status == JobStatus.PENDING:
            return QiskitJobStatus.INITIALIZING
        if mqss_status == JobStatus.WAITING:
            return QiskitJobStatus.QUEUED
        if mqss_status == JobStatus.CANCELLED:
            return QiskitJobStatus.CANCELLED
        if mqss_status == JobStatus.FAILED:
            return QiskitJobStatus.ERROR
        if mqss_status == JobStatus.COMPLETED:
            return QiskitJobStatus.DONE
        raise RuntimeWarning(f"Unknown job status: {mqss_status}.")

    def result(self):
        """Fetches the results from the MQSS Backend and returns it to the user.
        The returned object includes:
        result_dict = {
            "job_id": self._job_id,
            "success": True|False,
            "results": [
                {
                    "shots": sum(_counts.values()),
                    "success": True|False,
                    "data": {
                        "counts": Counts(_counts),
                    },
                }
                for _counts in res_counts
            ],
            "timestamps": {
                "submitted": res.timestamp_submitted,
                "scheduled": res.timestamp_scheduled,
                "completed": res.timestamp_completed,
            },
        }

        Returns:
            result_dict: dict
        """
        res = self.client.wait_for_job_result(self.job_id(), self.job_request)
        if isinstance(res.counts, list):
            res_counts = res.counts
        else:
            res_counts = [res.counts]
        result_dict = {
            "backend_name": None,
            "backend_version": None,
            "qobj_id": None,
            "job_id": self.job_id(),
            "success": True,
            "results": [
                {
                    "shots": sum(_counts.values()),
                    "success": True,
                    "data": {
                        "counts": Counts(_counts),
                    },
                }
                for _counts in res_counts
            ],
            "timestamps": {
                "submitted": res.timestamp_submitted,
                "scheduled": res.timestamp_scheduled,
                "completed": res.timestamp_completed,
            },
        }
        return Result.from_dict(result_dict)
