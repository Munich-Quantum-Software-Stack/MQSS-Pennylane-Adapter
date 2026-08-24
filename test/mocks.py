from unittest.mock import MagicMock, patch

from mqss_client.rest_client import RESTClient

from .config import CURRENT_RESOURCES

# =================== MOCK DATA ===================
MOCK_RESOURCES = {
    key: CURRENT_RESOURCES[key].__dict__ for key in CURRENT_RESOURCES
}

MOCK_JOB_DATA = {
    "normal": {# For resource endpoints
        "resources": MOCK_RESOURCES,
        "resources/Q5": CURRENT_RESOURCES["Q5"].__dict__,
        "resources/Q5/num_pending_jobs": {"num_pending_jobs": 3},
        # For job endpoints
        "job": {"jobs": ["mock-uuid-12345"]},
        "hamiltonian_job": {"jobs": ["mock-uuid-12345"]},
        # Status endpoints
        "job/mock-uuid-12345/status": {"status": "PENDING"},
        "hamiltonian_job/mock-uuid-12345/status": {"status": "PENDING"},
        # Result endpoints
        "job/mock-uuid-12345/result": {
            "result": '{"00": 500, "11": 500}',
            "timestamp_completed": "2023-04-14 10:15:30.123456",
            "timestamp_submitted": "2023-04-14 10:00:00.123456",
            "timestamp_scheduled": "2023-04-14 10:05:00.123456",
        },
        "hamiltonian_job/mock-uuid-12345/result": {
            "result": '{"00": 500, "11": 500}',
            "timestamp_completed": "2023-04-14 10:15:30.123456",
            "timestamp_submitted": "2023-04-14 10:00:00.123456",
            "timestamp_scheduled": "2023-04-14 10:05:00.123456",
        }


        },
    "hamiltonian": {
             "resources": MOCK_RESOURCES,
                "resources/Q5": CURRENT_RESOURCES["MAQCS"].__dict__,
                "resources/Q5/num_pending_jobs": {"num_pending_jobs": 3},
                # For job endpoints
                "job": {"jobs": ["mock-uuid-12345"]},
                "hamiltonian_job": {"jobs": ["mock-uuid-12345"]},
                # Status endpoints
                "job/mock-uuid-12345/status": {"status": "COMPLETED"},
                "hamiltonian_job/mock-uuid-12345/status": {"status": "COMPLETED"},
                # Result endpoints
                "job/mock-uuid-12345/result": {
                    'result': '[{"00": 3, "01": 85, "11": 936}, {"00": 40, "01": 44, "10": 501, "11": 439}]', 
                    'timestamp_completed': '2026-08-24 11:54:49.593776', 
                    'timestamp_scheduled': '2026-08-24 11:52:10.840863', 
                    'timestamp_submitted': '2026-08-24 11:52:07.858568'
                },
                "hamiltonian_job/mock-uuid-12345/result": {
                    'result': '[{"00": 3, "01": 85, "11": 936}, {"00": 40, "01": 44, "10": 501, "11": 439}]', 
                    'timestamp_completed': '2026-08-24 11:54:49.593776', 
                    'timestamp_scheduled': '2026-08-24 11:52:10.840863', 
                    'timestamp_submitted': '2026-08-24 11:52:07.858568'
                }
    }
    
}


'''
Result JSON: 
{'result': '[{"00": 3, "01": 85, "11": 936}, {"00": 40, "01": 44, "10": 501, "11": 439}]', 
'timestamp_completed': '2026-08-24 11:54:49.593776', 
'timestamp_scheduled': '2026-08-24 11:52:10.840863', 
'timestamp_submitted': '2026-08-24 11:52:07.858568'}

'''

# =================== REST CLIENT MOCKS ===================
def create_rest_mock():
    """Create and configure a mock REST client"""
    mock = MagicMock(spec=RESTClient)
    mock.post.return_value = {"uuid": "mock-uuid-12345"}
    mock.get.side_effect = lambda path: MOCK_JOB_DATA.get(path, {})
    return mock


def patch_mqss_rest_client():
    """Patch the RESTClient within MQSSClient with a mock"""
    return patch("mqss_client.mqss_client.RESTClient", return_value=create_rest_mock())
