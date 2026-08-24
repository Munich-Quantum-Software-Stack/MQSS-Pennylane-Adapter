from unittest.mock import MagicMock, patch

from mqss_client.rest_client import RESTClient

from .config import CURRENT_RESOURCES

# =================== MOCK DATA ===================
MOCK_RESOURCES = {
    key: CURRENT_RESOURCES[key].__dict__ for key in CURRENT_RESOURCES
}

MOCK_JOB_DATA = {
    "normal": {  # For resource endpoints
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
        },
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
            "result": '[{"00": 3, "01": 85, "11": 936}, {"00": 40, "01": 44, "10": 501, "11": 439}]',
            "timestamp_completed": "2026-08-24 11:54:49.593776",
            "timestamp_scheduled": "2026-08-24 11:52:10.840863",
            "timestamp_submitted": "2026-08-24 11:52:07.858568",
        },
        "hamiltonian_job/mock-uuid-12345/result": {
            "result": '[{"00": 3, "01": 85, "11": 936}, {"00": 40, "01": 44, "10": 501, "11": 439}]',
            "timestamp_completed": "2026-08-24 11:54:49.593776",
            "timestamp_scheduled": "2026-08-24 11:52:10.840863",
            "timestamp_submitted": "2026-08-24 11:52:07.858568",
        },
    },
}


MOCK_RESULTS = {
    "test_compare_generated_circuits[params0]": {
        "result": '[{"00": 277, "01": 236, "10": 241, "11": 270}]',
        "timestamp_completed": "2026-08-24 20:47:46.308632",
        "timestamp_scheduled": "2026-08-24 20:46:18.527586",
        "timestamp_submitted": "2026-08-24 20:46:15.905337",
    },
    "test_counts_single_wire[params0]": {
        "result": '[{"01": 86, "11": 938}]',
        "timestamp_completed": "2026-08-24 20:49:18.584341",
        "timestamp_scheduled": "2026-08-24 20:47:52.364818",
        "timestamp_submitted": "2026-08-24 20:47:49.735804",
    },
    "test_counts_all_wires[params0]": {
        "result": '[{"00": 3, "01": 99, "11": 922}]',
        "timestamp_completed": "2026-08-24 20:50:42.609569",
        "timestamp_scheduled": "2026-08-24 20:49:19.637926",
        "timestamp_submitted": "2026-08-24 20:49:19.186776",
    },
    "test_counts_all_outcomes": {
        "result": '[{"00": 1021, "10": 1, "11": 2}]',
        "timestamp_completed": "2026-08-24 20:52:09.944271",
        "timestamp_scheduled": "2026-08-24 20:50:46.664719",
        "timestamp_submitted": "2026-08-24 20:50:44.554766",
    },
    "test_counts_matches_simulator[params0]": {
        "result": '[{"00": 804, "01": 9, "10": 193, "11": 18}]',
        "timestamp_completed": "2026-08-24 20:53:39.054322",
        "timestamp_scheduled": "2026-08-24 20:52:13.876448",
        "timestamp_submitted": "2026-08-24 20:52:10.891891",
    },
    "test_counts_matches_simulator[params1]": {
        "result": '[{"00": 1018, "01": 2, "11": 4}]',
        "timestamp_completed": "2026-08-24 20:55:06.881543",
        "timestamp_scheduled": "2026-08-24 20:53:44.422734",
        "timestamp_submitted": "2026-08-24 20:53:41.148439",
    },
    "test_expectation_value_measurements[obs0-params0]": {
        "result": '[{"00": 467, "01": 39, "10": 493, "11": 25}]',
        "timestamp_completed": "2026-08-24 20:56:31.633212",
        "timestamp_scheduled": "2026-08-24 20:55:11.593282",
        "timestamp_submitted": "2026-08-24 20:55:08.436907",
    },
    "test_expectation_value_measurements[obs0-params1]": {
        "result": '[{"00": 518, "01": 2, "10": 504}]',
        "timestamp_completed": "2026-08-24 20:57:55.452667",
        "timestamp_scheduled": "2026-08-24 20:56:35.665531",
        "timestamp_submitted": "2026-08-24 20:56:33.411548",
    },
    "test_expectation_value_measurements[obs1-params0]": {
        "result": '[{"00": 800, "01": 1, "10": 183, "11": 40}]',
        "timestamp_completed": "2026-08-24 20:59:24.119698",
        "timestamp_scheduled": "2026-08-24 20:57:59.906564",
        "timestamp_submitted": "2026-08-24 20:57:58.428469",
    },
    "test_expectation_value_measurements[obs1-params1]": {
        "result": '[{"00": 1022, "10": 2}]',
        "timestamp_completed": "2026-08-24 21:08:49.577835",
        "timestamp_scheduled": "2026-08-24 21:07:18.625465",
        "timestamp_submitted": "2026-08-24 21:07:16.820790",
    },
    "test_hamiltonian_measurements[hamiltonian_data0-params1]": {
        "result": '[{"00": 1017, "01": 2, "10": 2, "11": 3}, {"00": 516, "01": 506, "11": 2}]',
        "timestamp_completed": "2026-08-24 22:15:44.694427",
        "timestamp_scheduled": "2026-08-24 22:12:48.998477",
        "timestamp_submitted": "2026-08-24 22:12:48.554165",
    },
    "test_hamiltonian_measurements[hamiltonian_data1-params0]": {
        "result": '[{"00": 470, "01": 14, "10": 530, "11": 10}]',
        "timestamp_completed": "2026-08-24 22:17:42.645021",
        "timestamp_scheduled": "2026-08-24 22:16:14.857459",
        "timestamp_submitted": "2026-08-24 22:16:12.044152",
    },
    "test_hamiltonian_measurements[hamiltonian_data1-params1]": {
        "result": '[{"00": 513, "10": 511}]',
        "timestamp_completed": "2026-08-24 22:19:13.353910",
        "timestamp_scheduled": "2026-08-24 22:17:48.232685",
        "timestamp_submitted": "2026-08-24 22:17:45.562559",
    },
    "test_hamiltonian_measurements[hamiltonian_data2-params0]": {
        "result": '[{"00": 793, "01": 7, "10": 199, "11": 25}, {"00": 377, "01": 137, "10": 159, "11": 351}]',
        "timestamp_completed": "2026-08-24 22:25:41.682810",
        "timestamp_scheduled": "2026-08-24 22:22:47.275222",
        "timestamp_submitted": "2026-08-24 22:22:46.590544",
    },
    "test_hamiltonian_measurements[hamiltonian_data2-params1]": {
        "result": '[{"00": 1023, "11": 1}, {"00": 297, "01": 224, "10": 225, "11": 278}]',
        "timestamp_completed": "2026-08-24 22:28:44.018090",
        "timestamp_scheduled": "2026-08-24 22:25:47.341915",
        "timestamp_submitted": "2026-08-24 22:25:44.143619",
    },
    "test_probs[params0]": {
            "result": '[{"00": 810, "01": 5, "10": 181, "11": 28}]',
            "timestamp_completed": "2026-08-24 22:28:44.018090",
            "timestamp_scheduled": "2026-08-24 22:25:47.341915",
            "timestamp_submitted": "2026-08-24 22:25:44.143619",
        },
    "test_probs[params1]": {
                "result": '[{"00": 1020, "01": 4}]',
                "timestamp_completed": "2026-08-24 22:28:44.018090",
                "timestamp_scheduled": "2026-08-24 22:25:47.341915",
                "timestamp_submitted": "2026-08-24 22:25:44.143619",
            },
    "test_multiple_expvals[list_obs0-params0]": {
                    'result': '[{"0000": 777, "0001": 3, "0010": 209, "0011": 35}, {"0000": 472, "0001": 24, "0010": 493, "0011": 31, "1000": 4}, {"0000": 806, "0001": 5, "0010": 192, "0011": 21}, {"0000": 815, "0001": 12, "0010": 177, "0011": 19, "1000": 1}]',
                    "timestamp_completed": "2026-08-24 22:28:44.018090",
                    "timestamp_scheduled": "2026-08-24 22:25:47.341915",
                    "timestamp_submitted": "2026-08-24 22:25:44.143619",
                },
}



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
