"""Nox sessions used locally and by the CI workflow.

The CI calls these sessions through the reusable Python workflows of
munich-quantum-toolkit/workflows (the same templates QDMI uses).

Run them locally with:

    uvx nox -s tests          # mock tests on every supported Python
"""

from __future__ import annotations

import os

import nox

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

PYTHON_ALL_VERSIONS = ["3.11", "3.12", "3.13"]

# Only the mocked tests run in CI; the `live` tests need an MQSS token and backend.
PYTEST_ARGS = ["-m", "mock"]


def _run_tests(session: nox.Session) -> None:
    session.install("-e", ".", "pytest", "pytest-cov")
    session.run(
        "pytest",
        *PYTEST_ARGS,
        *session.posargs,
        env={"MQSS_TOKEN": os.environ.get("MQSS_TOKEN", "")},
    )


@nox.session(python=PYTHON_ALL_VERSIONS, reuse_venv=True, default=True)
def tests(session: nox.Session) -> None:
    """Run the mocked test suite."""
    _run_tests(session)
