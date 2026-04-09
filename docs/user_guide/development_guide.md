# Development Guide

## Project Overview

This project provides a Pennylane Adapter for MQSS, allowing users to access MQSS backends through Pennylane. The main components of the project are:

- `device.py`: Contains the [`MQSSPennylaneDevice`](../api/MQSSPennylaneDevice.md) class, which serves as the main entry point for defining MQSS devices in Pennylane.
- `backend.py`: Contains the [`MQSSPennylaneBackend`](../api/MQSSPennylaneBackend.md) class, which interfaces with the MQSS backends.
- `job.py`: Contains the [`MQPJob`](../api/MQPJob.md) class, which handles job cancellation, status checking, and result retrieval.

## Prerequisites

Before you start developing, ensure you have the following installed:

- Python >=3.11
- [`uv`](https://docs.astral.sh/uv/) package manager for python

## Setting Up the Development Environment

**Clone the repository:**

```sh
git clone git@github.com:Munich-Quantum-Software-Stack/MQSS-Pennylane-Adapter.git
cd MQSS-Pennylane-Adapter
```

**Create a virtual environment and install the dependencies:**

You can create the virtual environment through `uv sync`. `--all-groups` option lets you install development dependencies as well.
```sh
uv sync --all-groups
```

## Running Tests

To run the tests, use `pytest` through `uv`:

```sh
uv run pytest
```
some useful flags are:
```sh
uv run pytest -s # Shows print statements in the terminal directly.
uv run pytest -m # Runs only the mock tests 
uv run pytest --maxfail=<N> # Allows N failures, can be useful for live testing.
uv run pytest --pdb # Goes into debug mode once a test fails.
```
## Publishing Documentation on GitHub Pages and Viewing it Locally

The documentation is published to GitHub Pages, after every succesful PR merged to main. To update it locally at [http://localhost:8000](http://localhost:8000), you can run:


```sh
uv run mkdocs build
uv run mkdocs serve
```