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
git clone https://github.com/Munich-Quantum-Software-Stack/MQP-Qiskit-Provider.git
cd MQP-Qiskit-Provider
```

**Create a virtual environment and install the dependencies:**

```sh
pdm install
```

## Running Tests

To run the tests, use `pytest`:

```sh
pdm run pytest
```

## Publishing Documentation on GitHub Pages

To publish the documentation on GitHub Pages, follow these steps:

**Install MkDocs and the Material theme:**

```sh
pdm install -G docs
```

**Build the documentation:**

```sh
pdm run mkdocs build
```

**View documentation locally**

Run the following and browse the documentation locally at: [http://localhost:8000](http://localhost:8000)

```sh
pdm run mkdocs serve
```

**Deploy the documentation to GitHub Pages:**

```sh
pdm run mkdocs gh-deploy --remote-name git@github.com:Munich-Quantum-Software-Stack/MQP-Qiskit-Provider-Documentation.git --remote-branch gh-pages
```

This will create a new branch named `gh-pages` in your repository and deploy the documentation to GitHub Pages.