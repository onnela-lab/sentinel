# Sentinel [![Sentinel](https://github.com/onnela-lab/sentinel/actions/workflows/build.yml/badge.svg)](https://github.com/onnela-lab/sentinel/actions/workflows/build.yml)

Bayesian tensor-decomposition model for epidemiological surveillance data,
demonstrated on [Project Tycho](https://www.tycho.pitt.edu/) weekly case counts.
The model decomposes disease incidence into location, type, and temporal effects
using Gaussian process priors with FFT-accelerated inference via NumPyro.

## Installation

Make sure you have [`uv`](https://docs.astral.sh/uv/) installed and execute the following commands to install dependencies and the package.

```bash
uv sync --all-groups
uv pip install -e .
```

Verify the installation by running the tests.

```bash
uv run pytest tests/
```

## Usage

Run the full analysis pipeline (download data, fit model, generate results):

```bash
make all
```

Or run individual steps:

```bash
make data      # Download and preprocess Project Tycho dataset
make fit       # Fit the model (uses configs/helmert.py by default)
make results   # Generate visualization notebook
```

Configuration and output paths can be customized:

```bash
make fit CONFIG=configs/default.py OUTPUT=workspace/default
```
