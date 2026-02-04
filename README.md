# Sentinel

Bayesian tensor-decomposition model for epidemiological surveillance data,
demonstrated on [Project Tycho](https://www.tycho.pitt.edu/) weekly case counts.
The model decomposes disease incidence into location, type, and temporal effects
using Gaussian process priors with FFT-accelerated inference via NumPyro.

## Install

```bash
pip install -e ".[dev]"
```

## Data

Download and preprocess the Project Tycho dataset:

```bash
make data
```

## Fit

```bash
python -m sentinel.fit --config configs/default.py --data data/project_tycho_processed_cases.csv output/
```

## Test

```bash
pytest
```
