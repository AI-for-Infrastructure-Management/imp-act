## Installation

With [Anaconda](https://www.anaconda.com/download#downloads) and [poetry](https://python-poetry.org/docs/#installation) (1.7 used):
```bash
conda env create -f conda_environment.yaml
conda activate imp-rl-competition-env
poetry install
```

With pip on any virtual environment:
```bash
pip install -r requirements/requirements.txt
pip install -e .
```

## Running tests
```bash
pytest
```

## Checkout contribution guidelines
Guidelines are available in [CONTRIBUTING README](CONTRIBUTING.md).
