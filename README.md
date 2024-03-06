## Installation

Option 1 : [Anaconda](https://www.anaconda.com/download#downloads) and [Poetry](https://python-poetry.org/docs/#installation) (v1.7.1),
```bash
conda env create -f conda_environment.yaml
conda activate imp-rl-competition-env
poetry install --with dev,vis
```

Option 2: pip on any virtual environment,
```bash
pip install -r requirements/requirements.txt
pip install -e .
```

(Optional) Verify your installation by running tests using this terminal command:

```bash
pytest
```

## Contribution guidelines
Guidelines are outlined in the [CONTRIBUTING README](CONTRIBUTING.md).
