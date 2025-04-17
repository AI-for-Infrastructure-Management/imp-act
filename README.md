## Installation

Prerequisites:
* Python >=3.7,<3.11 (note that you will need Python < 3.10 to run PyMARL or EPyMARL)
* [Poetry 1.7.1+](https://python-poetry.org/docs/#installation)

Installation via *Poetry*
```bash
poetry install --with dev,vis,jax
```

Installation via *pip requirements*
```bash
pip install -r requirements/requirements.txt
pip install -e .
```

### Conda virtual environment (optional)
You can install a [conda](https://www.anaconda.com/download#downloads) virtual environment as:
```bash
conda env create -f conda_environment.yaml
conda activate impact-env
```

### Pytests (Optional)
Verify your installation by running tests using this terminal command:
```bash
pytest
```

## Contribution guidelines
Guidelines are outlined in the [contribution file](CONTRIBUTING.md).
