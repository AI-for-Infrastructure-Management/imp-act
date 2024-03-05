Installation
============

Basic Installation
------------------

This project is built using Python 3.10.0. We recommend using a virtual environment to install the required packages. The following instructions will guide you through the installation process.

* **Option 1 (recommended)**: `Anaconda <https://www.anaconda.com/download#downloads>`_ and `Poetry <https://python-poetry.org/docs/#installation>`_ (v1.7.1)


.. code-block:: bash

    git clone ... && cd <dir-name>
    conda env create -f conda_environment.yaml
    conda activate imp-rl-competition-env
    poetry install


* **Option 2**: pip on any virtual environment,


.. code-block:: bash

    pip install -r requirements/requirements.txt
    pip install -e .


(Optional) Verify your installation by running tests using this terminal command:

.. code-block:: bash

    pytest

If all tests pass, you are ready to go!




Development Installation
------------------------
