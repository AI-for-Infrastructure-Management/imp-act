# Installation

This project is built using Python >=3.9. We recommend using a virtual environment to install the required packages. **Part A** describes installing the environment and **Part B** details how additional packages (RL libraries etc.) can be installed.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-8bgf{border-color:inherit;font-style:italic;text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
@media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;margin: auto 0px;}}</style>
<div class="tg-wrap"><table class="tg">
<tbody>
  <tr>
    <td class="tg-c3ow" colspan="2" rowspan="2"><span style="font-weight:400;font-style:normal;text-decoration:none">Python:</span><br><br><span style="font-weight:400;font-style:normal;text-decoration:none">&gt;=3.9, &lt;3.11</span></td>
    <td class="tg-7btt" colspan="3">Virtual Environment</td>
  </tr>
  <tr>
    <td class="tg-8bgf">Anaconda</td>
    <td class="tg-8bgf">venv</td>
    <td class="tg-8bgf">Poetry</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="2"><span style="font-weight:bold">Package</span><br><span style="font-weight:bold">Manager</span></td>
    <td class="tg-8bgf">Poetry</td>
    <td class="tg-7btt"># Option 1</td>
    <td class="tg-c3ow"># Option 2</td>
    <td class="tg-c3ow"># Option 4</td>
  </tr>
  <tr>
    <td class="tg-8bgf">pip</td>
    <td class="tg-c3ow" colspan="2"># Option 3</td>
    <td class="tg-c3ow">X</td>
  </tr>
</tbody>
</table></div>

## Part A: Installing the environment

* **Option 1 (recommended)**: Using [Anaconda](https://www.anaconda.com/download#downloads) and [Poetry](https://python-poetry.org/docs/#installation) (v1.7.1),

Clone the repository, create conda environment (uses Python 3.9) and install Poetry to install other dependencies,

```bash
git clone ... && cd <dir-name>
conda env create -f conda_environment.yaml
conda activate imp-act-challenge-env
poetry install
```

* **Option 2**: On any virtual environment and [Poetry](https://python-poetry.org/docs/#installation) (v1.7.1),

```bash
# <create a virtual environment>
poetry install
```

* **Option 3**: On any virtual environment and [pip](https://pypi.org/project/pip/),

```bash
# <create a virtual environment>
pip install -r requirements/requirements.txt
pip install -e .
```

* **Option 4**: Using [Poetry](https://python-poetry.org/docs/#installation) (v1.7.1),

``` bash
poetry  
```


(Optional) Verify your installation by running tests using this terminal command:

```bash
pytest
```
If all tests pass, you are ready to go!


## Part B: Adding custom packages 

You will most likely need to install additional packages to solve the environment. To minimize dependency conflicts, `pyproject.toml` specifies minimum dependencies and relaxed version specifications. 

* **Option 1 (recommended)**: [Poetry](https://python-poetry.org/docs/#installation) (v1.7.1)

Poetry provides a elegant way to resolve and install dependencies via `poetry add`. The add command adds required packages to your `pyproject.toml` and installs them.

For example,

```bash
poetry add pytorch
```

If you do not specify a version constraint, poetry will choose a suitable one based on the available package versions. Version constraints can also be easily specified ([official documentation: `poetry add`](https://python-poetry.org/docs/cli/#add))


* **Option 2**: [pip](https://pypi.org/project/pip/),

You can specify additional dependencies in `requirements.txt`