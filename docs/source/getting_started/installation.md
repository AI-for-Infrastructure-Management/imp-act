# Installation


This project is built using Python >=3.9. We recommend using a virtual environment to install the required packages. The following instructions will guide you through the installation process.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;margin:0px auto;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-7rv2{border-color:inherit;font-family:"Lucida Console", Monaco, monospace !important;text-align:left;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-yla0{font-weight:bold;text-align:left;vertical-align:middle}
@media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;margin: auto 0px;}}</style>
<div class="tg-wrap"><table class="tg">
<tbody>
  <tr>
    <td class="tg-nrix" colspan="2" rowspan="2">Python:<br>&gt;=3.9, &lt;3.11</td>
    <td class="tg-uzvj" colspan="3">Virtual Environment</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Anaconda</td>
    <td class="tg-9wq8">venv</td>
    <td class="tg-9wq8">Poetry</td>
  </tr>
  <tr>
    <td class="tg-yla0" rowspan="2">Package<br>Manager</td>
    <td class="tg-9wq8">Poetry</td>
    <td class="tg-7rv2"># Option 1<br>conda env create -f conda_environment.yaml<br>conda activate imp-rl-competition-env<br>poetry install</td>
    <td class="tg-7rv2"># Option 2<br># &lt;create a virtual environment&gt;<br>poetry install</td>
    <td class="tg-7rv2"># Option 4<br>poetry install</td>
  </tr>
  <tr>
    <td class="tg-9wq8">pip</td>
    <td class="tg-7rv2" colspan="2"># Option 3<br># &lt;create a virtual environment&gt;<br>pip install -r requirements/requirements.txt<br>pip install -e .</td>
    <td class="tg-9wq8">X</td>
  </tr>
</tbody>
</table></div>

* **Option 1 (recommended)**: Using [Anaconda](https://www.anaconda.com/download#downloads) and [Poetry](https://python-poetry.org/docs/#installation) (v1.7.1)

Clone the repository, create conda environment (uses Python 3.9) and install Poetry to install other dependencies,

```bash
git clone ... && cd <dir-name>
conda env create -f conda_environment.yaml
conda activate imp-rl-competition-env
poetry install
```

* **Option 2**: On any virtual environment and use Poetry,

```bash
# <create a virtual environment>
poetry install
```

* **Option 3**: Using [Poetry](https://python-poetry.org/docs/#installation) (v1.7.1),

``` bash
poetry  
```




(Optional) Verify your installation by running tests using this terminal command:

```bash
pytest
```
If all tests pass, you are ready to go!
