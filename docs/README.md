# Notes

- [Official Sphinx documentation](https://www.sphinx-doc.org/en/master/tutorial/index.html)

- [MyST - Markedly Structured Text - Parser](https://myst-parser.readthedocs.io/en/latest/)

## Getting started

1. Make sure you have the developer version of the package installed with `poetry install --with dev,vis,docs`. This installs the additional dependencies necessary to build the documentation.

2. Make changes to the documentation.

3. Build the docs locally,

```bash
cd docs
make html 
```

Once completed successfully, the html version of the documentation should be built under `docs/_build/html/`. 

4. Open `docs/_build/html/index.html` in your browser, to open the home page of the documentation.

5. You have to rebuild the html each time you make changes. To see latest changes, just refresh the pages. Sometimes new content does not appear or work as expected, in that case, you can just delete the `_build` folder and build again.

## Structure

The skeleton that outlines the main idea of the documentation is in [outline.md](./outline.md). The goal is to write the idea before writing the documentation, to ensure it is coherent, and easy to follow. Once that is sketched, it is easier to write down the documentation. 

## Syntax

We are using [MyST - Markedly Structured Text - Parser](https://myst-parser.readthedocs.io/en/latest/)

> A Sphinx and Docutils extension to parse MyST, a rich and extensible flavour of Markdown for authoring technical and scientific documentation.

It is pretty similar to Markdown but has more features [see](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html).


## Extensions

sphinx provides many nice extensions which can be added to [conf.py](./source/conf.py). If you do so, make sure you add it to the pyproject.toml and update the lock file with relaxed dependencies!