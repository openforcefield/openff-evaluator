Building the Docs
=================

Although documentation for the OpenFF Evaluator is `readily available online
<https://openff-evaluator.readthedocs.io/en/latest/>`_, it is sometimes useful
to build a local version such as when

- developing new pages which you wish to preview without having to wait
  for ReadTheDocs to finish building.

- debugging errors which occur when building on ReadTheDocs.

In these cases, the docs can be built locally by doing the following::

    git clone https://github.com/openforcefield/openff-evaluator.git
    cd openff-evaluator
    conda env create --name openff-evaluator-docs --file devtools/conda-envs/docs_env.yaml
    conda activate openff-evaluator-docs
    rm -rf docs/api docs/_build/html && sphinx-build -b html -j auto docs docs/_build/html

The above will yield a new directory named `docs/_build` which will contain the
built html files which can be viewed in your local browser.
