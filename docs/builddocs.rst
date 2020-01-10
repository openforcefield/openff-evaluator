Building the Docs
=================

Although documentation for the Property Estimator is `readily available online
<https://property-estimator.readthedocs.io/en/latest/>`_, it is sometimes useful
to build a local version such as when

- developing new pages and you wish to preview those pages without having to wait
  for ReadTheDocs to finish building.

- debugging errors which occur when building on ReadTheDocs.

In these cases, the docs can be built locally by doing the following::

    git clone https://github.com/openforcefield/propertyestimator.git
    cd propertyestimator/docs
    conda env create --name propertyestimator-docs --file environment.yaml
    conda activate propertyestimator-docs
    rm -rf api && make clean && make html

The above will yield a new directory named `_build` which will contain the built
html files which can be viewed in your local browser.