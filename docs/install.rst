Install Property Estimator
==========================

The **Property Estimator** is currently only installable from source,
although other installation routes (such as conda) will be available
in the near future.

Install from Source
-------------------

The most convenient and cleanest way to install the **Property Estimator** and
all of it's dependencies is via a custom *Conda* environment. Instructions to
perform a very basic Miniconda installation `can be found here
<https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>`_.

The *Property Estimator* can be installed by doing the following:

1. First clone the source code from the GitHub repository ::

    git clone https://github.com/openforcefield/propertyestimator.git
    cd propertyestimator

2. Create a new conda environment called *propetyestimator* and populate it with
   all of the required dependencies ::

    conda env create --name propertyestimator --file devtools/conda-envs/test_env.yaml

3. Activate the new environment ::

    conda activate propertyestimator

4. Install the estimator itself ::

    python setup.py develop

Building the Documentation Locally
----------------------------------

As well as being available on `read the docs
<https://property-estimator.readthedocs.io/en/latest/>`_, the
**Property Estimator** documentation can be built on a local machine:

1. First clone the source code from the GitHub repository ::

    git clone https://github.com/openforcefield/propertyestimator.git
    cd propertyestimator/docs

2. Create a conda environment which contains the docs dependencies ::

    conda env create --name propertyestimator-docs --file docs/environment.yml

3. Activate the new environment ::

    conda activate propertyestimator-docs

4. Clean up any cached files ::

    rm -rf api && make clean

5. Build the docs ::

    make html

This will create a new directory called ``_build`` which contains a ``html`` folder
which contains the built documentation if no errors occurred.