Installing the Property Estimator
=================================

The Property Estimator is currently only installable from source, although other installation routes
(such as conda) will be available in the near future. Whichever route is chosen, it is recommended to
install the estimator within a conda environment, and allow the conda package manager to install the
required dependencies.

More information about conda, and instructions to perform a lightweight miniconda installation `can be
found here <https://docs.conda.io/en/latest/miniconda.html>`_. It will be assumed that these have been
followed and conda is available on your machine.

Installation from Source
------------------------

To install Property Estimator from source, clone the repository from `github
<https://github.com/openforcefield/propertyestimator>`_::

    git clone https://github.com/openforcefield/propertyestimator.git
    cd propertyestimator

Create a custom conda environment which contains *most* of the required dependencies and activate it::

    conda env create --name propertyestimator --file devtools/conda-envs/test_env.yaml
    conda activate propertyestimator

As the estimator is still under heavy development and relies on a handful of 'in development' features of other
projects, a number of dependencies need to be directly installed from development branches hosted on github::

    conda install --yes --only-deps openforcefield
    pip install git+https://github.com/openforcefield/openforcefield.git@0.3.0-branch

    conda install --yes --only-deps yank
    pip install git+https://github.com/choderalab/yank.git@trailblaze_checkpoint

*Note - This step is only temporary, and will not be required in future releases.*

The final step is to install the estimator itself::

    python setup.py develop

And that's it!