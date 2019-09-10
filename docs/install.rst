Installing the Property Estimator
=================================

The Property Estimator is currently installable both from source and through ``conda``. Whichever route
is chosen, it is recommended to install the estimator within a conda environment, and allow the conda
package manager to install the required dependencies.

More information about conda and instructions to perform a lightweight miniconda installation `can be
found here <https://docs.conda.io/en/latest/miniconda.html>`_. It will be assumed that these have been
followed and conda is available on your machine.

Installation from Conda
-----------------------

To install the ``propertyestimator`` from the ``omnia`` channel, simply run::

    conda install -c openeye -c omnia/label/rc propertyestimator

Optional Dependencies
---------------------

To parameterize systems with the Amber ``tleap`` tool using a ``TLeapForceFieldSource`` the ``ambertools19`` package must be installed::

    conda install -c ambermd 'ambertools ==19.0'

Installation from Source
------------------------

To install Property Estimator from source, clone the repository from `github
<https://github.com/openforcefield/propertyestimator>`_::

    git clone https://github.com/openforcefield/propertyestimator.git
    cd propertyestimator

Create a custom conda environment which contains the required dependencies and activate it::

    conda env create --name propertyestimator --file devtools/conda-envs/test_env.yaml
    conda activate propertyestimator

The final step is to install the estimator itself::

    python setup.py develop

And that's it!

