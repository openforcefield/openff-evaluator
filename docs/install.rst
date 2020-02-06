Installation
============

The OpenFF Evaluator is currently installable either through ``conda`` or directly from the source code. Whichever
route is chosen, it is recommended to install the framework within a conda environment and allow the conda
package manager to install the required and optional dependencies.

More information about conda and instructions to perform a lightweight miniconda installation `can be
found here <https://docs.conda.io/en/latest/miniconda.html>`_. It will be assumed that these have been
followed and conda is available on your machine.

Installation from Conda
-----------------------

To install the ``openff-evaluator`` from the ``omnia`` channel, simply run::

    conda install -c omnia openff-evaluator

Recommended Dependencies
------------------------

If you have access to the fantastic `OpenEye toolkit <https://docs.eyesopen.com/toolkits/python/index.html>`_
we recommend installing this to enable (among many other things) the use of the ``BuildDockedCoordinates``
protocol and faster conformer generation / AM1BCC partial charge calculations::

    conda install -c openeye openeye-toolkits

To parameterize systems with the Amber ``tleap`` tool using a ``TLeapForceFieldSource`` the ``ambertools``
package must be installed::

    conda install -c ambermd 'ambertools >=19.0'

Installation from Source
------------------------

To install the OpenFF Evaluator from source begin by cloning the repository from `github
<https://github.com/openforcefield/openff-evaluator>`_::

    git clone https://github.com/openforcefield/openff-evaluator.git
    cd openff-evaluator

Create a custom conda environment which contains the required dependencies and activate it::

    conda env create --name openff-evaluator --file devtools/conda-envs/test_env.yaml
    conda activate openff-evaluator

Finally, install the estimator itself::

    python setup.py develop

