================
OpenFF Evaluator
================

.. role:: green-font
.. role:: red-font
.. role:: ignore-width

.. |tick|    replace:: :green-font:`✓`
.. |cross|   replace:: :red-font:`✕`
.. |delta|   unicode:: U+0394
.. |ast|     replace:: :ignore-width:`*`

*An automated and scalable framework for curating, manipulating, and computing data sets of physical properties
from molecular simulation and simulation data.*

The framework is built around four central ideas:

.. rst-class:: spaced-list

    - **Flexibility:** New physical properties, data sources and calculation approaches are easily added via
      the extensible plugin system and flexible workflow engine.

    - **Automation:** Physical property measurements are readily importable from open data sources (such as the
      `NIST ThermoML Archive <http://trc.nist.gov/ThermoML.html>`_) through the data set API, and automatically
      calculated using the built-in or user specified calculation workflows.

    - **Scalability:** Calculations are readily scalable from single machines and laptops up to large HPC clusters and
      supercomputers through seamless integration with libraries such as `dask <https://distributed.dask.org/en/
      latest/>`_.

    - **Efficiency:** Properties will automatically be estimated using the fastest approach possible, whether that be
      through evaluated a trained surrogate model, re-evaluating cached simulation data, or by running simulations
      directly.

Calculation Approaches
----------------------

Supported Physical Properties
-----------------------------
The framework has built-in support for evaluating a number of physical properties, ranging from relatively
'cheap' properties such as liquid densities, up to more computationally demanding properties such as solvation
free energies and host-guest binding affinities.

Included for most of these properties is the ability to calculate their derivatives with respect to force field
parameters, making the framework ideal for evaluating the objective function and it's gradient as part of a force
field optimisation.

.. table::
   :widths: auto
   :align: center
   :class: property-table

   +----------------------------------+---------------------------------+--------------------------------+
   || Physical Property               || Direct Simulation              || Reweighting Cached Data       |
   ||                                 +----------------+----------------+--------------+-----------------+
   ||                                 || Supported     || Gradients     || Supported   || Gradients      |
   +==================================+================+================+==============+=================+
   || Density                         || |tick|        || |tick|        || |tick|      || |tick|         |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || Dielectric Constant             || |tick|\ |ast| || |tick|        || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ H\ :sub:`vaporization` || |tick|        || |tick|        || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ H\ :sub:`mixing`       || |tick|        || |tick|\ |ast| || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ V\ :sub:`excess`       || |tick|        || |tick|\ |ast| || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ G\ :sub:`solvation`    || |tick|\ |ast| || |cross|       || |cross|     || |cross|        |
   +----------------------------------+----------------+----------------+--------------+-----------------+

*\* Entries marked with an asterisk are supported but have not yet been extensively tested and validated.*

.. Setup the side-pane table of contents.

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Getting Started

  install

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Tutorials

  tutorial01

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Data Set Documentation

  physicalproperties
  thermomldatasets

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Layer Documentation

  simulationlayer
  reweightinglayer

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Backend Documentation

  calculationbackend
  storagebackend

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Developer Documentation

  builddocs
  api
  releasehistory
  releaseprocess