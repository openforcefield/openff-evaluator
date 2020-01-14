================
OpenFF Evaluator
================

*An automated and scalable framework for curating, manipulating, and computing data sets of physical properties
from molecular simulation and simulation data.*

The framework is built around three central tenets:

- **Flexibility:** New physical properties, data sources and calculation approaches are easily added via
  the extensible plugin system and flexible workflow engine.

- **Automation:** Physical property measurements are readily importable from open data source (such as
  ThermoML) through the data set API, and automatically calculated using the builtin workflow definitions
  with no user intervation.

- **Scalability:** Calculations are readily scalable from single machines and laptops up to large HPC clusters and
  supercomputers through seamless integration with libraries like `dask <https://distributed.dask.org/en/latest/>`_.

..  The framework is designed to be as flexible as possible, with a focus on users being able to readily plug-in the
    physical properties which are of interest to them, as well as different approach to estimate them, such as by
    direct simulation, by re-using cached simulation, or by evaluating regressed models trained upon such (see ...).
    The framework enables a high throughput of calculations thanks to a distributed architecture, whereby all requests
    for data sets to be estimated may be seamlessly submitted from local machines with modest hardware requirements to
    high performance compute clusters and supercomputers whereby compute backends such as `dask` may be used to
    distribute the calculations across many compute nodes (see ..).

..  Calculation Approaches
    ----------------------



..  Supported Physical Properties
    -----------------------------
    The framework currently has built-in support for evaluating a number of physical properties, ranging from
    relatively 'cheap' properties such as liquid densities, up to more computationally demanding properties such
    as solvation free energies and host-guest binding affinities.
    Included for most of these properties is the ability to calculate their derivates with respect to force field
    parameters, making the framework ideal for evaluating the objective function and it's gradient as part of a
    force field optimisation.
    The properties currently supported are:
    +-------------------------+---------------------------+---------------------------+
    || Physical Property      || Simulation Layer         || Reweighting Layer        |
    ||                        +--------------+------------+--------------+------------+
    ||                        || Implemented || Gradients || Implemented || Gradients |
    +=========================+==============+============+==============+============+
    || Density                || Yes         || Yes       || Yes         || Yes       |
    +-------------------------+--------------+------------+--------------+------------+
    || Dielectric Constatant  || Yes         || Yes*      || Yes         || Yes*      |
    +-------------------------+--------------+------------+--------------+------------+
    || H\ :sub:`vaporization` || Yes         || Yes       || Yes         || Yes*      |
    +-------------------------+--------------+------------+--------------+------------+
    || H\ :sub:`mixing`       || Yes         || Yes*      || Yes         || Yes*      |
    +-------------------------+--------------+------------+--------------+------------+
    || V\ :sub:`excess`       || Yes         || Yes*      || Yes         || Yes*      |
    +-------------------------+--------------+------------+--------------+------------+
    || G\ :sub:`solvation`    || Yes*        || No        || No          || No        |
    +-------------------------+--------------+------------+--------------+------------+
    \* Entries marked with an asterisk are implemented but have not yet been extensively tested and validated.

=====

Index
-----

**Getting Started**

* :doc:`install`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Getting Started

  install

**Tutorials**

* :doc:`tutorial01`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Tutorials

  tutorial01

**Data Set Documentation**

* :doc:`physicalproperties`
* :doc:`thermomldatasets`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Data Set Documentation

  physicalproperties
  thermomldatasets

**Layer Documentation**

* :doc:`reweightinglayer`
* :doc:`simulationlayer`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Layer Documentation

  simulationlayer
  reweightinglayer

**Backend Documentation**

* :doc:`builddocs`
* :doc:`builddocs`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Backend Documentation

  calculationbackend
  storagebackend

**Developer Documentation**

* :doc:`builddocs`
* :doc:`api`
* :doc:`releasehistory`
* :doc:`releaseprocess`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Developer Documentation

  builddocs
  api
  releasehistory
  releaseprocess