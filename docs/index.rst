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

.. |simulation|     replace:: :doc:`Direct Simulation <layers/simulationlayer>`
.. |reweighting|     replace:: :doc:`MBAR Reweighting <layers/reweightinglayer>`

*An automated and scalable framework for curating, manipulating, and computing data sets of physical properties
from molecular simulation and simulation data.*

The framework is built around four central ideas:

.. rst-class:: spaced-list

    - **Flexibility:** New physical properties, data sources and calculation approaches are easily added via
      an extensible plug-in system and a flexible workflow engine.

    - **Automation:** Physical property measurements are readily importable from open data sources (such as the
      `NIST ThermoML Archive <http://trc.nist.gov/ThermoML.html>`_) through the data set APIs, and automatically
      calculated using either the built-in or user specified calculation schemas.

    - **Scalability:** Calculations are readily scalable from single machines and laptops up to large HPC clusters and
      supercomputers through seamless integration with libraries such as `dask <https://distributed.dask.org/en/
      latest/>`_.

    - **Efficiency:** Properties are estimated using the fastest approach available to the framework, whether that
      be through evaluating a trained surrogate model, re-evaluating cached simulation data, or by running simulations
      directly.

Calculation Approaches
----------------------
The framework is designed around the idea of allowing multiple calculation approaches for estimating the same set
of properties, in addition to estimation directly from molecular simulation, all using a uniform API.

The primary purpose of this is to take advantage of the many techniques exist which are able to leverage data from
previous simulations to rapidly estimate sets of properties, such as `reweighting cached simulation data <http://
www.alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio>`_, or evaluating `surrogate models <https://pubs.acs
.org/doi/abs/10.1021/acs.jctc.8b00223>`_ trained upon cached data. The most rapid approach which may accurately
estimate a set of properties is automatically determined by the framework on the fly.

Each approach supported by the framework is implemented as a :doc:`calculation layer <layers/calculationlayers>`. Two
such layers are currently supported (although new calculation layers can be readily added via the plug-in system):

* evaluating physical properties directly from molecular simulation using the :doc:`SimulationLayer
  <layers/simulationlayer>`.
* reprocessing cached simulation data with `MBAR reweighting <http://www.alchemistry.org/wiki/
  Multistate_Bennett_Acceptance_Ratio>`_ using the :doc:`ReweightingLayer <layers/reweightinglayer>`.

Supported Physical Properties
-----------------------------
The framework has built-in support for evaluating a number of physical properties, ranging from relatively
'cheap' properties such as liquid densities, up to more computationally demanding properties such as solvation
free energies and host-guest binding affinities.

Included for most of these properties is the ability to calculate their derivatives with respect to force field
parameters, making the framework ideal for evaluating an objective function and it's gradient as part of a force
field optimisation.

.. table:: The physical properties which are natively supported by the framework.
   :widths: auto
   :align: center
   :class: clean-table property-table

   +----------------------------------+---------------------------------+--------------------------------+
   ||                                 || |simulation|                   || |reweighting|                 |
   ||                                 +----------------+----------------+--------------+-----------------+
   ||                                 || Supported     || Gradients     || Supported   || Gradients      |
   +==================================+================+================+==============+=================+
   || Density                         || |tick|        || |tick|        || |tick|      || |tick|         |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || Dielectric Constant             || |tick|        || |tick|        || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ H\ :sub:`vaporization` || |tick|        || |tick|        || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ H\ :sub:`mixing`       || |tick|        || |tick|        || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ V\ :sub:`excess`       || |tick|        || |tick|        || |tick|      || |tick|\ |ast|  |
   +----------------------------------+----------------+----------------+--------------+-----------------+
   || |delta|\ G\ :sub:`solvation`    || |tick|        || |cross|       || |cross|     || |cross|        |
   +----------------------------------+----------------+----------------+--------------+-----------------+

*\* Entries marked with an asterisk are supported but have not yet been extensively tested and validated.*

.. Setup the side-pane table of contents.

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Getting Started

  Overview <self>
  install
  gettingstarted/architecture
  gettingstarted/client
  gettingstarted/server

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Beginner Tutorials

  01 - Loading Data Sets <tutorials/tutorial01>
  02 - Estimating Data Sets <tutorials/tutorial02>
  03 - Analysing Data Sets <tutorials/tutorial03>

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Intermediate Tutorials

  04 - Optimizing Force Fields <tutorials/tutorial04>

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Data Sets

  datasets/physicalproperties
  datasets/thermomldatasets

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Calculation Layers

  Overview <layers/calculationlayers>
  Workflow Layers <layers/workflowlayer>
  Direct Simulation <layers/simulationlayer>
  MBAR Reweighting <layers/reweightinglayer>

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Workflows

  Overview <workflows/workflows>
  workflows/replicators
  workflows/workflowgraphs
  workflows/protocols
  workflows/protocolgroups

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Calculation Backends

  Overview <backends/calculationbackend>
  backends/daskbackends

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Storage Backends

  Overview <storage/storagebackend>
  storage/dataclasses
  Local File Backend <storage/localstorage>

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: Developer Documentation

  developer/builddocs
  api
  releasehistory
  developer/releaseprocess