.. propertyestimator documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================
Property Estimator
==================

The property estimator is a distributed framework from the `Open Forcefield Consortium <http://openforcefield.org>`_
for storing, manipulating, and computing measured physical properties from simulation data.

.. warning:: This framework is still in **pre-alpha** and under heavy development. Although all steps
             have been taken to ensure correctness of the code and the results it produces, the authors
             accept no liability for any incorrectness any bugs or unintended behaviour may cause.

The framework currently has built in support for estimating the following properties:

+-------------------------+---------------------------+---------------------------+
|| Physical Property      || Simulation               || Reweighting              |
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

========

Index
-----

**User Guide**

* :doc:`install`
* :doc:`gettingstarted`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: User Guide

  install
  gettingstarted

**Class Guide**

* :doc:`physicalproperties`
* :doc:`propertydatasets`

.. toctree::
  :maxdepth: 2
  :hidden:
  :caption: User Guide

  physicalproperties
  propertydatasets

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