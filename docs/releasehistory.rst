Release History
===============

Releases follow the ``major.minor.micro`` scheme recommended by
`PEP440 <https://www.python.org/dev/peps/pep-0440/#final-releases>`_, where

* ``major`` increments denote a change that may break API compatibility with previous ``major`` releases
* ``minor`` increments add features but do not break API compatibility
* ``micro`` increments represent bugfix releases or improvements in documentation


0.3.0
-----

The main feature of this release is the overhauling of how the framework computes the gradients of observables with
respect to force field parameters.

In particular, from this release onwards all gradients will be computed using the fluctuation formula (also referred
to as the thermodynamic gradient), rather than calculation be the re-weighted finite difference approach (PR
`#280 <https://github.com/openforcefield/openff-evaluator/pull/280>`_). In general the two methods produce gradients
which are numerically indistinguishable, and so this should not markedly change any scientific output of this framework.

The change was made to, in future, enable better integration with automatic differentiation libraries such as
`jax <https://github.com/google/jax>`_, and differentiable simulation engines such as
`timemachine <https://github.com/proteneer/timemachine>`_ which readily and rapdily give access to
:math:`\mathrm{d} U / \mathrm{d} \theta_i`.

Bugfixes
""""""""

* PR `#285 <https://github.com/openforcefield/openff-evaluator/pull/285>`_: Use merged protocols in workflow provenance.
* PR `#287 <https://github.com/openforcefield/openff-evaluator/pull/287>`_: Fix merging of nested protocol inputs

New Features
""""""""""""

* PR `#280 <https://github.com/openforcefield/openff-evaluator/pull/280>`_: Switch to computing thermodynamic gradients.
* PR `#309 <https://github.com/openforcefield/openff-evaluator/pull/309>`_: Add a date to the timestamp logging output.

Breaking Changes
""""""""""""""""

* The ``StatisticsArray`` array has been completely removed and replaced with a new set of observable (``Observable``, ``ObservableArray``, ``ObservableFrame`` objects (`#279 <https://github.com/openforcefield/openff-evaluator/pull/279>`_, `#286 <https://github.com/openforcefield/openff-evaluator/pull/279>`_).

* The following classes have been renamed:

    - ``OpenMMReducedPotentials`` -> ``OpenMMEvaluateEnergies``.
    - ``AveragePropertyProtocol`` -> ``BaseAverageObservable``, ``ExtractAverageStatistic`` -> ``AverageObservable``, ``ExtractUncorrelatedData`` -> ``BaseDecorrelateProtocol``, ``ExtractUncorrelatedTrajectoryData`` -> ``DecorrelateTrajectory``, ``ExtractUncorrelatedStatisticsData`` -> ``DecorrelateObservables``
    - ``ConcatenateStatistics`` -> ``ConcatenateObservables``, ``BaseReducedPotentials`` -> ``BaseEvaluateEnergies``, ``ReweightStatistics -> ReweightObservable``

* The following classes have been removed:

    - ``OpenMMGradientPotentials``, ``BaseGradientPotentials``, ``CentralDifferenceGradient``

* The final value estimated by a workflow must now be an ``Observable`` object which contains any gradient information to return. (`#296 <https://github.com/openforcefield/openff-evaluator/pull/296>`_).

0.2.2
-----

This release adds documentation for how physical properties are computed within the framework (both for this, and for
previous releases.

Documentation
"""""""""""""

* PR `#281 <https://github.com/openforcefield/openff-evaluator/pull/281>`_: Initial pass at physical property documentation.


0.2.1
-----

A patch release offering minor bug fixes and quality of life improvements.

Bugfixes
""""""""

* PR `#259 <https://github.com/openforcefield/propertyestimator/pull/259>`_: Adds ``is_file_and_not_empty`` and addresses OpenMM failure modes.
* PR `#275 <https://github.com/openforcefield/propertyestimator/pull/275>`_: Workaround for N substance molecules > user specified maximum.

New Features
""""""""""""

* PR `#267 <https://github.com/openforcefield/propertyestimator/pull/267>`_: Adds workflow protocol to Boltzmann average free energies.
* PR `#269 <https://github.com/openforcefield/propertyestimator/pull/269>`_: Expose exclude exact amount from max molecule cap.

0.2.0
-----

This release overhauls the frameworks data curation abilities. In particular, it adds

* a significant amount of data filters, including to filter by state, substance composition and chemical
  functionalities.

and components to

* easily import all of the ThermoML and FreeSolv archives.
* convert between property types (currently density <-> excess molar volume).
* select data points close to a set of target states, and substances which contain specific functionalities (i.e.
  select only data points measured for ketones, alcohols or alkanes).

More information about the new curation abilities can be found :ref:`in the documentation here <datasets/curation:Data Set Curation>`.

New Features
""""""""""""

* PR `#260 <https://github.com/openforcefield/propertyestimator/pull/260>`_: Data set curation overhaul.
* PR `#261 <https://github.com/openforcefield/propertyestimator/pull/261>`_: Adds ``PhysicalPropertyDataSet.from_pandas``.

Breaking Changes
""""""""""""""""

* All of the ``PhysicalPropertyDataSet.filter_by_XXX`` functions have now been removed in favor of the new curation
  components. See the :ref:`documentation <datasets/curation:Examples>` for information about the newly available
  filters and more.

0.1.2
-----

A patch release offering minor bug fixes and quality of life improvements. 

Bugfixes
""""""""

* PR `#254 <https://github.com/openforcefield/propertyestimator/pull/254>`_: Fix incompatible protocols being merged due to an id replacement bug.
* PR `#255 <https://github.com/openforcefield/propertyestimator/pull/255>`_: Fix recursive ``ThermodynamicState`` string representation.
* PR `#256 <https://github.com/openforcefield/propertyestimator/pull/256>`_: Fix incorrect version when installing from tarballs.

0.1.1
-----

A patch release offering minor bug fixes and quality of life improvements.

Bugfixes
""""""""

* PR `#249 <https://github.com/openforcefield/propertyestimator/pull/249>`_: Fix replacing protocols of non-existent workflow schema.
* PR `#253 <https://github.com/openforcefield/propertyestimator/pull/253>`_: Fix `antechamber` truncating charge file.

Documentation
"""""""""""""

* PR `#252 <https://github.com/openforcefield/propertyestimator/pull/252>`_: Use `conda-forge` for `ambertools` installation.

0.1.0 - OpenFF Evaluator
------------------------

Introducing the OpenFF Evaluator! The release marks a significant
milestone in the development of this project, and constitutes an almost
full redesign of the framework with a focus on stability and ease of
use.

**Note:** *because of the extensive changes made throughout the entire
framework, this release should almost be considered as an entirely new
package. No files produced by previous versions of this will work with
this new release.*

Clearer Branding
""""""""""""""""

First and foremost, this release marks the complete rebranding from the
previously named *propertyestimator* to the new *openff-evaluator*
package. This change is accompanied by the introduction of a new
``openff`` namespace for the package, signifying it's position in the
larger Open Force Field infrastructure and piplelines.

What was previously::

   import propertyestimator

now becomes::

   import openff.evaluator

The rebranded package is now shipped on ``conda`` under the new name of
``openff-evaluator``::

   conda install -c conda-forge -c omnia openff-evaluator

Markedly Improved Documentation
"""""""""""""""""""""""""""""""

In addition, the release includes for the first time a significant
amount of documentation for using the `framework and it's features`_ as
well as a collection of user focused tutorials which can be ran directly
in the browser.

Support for RDKit
"""""""""""""""""

This release almost entirely removes the dependence on OpenEye thanks to
support for RDKit almost universally across the framework.

The only remaining instance where OpenEye is still required is for host-guest
binding affinity calculations where it is used to perform docking.

Model Validation
""""""""""""""""

Starting with this release almost all models, range from
``PhysicalProperty`` entries to ``ProtocolSchema`` objects, are now
heavily validated to help catch any typos or errors early on.

Batching of Similar Properties
""""""""""""""""""""""""""""""

The ``EvaluatorServer`` now more intelligently attempts to batch
properties which may be computed using the same simulations into a
single batch to be estimated. While the behaviour was already supported
for pure properties in previous, this has now been significantly
expanded to work well with mixture properties.

0.0.9 - Multi-state Reweighting Fix
-----------------------------------

This release implements a fix for calculating the gradients of properties being estimated by reweighting data cached from multiple independant simulations.

Bugfixes
""""""""

* PR `#143 <https://github.com/openforcefield/propertyestimator/pull/143>`_: Fix for multi-state gradient calculations.


0.0.8 - ThermoML Improvements
-----------------------------

This release is centered around cleaning up the ThermoML data set utilities. The main change is that ThermoML archive files can now be loaded even if they don't contain measurement uncertainties.

New Features
""""""""""""

* PR `#142 <https://github.com/openforcefield/propertyestimator/pull/142>`_: ThermoML archives without uncertainties can now be loaded.

Breaking Changes
""""""""""""""""

* PR `#142 <https://github.com/openforcefield/propertyestimator/pull/142>`_: All `ThermoMLXXX` classes other than `ThermoMLDataSet` are now private.


0.0.7 - Bug Quick Fixes
-----------------------

This release aims to fix a number of minor bugs.

Bugfixes
""""""""

* PR `#136 <https://github.com/openforcefield/propertyestimator/pull/136>`_: Fix for comparing thermodynamic states with unset pressures.
* PR `#138 <https://github.com/openforcefield/propertyestimator/pull/138>`_: Fix for a typo in the maximum number of minimization iterations.


0.0.6 - Solvation Free Energies
-------------------------------

This release centers around two key changes - 

i) a general refactoring of the protocol classes to be much cleaner and extensible through the removal of the old stub functions and the addition of cleaner descriptors.
ii) the addition of workflows to estimate solvation free energies via the new ``SolvationYankProtocol`` and ``SolvationFreeEnergy`` classes.

The implemented free energy workflow is still rather basic, and does not yet support calculating parameter gradients or estimation from cached simulation data through reweighting. 

A new table has been added to the documentation to make clear which built-in properties support which features.

New Features
""""""""""""

* PR `#110 <https://github.com/openforcefield/propertyestimator/pull/110>`_: Cleanup and refactor of protocol classes.
* PR `#125 <https://github.com/openforcefield/propertyestimator/pull/125>`_: Support for PBS based HPC clusters.
* PR `#127 <https://github.com/openforcefield/propertyestimator/pull/127>`_: Adds a basic workflow for estimating solvation free energies with `YANK <http://getyank.org/latest/>`_.
* PR `#130 <https://github.com/openforcefield/propertyestimator/pull/130>`_: Adds a cleaner mechanism for restarting simulations from checkpoints.
* PR `#134 <https://github.com/openforcefield/propertyestimator/pull/134>`_: Update to a more stable dask version.

Bugfixes
""""""""

* PR `#128 <https://github.com/openforcefield/propertyestimator/pull/128>`_: Removed the defunct dask backend `processes` kwarg.
* PR `#133 <https://github.com/openforcefield/propertyestimator/pull/133>`_: Fix for tests failing on MacOS due to `travis` issues.


Breaking Changes
""""""""""""""""

* PR `#130 <https://github.com/openforcefield/propertyestimator/pull/130>`_: The ``RunOpenMMSimulation.steps`` input has now been split into the ``steps_per_iteration`` and ``total_number_of_iterations`` inputs.

Migration Guide
"""""""""""""""

This release contained several public API breaking changes. For the most part, these can be
remedied by the follow steps:

* Replace all instances of ``run_openmm_simulation_protocol.steps`` to ``run_openmm_simulation_protocol.steps_per_iteration``


0.0.5 - Fix For Merging of Estimation Requests
----------------------------------------------

This release implements a fix for a major bug which caused incorrect results to be returned when submitting multiple estimation requests at the same time - namely, the returned results became jumbled between the different requests. As an example, if a request was made to estimate a data set using the `smirnoff99frosst` force field, and then straight after with the `gaff 1.81` force field, the results of the `smirnoff99frosst` request may contain some properties estimated with `gaff 1.81` and vice versa.

This issue does not affect cases where only a single request was made and completed at a time (i.e the results of the previous request completed before the next estimation request was made).

Bugfixes
""""""""

* PR `#119 <https://github.com/openforcefield/propertyestimator/pull/119>`_: Fixes gather task merging.
* PR `#121 <https://github.com/openforcefield/propertyestimator/pull/121>`_: Update to distributed 2.5.1.


0.0.4 - Initial Support for Non-SMIRNOFF FFs
--------------------------------------------

This release adds initial support for estimating property data sets using force fields
not based on the ``SMIRNOFF`` specification. In particular, initial AMBER force field support
has been added, along with a protocol which applies said force fields using ``tleap``.

New Features
""""""""""""

* PR `#96 <https://github.com/openforcefield/propertyestimator/pull/96>`_: Adds a mechanism for specifying force fields not in the ``SMIRNOFF`` spec.
* PR `#99 <https://github.com/openforcefield/propertyestimator/pull/99>`_: Adds support for applying ``AMBER`` force field parameters through ``tleap``
* PR `#111 <https://github.com/openforcefield/propertyestimator/pull/111>`_: Protocols now stream trajectories from disk, rather than pre-load the whole thing.
* PR `#112 <https://github.com/openforcefield/propertyestimator/pull/112>`_: Specific types of protocols can now be easily be replaced using ``WorkflowOptions``.
* PR `#117 <https://github.com/openforcefield/propertyestimator/pull/117>`_: Adds support for converting ``PhysicalPropertyDataSet`` objects to ``pandas.DataFrame``.

Bugfixes
""""""""

* PR `#115 <https://github.com/openforcefield/propertyestimator/pull/115>`_: Fixes caching data for substances whose smiles contain forward slashes.
* PR `#116 <https://github.com/openforcefield/propertyestimator/pull/116>`_: Fixes inconsistent mole fraction rounding.

Breaking Changes
""""""""""""""""

* PR `#96 <https://github.com/openforcefield/propertyestimator/pull/96>`_: The ``PropertyEstimatorClient.request_estimate(force_field=...`` argument has been renamed to ``force_field_source``.

Migration Guide
"""""""""""""""

This release contained several public API breaking changes. For the most part, these can be
remedied by the follow steps:

* Change all instances of ``PropertyEstimatorClient.request_estimate(force_field=...)`` to ``PropertyEstimatorClient.request_estimate(force_field_source=...)``


0.0.3 - ExcessMolarVolume and Typing Improvements 
-------------------------------------------------

This release implements a number of bug fixes and adds two key new features, namely built in support
for estimating excess molar volume measurements, and improved type checking for protocol inputs 
and outputs.

New Features
""""""""""""

* PR `#98 <https://github.com/openforcefield/propertyestimator/pull/98>`_: ``Substance`` objects may now have components with multiple amount types.
* PR `#101 <https://github.com/openforcefield/propertyestimator/pull/101>`_: Added support for estimating ``ExcessMolarVolume`` measurements from simulations.
* PR `#104 <https://github.com/openforcefield/propertyestimator/pull/104>`_: ``typing.Union`` is now a valid type arguemt to ``protocol_output`` and ``protocol_input``.

Bugfixes
""""""""

* PR `#94 <https://github.com/openforcefield/propertyestimator/pull/94>`_: Fixes exception when testing equality of ``ProtocolPath`` objects.
* PR `#100 <https://github.com/openforcefield/propertyestimator/pull/100>`_: Fixes precision issues when ensuring mole fractions are `<= 1.0`.
* PR `#102 <https://github.com/openforcefield/propertyestimator/pull/102>`_: Fixes replicated input for children of replicated protocols.
* PR `#105 <https://github.com/openforcefield/propertyestimator/pull/105>`_: Fixes excess properties weighting by the wrong mole fractions.
* PR `#107 <https://github.com/openforcefield/propertyestimator/pull/107>`_: Fixes excess properties being converged to the wrong uncertainty.
* PR `#108 <https://github.com/openforcefield/propertyestimator/pull/108>`_: Fixes calculating MBAR gradients of reweighted properties.

Breaking Changes
""""""""""""""""

* PR `#98 <https://github.com/openforcefield/propertyestimator/pull/98>`_: ``Substance.get_amount`` renamed to ``Substance.get_amounts`` and now returns an
  immutable ``frozenset`` of ``Amount`` objects, rather than a single ``Amount``.
* PR `#104 <https://github.com/openforcefield/propertyestimator/pull/104>`_: The ``DivideGradientByScalar``, ``MultiplyGradientByScalar``, ``AddGradients``, ``SubtractGradients`` and
  ``WeightGradientByMoleFraction`` protocols have been removed. The ``WeightQuantityByMoleFraction`` protocol has been renamed
  to ``WeightByMoleFraction``.

Migration Guide
"""""""""""""""

This release contained several public API breaking changes. For the most part, these can be
remedied by the follow steps:

* Change all instances of ``Substance.get_amount`` to ``Substance.get_amounts`` and handle
  the newly returned frozenset of amounts, rather than the previously returned single amount.
* Replace the now removed protocols as follows:

  - ``DivideGradientByScalar`` -> ``DivideValue``
  - ``MultiplyGradientByScalar`` -> ``MultiplyValue``
  - ``AddGradients`` -> ``AddValues``
  - ``SubtractGradients`` -> ``SubtractValues``
  - ``WeightGradientByMoleFraction`` -> ``WeightByMoleFraction``
  - ``WeightQuantityByMoleFraction`` -> ``WeightByMoleFraction``


0.0.2 - Replicator Quick Fixes
------------------------------

A minor release to fix a number of minor bugs related to replicating protocols.

Bugfixes
""""""""

* PR `#90 <https://github.com/openforcefield/propertyestimator/pull/90>`_: Fixes merging gradient protocols with
  the same id.
* PR `#92 <https://github.com/openforcefield/propertyestimator/pull/92>`_: Fixes replicating protocols for more
  than 10 template values.
* PR `#93 <https://github.com/openforcefield/propertyestimator/pull/93>`_: Fixes ``ConditionalGroup`` objects losing
  their conditions input.

0.0.1 - Initial Release
-----------------------

The initial pre-alpha release of the framework.
