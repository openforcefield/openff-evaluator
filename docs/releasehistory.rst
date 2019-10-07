Release History
===============

Releases will eventually follow the ``major.minor.micro`` scheme recommended by
`PEP440 <https://www.python.org/dev/peps/pep-0440/#final-releases>`_, where

* ``major`` increments denote a change that may break API compatibility with previous ``major`` releases
* ``minor`` increments add features but do not break API compatibility
* ``micro`` increments represent bugfix releases or improvements in documentation

All early releases however will simply recieve a ``micro`` version bump regardless of
how major the changes may be.

0.0.5 - Fix For Merging of Estimation Requests
----------------------------------------------

This release implements a fix for a major bug which caused incorrect results to be returned when issuing multiple estimation requestions at the same time - namely, the returned results became jumbled between the different requests. As an example, if a request was made to estimate a data set using the `smirnoff99frosst` force field, and then straight after with the `gaff 1.81` force field, the results of the `smirnoff99frosst` request may contain some properties estimated with `gaff 1.81` and vice versa.

This issue does not affect cases where only a single request was made and completed at a time (i.e the results of the previous request completed before the next estimate request was made).

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
