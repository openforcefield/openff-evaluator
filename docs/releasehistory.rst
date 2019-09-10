Release History
===============

Releases will eventually follow the ``major.minor.micro`` scheme recommended by
`PEP440 <https://www.python.org/dev/peps/pep-0440/#final-releases>`_, where

* ``major`` increments denote a change that may break API compatibility with previous ``major`` releases
* ``minor`` increments add features but do not break API compatibility
* ``micro`` increments represent bugfix releases or improvements in documentation

All early releases however will simply recieve a ``micro`` version bump regardless of
how major the changes may be.


0.0.3 - ExcessMolarVolume and Typing Improvements 
-------------------------------------------------

This release implements a number of bug fixes and adds two key new features, namely built in support
for estimating excess molar volume measurements, and improved type checking for protocol inputs 
and outputs.

New Features
""""""""""""

* PR #98: ``Substance`` objects may now have components with multiple amount types.
* PR #101: Added support for estimating ``ExcessMolarVolume`` measurements from simulations.
* PR #104: ``typing.Union`` is now a valid type arguemt to ``protocol_output`` and ``protocol_input``.

Bugfixes
""""""""

* PR #94: Fixes exception when testing equality of ``ProtocolPath`` objects.
* PR #100: Fixes precision issues when ensuring mole fractions are `<= 1.0`.
* PR #102: Fixes replicated input for children of replicated protocols.
* PR #105: Fixes excess properties weighting by the wrong mole fractions.
* PR #107: Fixes excess properties being converged to the wrong uncertainty.

Breaking Changes
""""""""""""""""

* PR #98: ``Substance.get_amount`` renamed to ``Substance.get_amounts`` and now returns an
  immutable ``frozenset`` of ``Amount`` objects, rather than a single ``Amount``.
* PR #104: The ``DivideGradientByScalar``, ``MultiplyGradientByScalar``, ``AddGradients`` and ``SubtractGradients`` 
  ``WeightGradientByMoleFraction`` protocols have been removed. The ``WeightQuantityByMoleFraction`` has been renamed
  to ``WeightByMoleFraction``.

Migration Guide
"""""""""""""""

This release contained several public API breaking changes. For the most part, these can be
remedied by the follow steps:

* Change all instances of ``Substance.get_amount`` to ``Substance.get_amounts`` and handle.
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

* PR #90: Fixes merging gradient protocols with the same id.
* PR #92: Fixes replicating protocols for more than 10 template values.
* PR #93: Fixes ``ConditionalGroup`` objects losing their conditions input.

0.0.1 - Initial Release
-----------------------

The initial pre-alpha release of the framework.
