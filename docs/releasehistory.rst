Release History
===============

Releases will eventually follow the ``major.minor.micro`` scheme recommended by
`PEP440 <https://www.python.org/dev/peps/pep-0440/#final-releases>`_, where

* ``major`` increments denote a change that may break API compatibility with previous ``major`` releases
* ``minor`` increments add features but do not break API compatibility
* ``micro`` increments represent bugfix releases or improvements in documentation

All early releases however will simply recieve a ``micro`` version bump regardless of
how major the changes may be.


0.0.3 -
------------------------------

This release...

New Features
""""""""""""

* PR #98: ``Substance`` objects may now have components with multiple amount types

Bugfixes
""""""""

* PR #94: Fixes exception when testing equality of ``ProtocolPath`` objects
* PR #100: Fixes precision issues when ensuring mole fractions are `<= 1.0`
* PR #102: Fixes replicated input for children of replicated protocols

Breaking Changes
""""""""""""""""

* PR #98: ``Substance.get_amount`` renamed to ``Substance.get_amounts`` and now returns an
  immutable ``frozenset`` of ``Amount`` objects, rather than a single ``Amount``.

Migration Guide
"""""""""""""""

This release contained several public API breaking changes. For the most part, these can be
remedied by the follow steps:

* Change all instances of ``Substance.get_amount`` to ``Substance.get_amounts`` and handle
  the newly returned list of amounts, rather than the previously returned single amount.



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
