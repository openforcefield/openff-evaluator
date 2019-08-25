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

Bugfixes
""""""""

* PR #X: Fixes ...


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
