Release History
===============

Releases will eventually follow the ``major.minor.micro`` scheme recommended by
`PEP440 <https://www.python.org/dev/peps/pep-0440/#final-releases>`_, where

* ``major`` increments denote a change that may break API compatibility with previous ``major`` releases
* ``minor`` increments add features but do not break API compatibility
* ``micro`` increments represent bugfix releases or improvements in documentation

All early releases however will simply recieve a ``micro`` version bump regardless of
how major the changes may be.

0.0.2 - Gradient Merging Bug Fix
--------------------------------

A minor release to fix a bug which caused an exception to be raised when merging workflows of
different property types.

Bugfixes
""""""""

* PR #90: Fixes gradient merging bug.

0.0.1 - Initial Release
-----------------------

The initial pre-alpha release of the framework.