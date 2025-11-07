Release Process
===============

This document aims to outline the steps needed to release the ``openff-evaluator`` on ``conda-forge``. This
should only be done with the approval of the core maintainers.

1. Update the Release History
-----------------------------

If no PR has been submitted, create a new one to keep track of changes to the release notes *only*.
Only the ``releasehistory.rst`` file may be edited in this PR.

Ensure that the release history file is up to date, and conforms to the below template:

::

    X.Y.Z - Descriptive Title
    ------------------------------

    This release...

    New Features
    """"""""""""

    * PR #X: Feature summary

    Bugfixes
    """"""""

    * PR #Y: Fix Summary

    Breaking Changes
    """"""""""""""""

    * PR #Z: Descriptive summary of the breaking change

    Migration Guide
    """""""""""""""

    This release contained several public API breaking changes. For the most part, these can be
    remedied by the follow steps:

    * A somewhat verbose guide on how users should upgrade their code given the new breaking changes.

2: Cut the Release on GitHub
----------------------------

To cut a new release on GitHub:

1) Go to the ``Releases`` tab on the front page of the repo and choose ``Create a new release``.
2) Set the release tag using the form: ``X.Y.Z``
3) Added a descriptive title using the form: ``X.Y.Z [Descriptive Title]``
4) Ensure the ``This is a pre-release`` checkbox is ticked.
5) Reformat the release notes from part 1) into markdown and paste into the description box.

  a) Append the following extra message above the `New Features` title:

::

    A richer version of these release notes with live links to API documentation is available
    on [our ReadTheDocs page](https://openff-evaluator.readthedocs.io/en/latest/releasehistory.html)

    See our [installation instructions](https://openff-evaluator.readthedocs.io/en/latest/install.html).

    Please report bugs, request features, or ask questions through our
    [issue tracker](https://github.com/openforcefield/openff-evaluator/issues).

    **Please note that this is a pre-alpha release and there will still be major changes to the API
    prior to a stable 1.0.0 release.**

*Note - You do not need to upload any files. The source code will automatically be added as a `.tar.gz` file.*

3: Trigger a New Build on Conda Forge
-------------------------------------

To trigger the build on ``conda-forge``:

1) Create a fork of the `openff-evaluator-feedstock <https://github.com/conda-forge/openff-evaluator-feedstock>`_ and
make the following changes to the ``recipe/meta.yaml`` file:

  a) Update the ``version`` to match the release.
  b) Set ``build`` to 0
  c) Update any dependencies in the ``requirements`` section
  d) Update the sha256 hash to the output of ``curl -sL https://github.com/openforcefield/openff-evaluator/archive/{{ version }}.tar.gz | openssl sha256``

2) Open PR to merge the fork into the main feedstock:

  a) The PR title should have the format ``Release X.Y.Z``
  b) No PR body text is needed
  c) The CI will run on this PR (~30 minutes) and attempt to build the package.
  d) If the build is successful the PR should be reviewed and merged by the feedstock maintainers.
  e) **Once merged** the package is built again on and uploaded to anaconda.

3) Test the ``conda-forge`` package:

  a) ``conda install -c conda-forge openff-evaluator``

4: Update the ReadTheDocs Build Versions
--------------------------------------------

To ensure that the read the docs pages are updated:

1) Trigger a RTD build of ``latest``.
2) Under the ``Versions`` tab add the new release version to the list of built versions and **save**.
3) Verify the new version docs have been built and pushed correctly
4) Under ``Admin`` | ``Advanced Settings``: Set the new release version as Default version to display and **save**.
