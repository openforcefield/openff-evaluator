Release Process
===============

This document aims to outline the steps needed to release the ``propertyestimator`` on ``omnia``. This
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
    on [our ReadTheDocs page](https://property-estimator.readthedocs.io/en/latest/releasehistory.html)

    See our [installation instructions](https://property-estimator.readthedocs.io/en/latest/installation.html).

    Please report bugs, request features, or ask questions through our
    [issue tracker](https://github.com/openforcefield/propertyestimator/issues).

    **Please note that this is a pre-alpha release and there will still be major changes to the API
    prior to a stable 1.0.0 release.**

*Note - You do not need to upload any files. The source code will automatically be added as a `.tar.gz` file.*

3: Trigger a New Build on Omnia
-------------------------------

To trigger the build in ``omnia``:

1) Create branch or fork of omnia-md/conda-recipes with the following changes to propertyestimator in
`meta.yaml <https://github.com/omnia-md/conda-recipes/blob/master/propertyestimator/meta.yaml>`_:

  a) Set ``git_tag`` to match the git release tag
  b) Update the ``version`` to match the release (this will go into the conda package name)
  c) Set ``build`` to 0
  d) Update any dependencies in the ``requirements`` section
  e) If we want to push to special ``rc`` label use ``extra.upload``

2) Open PR to merge branch or fork into omnia-md master:

  a) The PR title should have the format ``[propertyestimator] X.Y.Z (label: rc)``
  b) No PR body text is needed
  c) Travis will run on this PR (~30 minutes) and attempt to build the package. Under no conditions will the package
     be uploaded before the PR is merged. This step is just to ensure that building doesn't crash.
  d) If the build is successful the PR should be reviewed and merged by the ``omnia`` maintainers
  e) **Once merged into master** the package is built again on travis, and pushed to the channel set in
     meta.yaml (``main``, ``beta``, or ``rc``)

3) Test the ``omnia`` package:

  a) ``conda install -c omnia/label/rc propertyestimator``

*Note: Omnia builds take about 30 minutes to run. When you open a PR the build will run, and you can check the bottom
of the travis logs for "package failed to build" listings. Some packages always fail (protons, assaytools), but
propertyestimator shouldn't be there. Ctrl-F for ``propertyestimator`` to ensure that it did build at all though.*

4: Update the ReadTheDocs Build Versions
--------------------------------------------

To ensure that the read the docs pages are updated:

1) Trigger a RTD build of ``latest``.
2) Under the ``Versions`` tab add the new release version to the list of built versions and **save**.
3) Verify the new version docs have been built and pushed correctly
4) Under ``Admin`` | ``Advanced Settings``: Set the new release version as Default version to display and **save**.
