.. |taproom_source|    replace:: :py:class:`~openff.evaluator.datasets.taproom.TaproomSource`
.. |taproom_data_set|  replace:: :py:class:`~openff.evaluator.datasets.taproom.TaproomDataSet`

Taproom
=======

The |taproom_data_set| object offers an API for retrieving host-guest binding affinity measurements from the
curated `taproom <https://github.com/slochower/host-guest-benchmarks>`_ repository.

.. note:: ``taproom`` may be installed by running ``conda install -c conda-forge taproom``

This includes retrieving all of the data available::

    from openff.evaluator.datasets.taproom import TaproomDataSet
    taproom_set = TaproomDataSet()

data measure for a single host molecule (e.g. alpha-cyclodextrin)::

    acd_taproom_set = TaproomDataSet(host_codes=["acd"])

or data for a particular host and guest pair::

    acd_taproom_set = TaproomDataSet(host_codes=["acd"], guest_codes=["bam"])

All measurements in this data set have an associated |taproom_source| as their source provenance. This tracks both
the original source of the measurement as well as the taproom identifier.

.. note:: Currently the data set object will assume a default set of buffer conditions (either no buffer, or a buffer
          of a salt with a specified ionic strength) rather than reading the buffer from the taproom measurement
          directory. This is consistent with previous applications of the data set.
