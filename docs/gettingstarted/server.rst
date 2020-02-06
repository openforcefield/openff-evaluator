.. |evaluator_server|      replace:: :py:class:`~evaluator.server.EvaluatorServer`
.. |batch|                 replace:: :py:class:`~evaluator.server.Batch`

.. |local_file_storage|    replace:: :py:class:`~evaluator.storage.LocalFileStorage`

Evaluator Server
================

The |evaluator_server| object is responsible for coordinating the estimation of physical property data sets as requested
by :doc:`evaluator clients <client>`. Its primary responsibilities are to:

.. rst-class:: spaced-list

    * recieve incoming requests from an :doc:`evaluator clients <client>` to either estimate a dataset of properties, or
      to query the status of a previous request.
    * request that each specified :doc:`calculation layers <../layers/calculationlayers>` attempt to estimate the data
      set of properties, cascading unestimated properties through the different layers.

An |evaluator_server| must be created with an accompanying :doc:`calculation backend <../backends/calculationbackend>`
which will be responsible for distributing any calculations launched by the different calculation layers::

    with DaskLocalCluster() as calculation_backend:

        evaluator_server = EvaluatorServer(calculation_backend)
        evaluator_server.start()

It may also be optionally created using a specific :doc:`storage backend <../storage/storagebackend>` if the default
|local_file_storage| is not sufficient::

    with DaskLocalCluster() as calculation_backend:

        storage_backend = LocalFileStorage()

        evaluator_server = EvaluatorServer(calculation_backend, storage_backend)
        evaluator_server.start()

By default the server will run synchronously until it is killed, however it may also be run asynchronously such that
it can be interacted with directly by a client in the same script::

    with DaskLocalCluster() as calculation_backend:

        with EvaluatorServer(calculation_backend) as evaluator_server:

            # Specify the data set.
            data_set = PhysicalPropertyDataSet()
            data_set.add_properties(...)

            # Specify the force field source.
            force_field = SmirnoffForceFieldSource.from_path("openff-1.0.0.offxml")

            # Request the estimation of the data set.
            request, errors = evaluator_client.request_estimate(data_set,force_field)
            # Wait for the results.
            results = request.results(synchronous=True)

Estimation Batches
------------------
By default when a server recieves a request from a client, it will attempt to split the requested set of properties into
smaller batches, represented by the |batch| object. The current behaviour is to batch together all properties which
were measured for the same substance.

This splitting into smaller batches allows the server to return back batches of properties as they complete, rather than
needing to wait for a full request to complete.

.. note:: This batching behaviour will be built upon and expanded in future versions of the evaluator framework.