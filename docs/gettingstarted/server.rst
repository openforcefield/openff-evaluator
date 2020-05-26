.. |evaluator_server|      replace:: :py:class:`~openff.evaluator.server.EvaluatorServer`
.. |batch|                 replace:: :py:class:`~openff.evaluator.server.Batch`

.. |local_file_storage|    replace:: :py:class:`~openff.evaluator.storage.LocalFileStorage`

.. |same_components|        replace:: :py:attr:`~openff.evaluator.client.BatchMode.SameComponents`
.. |shared_components|     replace:: :py:attr:`~openff.evaluator.client.BatchMode.SharedComponents`
.. |batch_mode_attr|       replace:: :py:attr:`~openff.evaluator.client.RequestOptions.batch_mode`

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
When a server recieves a request from a client, it will attempt to split the requested set of properties into
smaller batches, represented by the |batch| object. The server is currently only able to mark entire batches of
estimated properties as being completed, as opposed to individual properties.

Currently the server supports two ways of batching properties:

.. rst-class:: spaced-list

    * |same_components|: All properties measured for the substance containing the *same* components will be batched
      together. As an example, the density of a 80:20 and a 20:80 mix of ethanol and water would be batched together,
      but the density of pure ethanol and the density of pure water would be placed into separate batches.

    * |shared_components|: All properties measured for substances containing at least one common component will be
      batched together. As an example, the densities of 80:20 and 20:80 mixtures of ethanol and water, and the pure
      densities of ethanol and water would be batched together.

The mode of batching is set by the client using the |batch_mode_attr| attribute of the request options.
