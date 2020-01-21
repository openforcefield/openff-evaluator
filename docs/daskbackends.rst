.. |dask_local_cluster|    replace:: :py:class:`~propertyestimator.backends.dask.DaskLocalCluster`
.. |job_queue_backend|     replace:: :py:class:`~propertyestimator.backends.dask.BaseDaskJobQueueBackend`
.. |dask_lsf_backend|      replace:: :py:class:`~propertyestimator.backends.dask.DaskLSFBackend`
.. |dask_pbs_backend|      replace:: :py:class:`~propertyestimator.backends.dask.DaskPBSBackend`
.. |queue_resources|       replace:: :py:class:`~propertyestimator.backends.dask.QueueWorkerResources`

Dask Backends
=============

The framework implements a number of calculation backends which integrate with the ``dask`` `distributed <https://
distributed.dask.org/>`_ and `job-queue <https://dask-jobqueue.readthedocs.io>`_ libraries.

Dask Local Cluster
------------------

The |dask_local_cluster| backend wraps around the dask `LocalCluster <https://distributed.dask.org/en/latest/local-
cluster.html>`_ class to distribute tasks on a single machine::

    worker_resources = ComputeResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=GPUToolkit.CUDA,
    )

    with DaskLocalCluster(number_of_workers=1, resources_per_worker=worker_resources) as local_backend:
        local_backend.submit_task(logging.info, "Hello World")
        ...

Its main purpose is for use when debugging calculations locally, or when running calculations on machines with large
numbers of CPUs or GPUs.

Dask HPC Cluster
----------------

The |dask_lsf_backend| and |dask_pbs_backend| backends wrap around the dask `LSFCluster <https://jobqueue.dask.org/en/
latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster>`_ and `PBSCluster <https://jobqueue.dask.org/
en/latest/generated/dask_jobqueue.PBSCluster.html#dask_jobqueue.PBSCluster>`_ classes respectively, and both inherit
the |job_queue_backend| class which implements the core of their functionality. They predominantly run in an
adaptive mode, whereby the backend will automatically scale up or down the number of workers based on the current number
of tasks that the backend is trying to execute.

These backends integrate with the queueing systems which most HPC cluster use to manage task execution. They work
by submitting jobs into the queueing system which themselves spawn `dask workers <https://distributed.dask.org/en/
latest/worker.html>`_, which in turn then execute tasks on the available compute nodes::

    # Create the object which describes the compute resources each worker should request from
    # the queueing system.
    worker_resources = QueueWorkerResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
        per_thread_memory_limit=worker_memory,
        wallclock_time_limit="05:59",
    )

    # Create the backend object.
    setup_script_commands = [
        f"conda activate openff-evaluator",
        f"module load cuda/10.1",
    ]

    calculation_backend = DaskLSFBackend(
        minimum_number_of_workers=1,
        maximum_number_of_workers=max_number_of_workers,
        resources_per_worker=queue_resources,
        queue_name="gpuqueue",
        setup_script_commands=setup_script_commands,
    )

    # Perform some tasks.
    with calculation_backend:
        calculation_backend.submit_task(logging.info, "Hello World")
        ...

The ``setup_script_commands`` argument takes a list of commands which should be run by the queue job submission
script before spawning the actual worker. This enables setting up custom environments, and setting any required
environmental variables.

Configuration
^^^^^^^^^^^^^
To ensure optimal behaviour we recommend changing / uncommenting the following settings in the dask distributed
configuration file (this can be found at ``~/.config/dask/distributed.yaml``)::

    distributed:

        worker:
            daemon: False

        comm:
            timeouts:
                connect: 10s
                tcp: 30s

        deploy:
            lost-worker-timeout: 15s


See the `dask documentation <https://docs.dask.org/en/latest/configuration.html>`_ for more information about changing
``dask`` settings.