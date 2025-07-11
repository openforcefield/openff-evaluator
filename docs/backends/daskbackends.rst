.. |dask_local_cluster|             replace:: :py:class:`~openff.evaluator.backends.dask.DaskLocalCluster`
.. |base_dask_job_queue_backend|    replace:: :py:class:`~openff.evaluator.backends.dask.BaseDaskJobQueueBackend`
.. |dask_lsf_backend|               replace:: :py:class:`~openff.evaluator.backends.dask.DaskLSFBackend`
.. |dask_pbs_backend|               replace:: :py:class:`~openff.evaluator.backends.dask.DaskPBSBackend`
.. |dask_slurm_backend|             replace:: :py:class:`~openff.evaluator.backends.dask.DaskSLURMBackend`
.. |queue_worker_resources|         replace:: :py:class:`~openff.evaluator.backends.dask.QueueWorkerResources`

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
        preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
    )

    with DaskLocalCluster(number_of_workers=1, resources_per_worker=worker_resources) as local_backend:
        local_backend.submit_task(logging.info, "Hello World")
        ...

Its main purpose is for use when debugging calculations locally, or when running calculations on machines with large
numbers of CPUs or GPUs.

Dask HPC Cluster
----------------

The |dask_lsf_backend|, |dask_pbs_backend|, and |dask_slurm_backend| backends wrap around the dask `LSFCluster
<https://jobqueue.dask.org/en/
latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster>`_, `PBSCluster <https://jobqueue.dask.org/
en/latest/generated/dask_jobqueue.PBSCluster.html#dask_jobqueue.PBSCluster>`_, and `SLURMCluster
<https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html#dask_jobqueue.SLURMCluster>`_ classes
respectively, and both inherit the |base_dask_job_queue_backend| class which implements the core of their
functionality. They predominantly run in an adaptive mode, whereby the backend will automatically scale up or down
the number of workers based on the current number of tasks that the backend is trying to execute.

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
        f"conda activate evaluator",
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

Selecting GPU Platform
----------------------
The calculation backends also allows the user to specify the GPU platform and precision level. Users can specify
either ``auto``, ``CUDA`` or ``OpenCL`` as the `preferred_gpu_toolkit` using the ``GPUToolkit`` enum class. The
default precision level is set to ``mixed`` but can be overridden by specifying `preferred_gpu_precision` using the
``GPUPrecision`` enum class::

    worker_resources = ComputeResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit.OpenCL,
        preferred_gpu_precision=ComputeResources.GPUPrecision.mixed,
    )

With ``GPUToolkit.auto``, the framework will determine the fastest available platform based on the precision level::

    worker_resources = ComputeResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit.auto,
        preferred_gpu_precision=ComputeResources.GPUPrecision.mixed,
    )
