"""
A collection of property estimator compute backends which use dask as the distribution engine.
"""
import importlib
import logging
import multiprocessing
import os
import shutil

import dask
from dask import distributed
from dask_jobqueue import LSFCluster
from distributed import get_worker
from distributed.metrics import time
from distributed.deploy.adaptive import Adaptive
from distributed.utils import ignoring
from simtk import unit

from .backends import PropertyEstimatorBackend, ComputeResources, QueueWorkerResources


class _JobQueueAdaptive(Adaptive):
    """Recent changes to the `distributed` package has
    lead to breaking changes in dask-jobqueue when running
    clusters in adaptive mode. This class aims to band aid
    the problem until a better fix may be found.
    """

    async def _adapt(self):
        if self._adapting:  # Semaphore to avoid overlapping adapt calls
            return

        self._adapting = True
        try:
            recommendations = await self.recommendations()
            if not recommendations:
                return
            status = recommendations.pop("status")
            if status == "up":
                f = self.cluster.scale_up(**recommendations)
                self.log.append((time(), "up", recommendations))
                if hasattr(f, "__await__"):
                    await f

            elif status == "down":
                self.log.append((time(), "down", recommendations["workers"]))
                workers = await self._retire_workers(workers=recommendations["workers"])
        finally:
            self._adapting = False


class _AdaptiveLSFCluster(LSFCluster):
    """A version of the `dask-jobqueue` cluster which
    uses the `_JobQueueAdaptive` adaptor in place of the
    default `Adaptive`.
    """

    def adapt(
        self,
        minimum_cores=None,
        maximum_cores=None,
        minimum_memory=None,
        maximum_memory=None,
        **kwargs
    ):
        """The method is identical to the base method, except
        that the `_JobQueueAdaptive` class is used in place of
        `distributed.Adaptive`.
        """
        with ignoring(AttributeError):
            self._adaptive.stop()
        if not hasattr(self, "_adaptive_options"):
            self._adaptive_options = {}
        if "minimum" not in kwargs:
            if minimum_cores is not None:
                kwargs["minimum"] = self._get_nb_workers_from_cores(minimum_cores)
            elif minimum_memory is not None:
                kwargs["minimum"] = self._get_nb_workers_from_memory(minimum_memory)
        if "maximum" not in kwargs:
            if maximum_cores is not None:
                kwargs["maximum"] = self._get_nb_workers_from_cores(maximum_cores)
            elif maximum_memory is not None:
                kwargs["maximum"] = self._get_nb_workers_from_memory(maximum_memory)
        self._adaptive_options.update(kwargs)
        try:
            self._adaptive = _JobQueueAdaptive(self.scheduler, self, **self._adaptive_options)
        except Exception:
            self._adaptive = _JobQueueAdaptive(self, **self._adaptive_options)
        return self._adaptive


class BaseDaskBackend(PropertyEstimatorBackend):
    """An base dask backend class, which implements functionality
    which is common to all other dask backends.
    """

    def __init__(self, number_of_workers=1, resources_per_worker=ComputeResources()):
        """Constructs a new BaseDaskBackend"""

        super().__init__(number_of_workers, resources_per_worker)

        self._cluster = None
        self._client = None

    def __del__(self):
        self.stop()

    def start(self):

        self._client = distributed.Client(self._cluster,
                                          processes=False)

    def stop(self):

        if self._client is not None:
            self._client.close()
        if self._cluster is not None:
            self._cluster.close()

        if os.path.isdir('dask-worker-space'):
            shutil.rmtree('dask-worker-space')

    @staticmethod
    def _wrapped_function(function, *args, **kwargs):
        """A function which is wrapped around any function submitted via
        `submit_task`, which adds extra meta data to the args and kwargs
        (such as the compute resources available to the function) and may
        perform extra validation before the function is passed to dask.

        Parameters
        ----------
        function: function
            The function which will be executed by dask.
        args: Any
            The list of args to pass to the function.
        kwargs: Any
            The list of kwargs to pass to the function.

        Returns
        -------
        Any
            Returns the output of the function without modification, unless
            an uncaught exception is raised in which case a PropertyEstimatorException
            is returned.
        """
        raise NotImplementedError()


class DaskLSFBackend(BaseDaskBackend):
    """A property estimator backend which uses a dask-jobqueue `LSFCluster`
    objects to run calculations within an existing LSF queue.
    """

    def __init__(self,
                 minimum_number_of_workers=1,
                 maximum_number_of_workers=1,
                 resources_per_worker=QueueWorkerResources(),
                 queue_name='default',
                 setup_script_commands=None,
                 extra_script_options=None,
                 adaptive_interval='10000ms',
                 disable_nanny_process=False):

        """Constructs a new DaskLocalCluster

        Parameters
        ----------
        minimum_number_of_workers: int
            The minimum number of workers to request from the queue system.
        maximum_number_of_workers: int
            The maximum number of workers to request from the queue system.
        resources_per_worker: QueueWorkerResources
            The resources to request per worker.
        queue_name: str
            The name of the queue which the workers will be requested
            from.
        setup_script_commands: list of str
            A list of bash script commands to call within the queue submission
            script before the call to launch the dask worker.

            This may include activating a python environment, or loading
            an environment module
        extra_script_options: list of str
            A list of extra job specific options to include in the queue
            submission script. These will get added to the script header in the form

            #BSUB <extra_script_options[x]>
        adaptive_interval: str
            The interval between attempting to either scale up or down
            the cluster, of of the from 'XXXms'.
        disable_nanny_process: bool
            If true, dask workers will be started in `--no-nanny` mode. This
            is required if using multiprocessing code within submitted tasks.

            This has not been fully tested yet and my lead to stability issues
            with the workers.

        Examples
        --------
        To create an LSF queueing compute backend which will attempt to spin up
        workers which have access to a single GPU.

        >>> # Create a resource object which will request a worker with
        >>> # one gpu which will stay alive for five hours.
        >>> from propertyestimator.backends import QueueWorkerResources
        >>>
        >>> resources = QueueWorkerResources(number_of_threads=1,
        >>>                                  number_of_gpus=1,
        >>>                                  preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
        >>>                                  wallclock_time_limit='05:00')
        >>>
        >>> # Define the set of commands which will set up the correct environment
        >>> # for each of the workers.
        >>> setup_script_commands = [
        >>>     'module load cuda/9.2',
        >>> ]
        >>>
        >>> # Define extra options to only run on certain node groups
        >>> extra_script_options = [
        >>>     '-m "ls-gpu lt-gpu"'
        >>> ]
        >>>
        >>>
        >>> # Create the backend which will adaptively try to spin up between one and
        >>> # ten workers with the requested resources depending on the calculation load.
        >>> from propertyestimator.backends import DaskLSFBackend
        >>> from simtk.unit import unit
        >>>
        >>> lsf_backend = DaskLSFBackend(minimum_number_of_workers=1,
        >>>                              maximum_number_of_workers=10,
        >>>                              resources_per_worker=resources,
        >>>                              queue_name='gpuqueue',
        >>>                              setup_script_commands=setup_script_commands,
        >>>                              extra_script_options=extra_script_options)
        """

        super().__init__(minimum_number_of_workers, resources_per_worker)

        assert isinstance(resources_per_worker, QueueWorkerResources)

        assert minimum_number_of_workers <= maximum_number_of_workers

        if resources_per_worker.number_of_gpus > 0:

            if resources_per_worker.preferred_gpu_toolkit == ComputeResources.GPUToolkit.OpenCL:
                raise ValueError('The OpenCL gpu backend is not currently supported.')

            if resources_per_worker.number_of_gpus > 1:
                raise ValueError('Only one GPU per worker is currently supported.')

        # For now we need to set this to some high number to ensure
        # jobs restarting because of workers being killed (due to
        # wall-clock time limits mainly) do not get terminated. This
        # should mostly be safe as we most wrap genuinely thrown
        # exceptions up as PropertyEstimatorExceptions and return these
        # gracefully (such that the task won't be marked as failed by
        # dask).
        dask.config.set({'distributed.scheduler.allowed-failures': 500})
        # dask.config.set({'distributed.worker.daemon': False})

        self._minimum_number_of_workers = minimum_number_of_workers
        self._maximum_number_of_workers = maximum_number_of_workers

        self._queue_name = queue_name

        self._setup_script_commands = setup_script_commands
        self._extra_script_options = extra_script_options

        self._adaptive_interval = adaptive_interval

        self._disable_nanny_process = disable_nanny_process

    def start(self):

        from dask_jobqueue.lsf import lsf_detect_units, lsf_format_bytes_ceil

        requested_memory = self._resources_per_worker.per_thread_memory_limit
        memory_bytes = requested_memory.value_in_unit(unit.byte)

        lsf_units = lsf_detect_units()
        memory_string = f'{lsf_format_bytes_ceil(memory_bytes, lsf_units=lsf_units)}{lsf_units.upper()}'

        job_extra = []

        if self._resources_per_worker.number_of_gpus > 0:

            job_extra = [
                '-gpu num={}:j_exclusive=yes:mode=shared:mps=no:'.format(self._resources_per_worker.number_of_gpus)
            ]

        if self._extra_script_options is not None:
            job_extra.extend(self._extra_script_options)

        extra = None if not self._disable_nanny_process else ['--no-nanny']

        self._cluster = _AdaptiveLSFCluster(queue=self._queue_name,
                                            cores=self._resources_per_worker.number_of_threads,
                                            walltime=self._resources_per_worker.wallclock_time_limit,
                                            memory=memory_string,
                                            mem=memory_bytes,
                                            job_extra=job_extra,
                                            env_extra=self._setup_script_commands,
                                            extra=extra,
                                            local_directory='dask-worker-space',
                                            log_directory='dask-worker-logs')

        self._cluster.adapt(minimum=self._minimum_number_of_workers,
                            maximum=self._maximum_number_of_workers, interval=self._adaptive_interval)

        super(DaskLSFBackend, self).start()

    @staticmethod
    def _wrapped_function(function, *args, **kwargs):

        from propertyestimator.workflow.plugins import available_protocols

        available_resources = kwargs['available_resources']
        per_worker_logging = kwargs.pop('per_worker_logging')

        gpu_assignments = kwargs.pop('gpu_assignments')

        # Set up the logging per worker if the flag is set to True.
        if per_worker_logging:
            # Each worker should have its own log file.
            formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                                          datefmt='%H:%M:%S')

            logger_path = '{}.log'.format(get_worker().id)

            logger = logging.getLogger()

            if not len(logger.handlers):
                logger_handler = logging.FileHandler(logger_path)
                logger_handler.setFormatter(formatter)

                logger.setLevel(logging.INFO)
                logger.addHandler(logger_handler)

        # Each spun up worker doesn't automatically import
        # all of the modules which were imported in the main
        # launch script, and as such custom plugins will no
        # longer be registered. We re-import / register them
        # here.
        if 'available_protocols' in kwargs:

            protocols_to_import = kwargs.pop('available_protocols')

            for protocol_class in protocols_to_import:
                module_name = '.'.join(protocol_class.split('.')[:-1])
                class_name = protocol_class.split('.')[-1]

                imported_module = importlib.import_module(module_name)
                available_protocols[class_name] = getattr(imported_module, class_name)

        if available_resources.number_of_gpus > 0:

            worker_id = distributed.get_worker().id

            available_resources._gpu_device_indices = ('0' if worker_id not in gpu_assignments
                                                       else gpu_assignments[worker_id])

            logging.info(f'Launching a job with access to GPUs {available_resources._gpu_device_indices}')

        return function(*args, **kwargs)

    def submit_task(self, function, *args, **kwargs):

        from propertyestimator.workflow.plugins import available_protocols

        key = kwargs.pop('key', None)

        protocols_to_import = [protocol_class.__module__ + '.' +
                               protocol_class.__qualname__ for protocol_class in available_protocols.values()]

        return self._client.submit(DaskLSFBackend._wrapped_function,
                                   function,
                                   *args,
                                   available_resources=self._resources_per_worker,
                                   available_protocols=protocols_to_import,
                                   gpu_assignments={},
                                   per_worker_logging=True,
                                   key=key)


class DaskLocalCluster(BaseDaskBackend):
    """A property estimator backend which uses a dask `LocalCluster` to
    run calculations.
    """

    def __init__(self, number_of_workers=1, resources_per_worker=ComputeResources()):
        """Constructs a new DaskLocalCluster"""

        super().__init__(number_of_workers, resources_per_worker)

        self._gpu_device_indices_by_worker = {}

        maximum_threads = multiprocessing.cpu_count()
        requested_threads = number_of_workers * resources_per_worker.number_of_threads

        if requested_threads > maximum_threads:

            raise ValueError('The total number of requested threads ({})is greater than is available on the'
                             'machine ({})'.format(requested_threads, maximum_threads))

        if resources_per_worker.number_of_gpus > 0:

            if resources_per_worker.preferred_gpu_toolkit == ComputeResources.GPUToolkit.OpenCL:
                raise ValueError('The OpenCL gpu backend is not currently supported.')

            if resources_per_worker.number_of_gpus > 1:
                raise ValueError('Only one GPU per worker is currently supported.')

            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

            if visible_devices is None:
                raise ValueError('The CUDA_VISIBLE_DEVICES variable is empty.')

            gpu_device_indices = visible_devices.split(',')

            if len(gpu_device_indices) != number_of_workers:
                raise ValueError('The number of available GPUs {} must match '
                                 'the number of requested workers {}.')

    def start(self):

        self._cluster = distributed.LocalCluster(self._number_of_workers,
                                                 1,
                                                 processes=False)

        if self._resources_per_worker.number_of_gpus > 0:

            for index, worker in enumerate(self._cluster.workers):
                self._gpu_device_indices_by_worker[worker.id] = str(index)

        super(DaskLocalCluster, self).start()

    @staticmethod
    def _wrapped_function(function, *args, **kwargs):

        available_resources = kwargs['available_resources']
        gpu_assignments = kwargs.pop('gpu_assignments')

        if available_resources.number_of_gpus > 0:

            worker_id = distributed.get_worker().id
            available_resources._gpu_device_indices = gpu_assignments[worker_id]

            logging.info('Launching a job with access to GPUs {}'.format(gpu_assignments[worker_id]))

        return function(*args, **kwargs)

    def submit_task(self, function, *args, **kwargs):

        key = kwargs.pop('key', None)

        return self._client.submit(DaskLocalCluster._wrapped_function,
                                   function,
                                   *args,
                                   key=key,
                                   available_resources=self._resources_per_worker,
                                   gpu_assignments=self._gpu_device_indices_by_worker)
