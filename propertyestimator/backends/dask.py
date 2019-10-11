"""
A collection of property estimator compute backends which use dask as the distribution engine.
"""
import importlib
import logging
import multiprocessing
import os
import shutil
import traceback

import dask
from dask import distributed
from dask_jobqueue import LSFCluster, PBSCluster
from distributed import get_worker
from propertyestimator import unit

from .backends import PropertyEstimatorBackend, ComputeResources, QueueWorkerResources


class _Multiprocessor:
    """A temporary utility class which runs a given
    function in a separate process.
    """

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        """A wrapper around the function to run in a separate
        process which sets up logging and handle any extra
        module loading.

        Parameters
        ----------
        func: function
            The function to run in this process.
        queue: Queue
            The queue used to pass the results back
            to the parent process.
        args: tuple
            The args to pass to the function
        kwargs: dict
            The kwargs to pass to the function
        """

        try:

            from propertyestimator.workflow.plugins import available_protocols

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

            if 'logger_path' in kwargs:

                formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                                              datefmt='%H:%M:%S')

                logger_path = kwargs.pop('logger_path')

                logger = logging.getLogger()

                if not len(logger.handlers):
                    logger_handler = logging.FileHandler(logger_path)
                    logger_handler.setFormatter(formatter)

                    logger.setLevel(logging.INFO)
                    logger.addHandler(logger_handler)

            return_value = func(*args, **kwargs)
            queue.put(return_value)
        except Exception as e:
            queue.put((e, e.__traceback__))

    @staticmethod
    def run(function, *args, **kwargs):
        """Runs a functions in its own process.

        Parameters
        ----------
        function: function
            The function to run.
        args: Any
            The arguments to pass to the function.
        kwargs: Any
            The key word arguments to pass to the function.

        Returns
        -------
        Any
            The result of the function
        """

        import propertyestimator
        # An unpleasant way to ensure that codecov works correctly
        # when testing on travis.
        if hasattr(propertyestimator, '_called_from_test'):
            return function(*args, **kwargs)

        # queue = multiprocessing.Queue()
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        target_args = [function, queue, args, kwargs]

        process = multiprocessing.Process(target=_Multiprocessor._wrapper, args=target_args)
        process.start()

        return_value = queue.get()
        process.join()

        if isinstance(return_value, tuple) and len(return_value) > 0 and isinstance(return_value[0], Exception):

            formatted_exception = traceback.format_exception(None, return_value[0], return_value[1])
            logging.info(f'{formatted_exception} {return_value[0]} {return_value[1]}')

            raise return_value[0]

        return return_value


class BaseDaskBackend(PropertyEstimatorBackend):
    """A base `dask` backend class, which implements functionality
    which is common to all other `dask` based backends.
    """

    def __init__(self, number_of_workers=1, resources_per_worker=ComputeResources()):
        """Constructs a new BaseDaskBackend object."""

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
        perform extra validation before the function is passed to `dask`.

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


class BaseDaskJobQueueBackend(BaseDaskBackend):
    """A property estimator backend which uses a `dask_jobqueue.JobQueueCluster`
    object to run calculations within an existing HPC queuing system.

    See Also
    --------
    dask_jobqueue.JobQueueCluster
    """

    def __init__(self,
                 minimum_number_of_workers=1,
                 maximum_number_of_workers=1,
                 resources_per_worker=QueueWorkerResources(),
                 queue_name='default',
                 setup_script_commands=None,
                 extra_script_options=None,
                 adaptive_interval='10000ms',
                 disable_nanny_process=False,
                 cluster_type=None):

        """Constructs a new BaseDaskJobQueueBackend object

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
        """

        super().__init__(minimum_number_of_workers, resources_per_worker)

        assert isinstance(resources_per_worker, QueueWorkerResources)
        assert minimum_number_of_workers <= maximum_number_of_workers

        assert cluster_type is not None

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

        self._minimum_number_of_workers = minimum_number_of_workers
        self._maximum_number_of_workers = maximum_number_of_workers

        self._queue_name = queue_name

        self._setup_script_commands = setup_script_commands
        self._extra_script_options = extra_script_options

        self._adaptive_interval = adaptive_interval

        self._disable_nanny_process = disable_nanny_process

        self._cluster_type = cluster_type

    def _get_env_extra(self):
        """Returns a list of extra commands to run before
        the dask worker is started.

        Returns
        -------
        list of str
            The extra commands to run.
        """
        env_extra = dask.config.get(f'jobqueue.{self._cluster_type}.env-extra', default=[])

        if self._setup_script_commands is not None:
            env_extra.extend(self._setup_script_commands)

        return env_extra

    def _get_job_extra(self):
        """Returns a list of extra options to add to the
        worker job script header lines.

        Returns
        -------
        list of str
            The extra header options to add.
        """
        job_extra = dask.config.get(f'jobqueue.{self._cluster_type}.job-extra', default=[])

        if self._extra_script_options is not None:
            job_extra.extend(self._extra_script_options)

        return job_extra

    def _get_cluster_class(self):
        """Returns the type of `dask_jobqueue.JobQueueCluster` to
        create.

        Returns
        -------
        class
            The class of cluster to create.
        """
        raise NotImplementedError()

    def _get_extra_cluster_kwargs(self):
        """Returns a dictionary of extra kwargs to pass to the cluster.

        Returns
        -------
        dict of str and Any
            The kwargs dictionary to pass.
        """
        return {}

    def job_script(self):
        """Returns the job script that dask will use to submit workers.
        The backend must be started before calling this function.

        Returns
        -------
        str
        """
        if self._cluster is None:

            raise ValueError('The cluster is not initialized. This is usually'
                             'caused by calling `job_script` before `start`.')

        return self._cluster.job_script()

    def start(self):

        requested_memory = self._resources_per_worker.per_thread_memory_limit
        memory_string = f'{requested_memory.to(unit.byte):~}'.replace(' ', '')

        job_extra = self._get_job_extra()
        env_extra = self._get_env_extra()

        extra = None if not self._disable_nanny_process else ['--no-nanny']

        cluster_class = self._get_cluster_class()

        self._cluster = cluster_class(queue=self._queue_name,
                                      cores=self._resources_per_worker.number_of_threads,
                                      memory=memory_string,
                                      walltime=self._resources_per_worker.wallclock_time_limit,
                                      job_extra=job_extra,
                                      env_extra=env_extra,
                                      extra=extra,
                                      local_directory='dask-worker-space',
                                      **self._get_extra_cluster_kwargs())

        # The very small target duration is an attempt to force dask to scale
        # based on the number of processing tasks per worker.
        self._cluster.adapt(minimum=self._minimum_number_of_workers,
                            maximum=self._maximum_number_of_workers,
                            interval=self._adaptive_interval,
                            target_duration='0.00000000001s')

        super(BaseDaskJobQueueBackend, self).start()

    @staticmethod
    def _wrapped_function(function, *args, **kwargs):

        available_resources = kwargs['available_resources']
        per_worker_logging = kwargs.pop('per_worker_logging')

        gpu_assignments = kwargs.pop('gpu_assignments')

        # Set up the logging per worker if the flag is set to True.
        if per_worker_logging:

            # Each worker should have its own log file.
            kwargs['logger_path'] = '{}.log'.format(get_worker().id)

        if available_resources.number_of_gpus > 0:

            worker_id = distributed.get_worker().id

            available_resources._gpu_device_indices = ('0' if worker_id not in gpu_assignments
                                                       else gpu_assignments[worker_id])

            logging.info(f'Launching a job with access to GPUs {available_resources._gpu_device_indices}')

        return_value = _Multiprocessor.run(function, *args, **kwargs)
        return return_value
        # return function(*args, **kwargs)

    def submit_task(self, function, *args, **kwargs):

        from propertyestimator.workflow.plugins import available_protocols

        key = kwargs.pop('key', None)

        protocols_to_import = [protocol_class.__module__ + '.' +
                               protocol_class.__qualname__ for protocol_class in available_protocols.values()]

        return self._client.submit(BaseDaskJobQueueBackend._wrapped_function,
                                   function,
                                   *args,
                                   available_resources=self._resources_per_worker,
                                   available_protocols=protocols_to_import,
                                   gpu_assignments={},
                                   per_worker_logging=True,
                                   key=key)


class DaskLSFBackend(BaseDaskJobQueueBackend):
    """A property estimator backend which uses a `dask_jobqueue.LSFCluster`
    object to run calculations within an existing LSF queue.

    See Also
    --------
    dask_jobqueue.LSFCluster
    DaskPBSBackend
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

        """Constructs a new DaskLSFBackend object

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
        >>>
        >>> lsf_backend = DaskLSFBackend(minimum_number_of_workers=1,
        >>>                              maximum_number_of_workers=10,
        >>>                              resources_per_worker=resources,
        >>>                              queue_name='gpuqueue',
        >>>                              setup_script_commands=setup_script_commands,
        >>>                              extra_script_options=extra_script_options)
        """

        super().__init__(minimum_number_of_workers,
                         maximum_number_of_workers,
                         resources_per_worker,
                         queue_name,
                         setup_script_commands,
                         extra_script_options,
                         adaptive_interval,
                         disable_nanny_process,
                         cluster_type='lsf')

    def _get_job_extra(self):

        job_extra = super(DaskLSFBackend, self)._get_job_extra()

        if self._resources_per_worker.number_of_gpus > 0:

            job_extra.append(
                '-gpu num={}:j_exclusive=yes:mode=shared:mps=no:'.format(self._resources_per_worker.number_of_gpus)
            )

        return job_extra

    def _get_extra_cluster_kwargs(self):

        requested_memory = self._resources_per_worker.per_thread_memory_limit
        memory_bytes = requested_memory.to(unit.byte).magnitude

        extra_kwargs = super(DaskLSFBackend, self)._get_extra_cluster_kwargs()
        extra_kwargs.update({'mem': memory_bytes})

        return extra_kwargs

    def _get_cluster_class(self):
        return LSFCluster


class DaskPBSBackend(BaseDaskJobQueueBackend):
    """A property estimator backend which uses a `dask_jobqueue.PBSCluster`
    object to run calculations within an existing PBS queue.

    See Also
    --------
    dask_jobqueue.LSFCluster
    DaskLSFBackend
    """

    def __init__(self,
                 minimum_number_of_workers=1,
                 maximum_number_of_workers=1,
                 resources_per_worker=QueueWorkerResources(),
                 queue_name='default',
                 setup_script_commands=None,
                 extra_script_options=None,
                 adaptive_interval='10000ms',
                 disable_nanny_process=False,
                 resource_line=None):

        """Constructs a new DaskLSFBackend object

        Parameters
        ----------
        resource_line: str
            The string to pass to the `#PBS -l` line.

        Examples
        --------
        To create a PBS queueing compute backend which will attempt to spin up
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
        >>> # Create the backend which will adaptively try to spin up between one and
        >>> # ten workers with the requested resources depending on the calculation load.
        >>> from propertyestimator.backends import DaskPBSBackend
        >>>
        >>> pbs_backend = DaskPBSBackend(minimum_number_of_workers=1,
        >>>                              maximum_number_of_workers=10,
        >>>                              resources_per_worker=resources,
        >>>                              queue_name='gpuqueue',
        >>>                              setup_script_commands=setup_script_commands)
        """

        super().__init__(minimum_number_of_workers,
                         maximum_number_of_workers,
                         resources_per_worker,
                         queue_name,
                         setup_script_commands,
                         extra_script_options,
                         adaptive_interval,
                         disable_nanny_process,
                         cluster_type='pbs')

        self._resource_line = resource_line

    def _get_extra_cluster_kwargs(self):

        extra_kwargs = super(DaskPBSBackend, self)._get_extra_cluster_kwargs()
        extra_kwargs.update({'resource_spec': self._resource_line})

        return extra_kwargs

    def _get_cluster_class(self):
        return PBSCluster


class DaskLocalCluster(BaseDaskBackend):
    """A property estimator backend which uses a `dask` `LocalCluster`
    object to run calculations on a single machine.

    See Also
    --------
    dask.LocalCluster
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

            for index, worker in self._cluster.workers.items():
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

        return_value = _Multiprocessor.run(function, *args, **kwargs)
        return return_value
        # return function(*args, **kwargs)

    def submit_task(self, function, *args, **kwargs):

        key = kwargs.pop('key', None)

        return self._client.submit(DaskLocalCluster._wrapped_function,
                                   function,
                                   *args,
                                   key=key,
                                   available_resources=self._resources_per_worker,
                                   gpu_assignments=self._gpu_device_indices_by_worker)
