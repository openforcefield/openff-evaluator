"""
A collection of property estimator compute backends which use dask as the distribution engine.
"""
import importlib
import logging
import multiprocessing
import os
import shutil

from dask import distributed
from simtk import unit

from propertyestimator.workflow.plugins import available_protocols
from .backends import PropertyEstimatorBackend, ComputeResources, QueueComputeResources


class DaskLSFBackend(PropertyEstimatorBackend):
    """A property estimator backend which uses a dask-jobqueue `LSFCluster`
    objects to run calculations within an existing LSF queue.
    """

    def __init__(self,
                 minimum_number_of_workers=1,
                 maximum_number_of_workers=1,
                 resources_per_worker=QueueComputeResources(),
                 default_memory_unit=unit.giga*unit.byte,
                 queue_name='default',
                 extra_script_commands=None):

        """Constructs a new DaskLocalClusterBackend

        Parameters
        ----------
        minimum_number_of_workers: int
            The minimum number of workers to request from the queue system.
        maximum_number_of_workers: int
            The maximum number of workers to request from the queue system.
        resources_per_worker: QueueComputeResources
            The resources to request per worker.
        default_memory_unit: simtk.Unit
            The default unit used by the LSF queuing system when
            defining memory usage limits / requirements - this
            must be compatible with `unit.bytes`.
        queue_name: str
            The name of the queue which the workers will be requested
            from.
        extra_script_commands: list of str
            A list of bash script commands to call within the queue submission
            script before the call to launch the dask worker.

            This may include activating a python environment, or loading
            an environment module

        Examples
        --------
        To create an LSF queueing compute backend which will attempt to spin up
        workers which have access to a GPU.

        >>> # Create a resource object which will request a worker with
        >>> # one gpu which will stay alive for five hours.
        >>> from propertyestimator.backends import QueueComputeResources
        >>>
        >>> resources = QueueComputeResources(number_of_threads=1,
        >>>                                   number_of_gpus=1,
        >>>                                   preferred_gpu_toolkit=QueueComputeResources.GPUToolkit.CUDA,
        >>>                                   wallclock_time_limit='05:00')
        >>>
        >>> # Define the set of commands which will set up the correct environment
        >>> # for each of the workers.
        >>> worker_script_commands = [
        >>>     'module load cuda/9.2',
        >>> ]
        >>>
        >>> # Create the backend which will adaptively try to spin up between one and
        >>> # ten workers with the requested resources depending on the calculation load.
        >>> from propertyestimator.backends import DaskLSFBackend
        >>> from simtk.unit import unit
        >>> lsf_backend = DaskLSFBackend(minimum_number_of_workers=1,
        >>>                              maximum_number_of_workers=10,
        >>>                              resources_per_worker=resources,
        >>>                              default_memory_unit=unit.gigabyte,
        >>>                              queue_name='gpuqueue',
        >>>                              extra_script_commands=worker_script_commands)
        """

        super().__init__(minimum_number_of_workers, resources_per_worker)

        assert isinstance(resources_per_worker, QueueComputeResources)

        assert minimum_number_of_workers <= maximum_number_of_workers

        if resources_per_worker.number_of_gpus > 0:

            if resources_per_worker.preferred_gpu_toolkit == ComputeResources.GPUToolkit.OpenCL:
                raise ValueError('The OpenCL gpu backend is not currently supported.')

            if resources_per_worker.number_of_gpus > 1:
                raise ValueError('Only one GPU per worker is currently supported.')

        self._cluster = None
        self._client = None

        self._minimum_number_of_workers = minimum_number_of_workers
        self._maximum_number_of_workers = maximum_number_of_workers

        self._default_memory_unit = default_memory_unit

        self._queue_name = queue_name

        self._extra_script_commands = extra_script_commands

    def start(self):

        from dask_jobqueue import LSFCluster

        requested_memory = self._resources_per_worker.per_thread_memory_limit

        memory_default_unit = requested_memory.value_in_unit(self._default_memory_unit)
        memory_bytes = requested_memory.value_in_unit(unit.byte)

        memory_string = '{}{}'.format(memory_default_unit, self._default_memory_unit.get_symbol())

        # Dask assumes we will be using mega bytes as the default unit, so we need
        # to multiply by a corrective factor to remove this assumption.
        lsf_byte_scale = (1 * (unit.mega * unit.byte)).value_in_unit(self._default_memory_unit)
        memory_bytes *= lsf_byte_scale

        job_extra = None

        if self._resources_per_worker.number_of_gpus > 0:

            job_extra = [
                '-gpu num={}:j_exclusive=yes:mode=shared:mps=no:'.format(self._resources_per_worker.number_of_gpus)
            ]

        self._cluster = LSFCluster(queue=self._queue_name,
                                   cores=self._resources_per_worker.number_of_threads,
                                   memory=memory_string,
                                   walltime=self._resources_per_worker.wallclock_time_limit,
                                   mem=memory_bytes,
                                   job_extra=job_extra,
                                   env_extra=self._extra_script_commands)

        self._cluster.adapt(minimum=self._minimum_number_of_workers,
                            maximum=self._maximum_number_of_workers)

        self._client = distributed.Client(self._cluster,
                                          processes=False)

    def stop(self):

        self._client.close()
        self._cluster.close()

    @staticmethod
    def _wrapped_function(function, *args, **kwargs):

        available_resources = kwargs['available_resources']
        protocols_to_import = kwargs.pop('available_protocols')

        gpu_assignments = kwargs.pop('gpu_assignments')

        for protocol_class in protocols_to_import:

            module_name = '.'.join(protocol_class.split('.')[:-1])
            class_name = protocol_class.split('.')[-1]

            imported_module = importlib.import_module(module_name)
            available_protocols[class_name] = getattr(imported_module, class_name)

        if available_resources.number_of_gpus > 0:

            worker_id = distributed.get_worker().id

            if gpu_assignments[worker_id] is not None:
                available_resources._gpu_device_indices = gpu_assignments[worker_id]
            else:
                available_resources._gpu_device_indices = '0'

            logging.info('Launching a job with access to GPUs {}'.format(gpu_assignments[worker_id]))

        return function(*args, **kwargs)

    def submit_task(self, function, *args):

        protocols_to_import = [protocol_class.__module__ + '.' +
                               protocol_class.__qualname__ for protocol_class in available_protocols.values()]

        return self._client.submit(DaskLSFBackend._wrapped_function,
                                   function,
                                   *args,
                                   available_resources=self._resources_per_worker,
                                   available_protocols=protocols_to_import,
                                   gpu_assignments={})


class DaskLocalClusterBackend(PropertyEstimatorBackend):
    """A property estimator backend which uses a dask `LocalCluster` to
    run calculations.
    """

    def __init__(self, number_of_workers=1, resources_per_worker=ComputeResources()):
        """Constructs a new DaskLocalClusterBackend"""

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

        self._cluster = None
        self._client = None

    def start(self):

        self._cluster = distributed.LocalCluster(self._number_of_workers,
                                                 1,
                                                 processes=False)

        if self._resources_per_worker.number_of_gpus > 0:

            for index, worker in enumerate(self._cluster.workers):
                self._gpu_device_indices_by_worker[worker.id] = str(index)

        self._client = distributed.Client(self._cluster,
                                          processes=False)

    def stop(self):

        self._client.close()
        self._cluster.close()

        if os.path.isdir('dask-worker-space'):
            shutil.rmtree('dask-worker-space')

    @staticmethod
    def _wrapped_function(function, *args, **kwargs):

        available_resources = kwargs['available_resources']
        gpu_assignments = kwargs.pop('gpu_assignments')

        if available_resources.number_of_gpus > 0:

            worker_id = distributed.get_worker().id
            available_resources._gpu_device_indices = gpu_assignments[worker_id]

            logging.info('Launching a job with access to GPUs {}'.format(gpu_assignments[worker_id]))

        return function(*args, **kwargs)

    def submit_task(self, function, *args):

        return self._client.submit(DaskLocalClusterBackend._wrapped_function,
                                   function,
                                   *args,
                                   available_resources=self._resources_per_worker,
                                   gpu_assignments=self._gpu_device_indices_by_worker)
