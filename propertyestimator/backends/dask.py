"""
A collection of property estimator compute backends which use dask as the distribution engine.
"""
import logging
import multiprocessing
import os

from dask import distributed

from .backends import PropertyEstimatorBackend, ComputeResources


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
