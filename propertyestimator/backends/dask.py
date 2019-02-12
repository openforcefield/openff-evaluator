"""
A collection of property estimator compute backends which use dask as the distribution engine.
"""

import multiprocessing

from dask import distributed

from .backends import PropertyEstimatorBackend, ComputeResources


class DaskLocalClusterBackend(PropertyEstimatorBackend):
    """A property estimator backend which uses a dask `LocalCluster` to
    run calculations.
    """

    def __init__(self, number_of_workers=1, resources_per_worker=ComputeResources()):

        """Constructs a new DaskLocalClusterBackend"""

        super().__init__(number_of_workers, resources_per_worker)

        maximum_threads = multiprocessing.cpu_count()
        requested_threads = number_of_workers * resources_per_worker.number_of_threads

        if requested_threads > maximum_threads:

            raise ValueError('The total number of requested threads ({})is greater than is available on the'
                             'machine ({})'.format(requested_threads, maximum_threads))

        # TODO: Check GPUs

        self._cluster = None
        self._client = None

    def start(self):

        self._cluster = distributed.LocalCluster(self._number_of_workers,
                                                 1,
                                                 processes=False,
                                                 resources=self._resources_per_worker.dict())

        self._client = distributed.Client(self._cluster,
                                          processes=False)

    def stop(self):

        self._client.close()
        self._cluster.close()

    def submit_task(self, function, *args):

        return self._client.submit(function, *args, resources=self._resources_per_worker.dict(),
                                   available_resources=self._resources_per_worker)
