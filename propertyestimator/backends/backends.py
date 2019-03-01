"""
Defines the base API for the property estimator task calculation backend.
"""
from enum import Enum


class ComputeResources:
    """An object which stores how many of each type of computational resource
    (threads or gpu's) is available to a calculation task."""

    class GPUToolkit(Enum):
        CUDA = 'CUDA'
        OpenCL = 'OpenCL'

    @property
    def number_of_threads(self):
        return self._number_of_threads

    @property
    def number_of_gpus(self):
        return self._number_of_gpus

    @property
    def preferred_gpu_toolkit(self):
        """GPUToolkit: The toolkit to use when running on gpus."""
        return self._preferred_gpu_toolkit

    def __init__(self, number_of_threads=1, number_of_gpus=0, preferred_gpu_toolkit=None):
        """Constructs a new ComputeResources object.

        Parameters
        ----------
        number_of_threads: int
            The number of the threads available.
        number_of_gpus: int
            The number of the gpu's available.
        preferred_gpu_toolkit: ComputeResources.GPUToolkit, optional
            The preferred toolkit to use when running on gpus.
        """

        self._number_of_threads = number_of_threads
        self._number_of_gpus = number_of_gpus

        self._preferred_gpu_toolkit = preferred_gpu_toolkit
        
        assert self._number_of_threads >= 0
        assert self._number_of_gpus >= 0

        assert self._number_of_threads > 0 or self._number_of_gpus > 0

        if self._number_of_gpus > 0:
            assert self._preferred_gpu_toolkit is not None

    def dict(self):
        return self.__getstate__()

    def __getstate__(self):
        return {
            'number_of_threads': self.number_of_threads,
            'number_of_gpus': self.number_of_gpus,
            'preferred_gpu_toolkit': self.preferred_gpu_toolkit
        }

    def __setstate__(self, state):

        self._number_of_threads = state['number_of_threads']
        self._number_of_gpus = state['number_of_gpus']
        self._preferred_gpu_toolkit = state['preferred_gpu_toolkit']

    def __eq__(self, other):
        return self.number_of_threads == other.number_of_threads and \
               self.number_of_gpus == other.number_of_gpus and \
               self.preferred_gpu_toolkit == other.preferred_gpu_toolkit

    def __ne__(self, other):
        return not self.__eq__(other)


class PropertyEstimatorBackend:
    """An abstract base representation of a property estimator backend.

    A backend will be responsible for coordinating and running calculations
    on the available hardware.

    Notes
    -----
    All estimator backend classes must inherit from this class, and must implement the
    `start`, `stop`, and `submit_task` method.
    """

    def __init__(self, number_of_workers=1, resources_per_worker=ComputeResources()):

        """Constructs a new PropertyEstimatorBackend object.

        Parameters
        ----------
        number_of_workers : int
            The number of works to run the calculations on. One worker
            can perform a single task (e.g run a simulation) at once.
        resources_per_worker: ComputeResources
            The number of resources to request per worker.
        """

        self._number_of_workers = number_of_workers
        self._resources_per_worker = resources_per_worker

    def _get_worker_resources_dict(self):
        """Get dict representation of the resources requested
        by a worker.

        Returns
        -------
        dict of str and int
        """
        return {
            'number_of_threads': self._resources_per_worker.number_of_threads,
            'number_of_gpus': self._resources_per_worker.number_of_gpus,
        }

    def start(self):
        """Start the calculation backend."""
        pass

    def stop(self):
        """Stop the calculation backend."""
        pass

    def submit_task(self, function, *args, **kwargs):
        """Submit a task to the compute resources
        managed by this backend.

        Parameters
        ----------
        function: function
            The function to run.

        Returns
        -------
        Future
            Returns a future object which will eventually point to the results
            of the submitted task.
        """
        pass
