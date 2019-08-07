"""
Defines the base API for the property estimator task calculation backend.
"""
import re
from enum import Enum

from propertyestimator import unit


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

    @property
    def gpu_device_indices(self):
        """str: The indices of the GPUs to run on."""
        return self._gpu_device_indices

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
        # A workaround for when using a local cluster
        # backend which is strictly for internal purposes
        # only for now.
        self._gpu_device_indices = None
        
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
            'preferred_gpu_toolkit': self.preferred_gpu_toolkit,
            '_gpu_device_indices': self._gpu_device_indices
        }

    def __setstate__(self, state):

        self._number_of_threads = state['number_of_threads']
        self._number_of_gpus = state['number_of_gpus']
        self._preferred_gpu_toolkit = state['preferred_gpu_toolkit']
        self._gpu_device_indices = state['_gpu_device_indices']

    def __eq__(self, other):
        return self.number_of_threads == other.number_of_threads and \
               self.number_of_gpus == other.number_of_gpus and \
               self.preferred_gpu_toolkit == other.preferred_gpu_toolkit

    def __ne__(self, other):
        return not self.__eq__(other)


class QueueWorkerResources(ComputeResources):
    """An extended resource object with properties specific
    to calculations which will run on queue based resources,
    such as LSF, PBS or SLURM.
    """

    @property
    def per_thread_memory_limit(self):
        """simtk.Quantity: The maximum amount of memory available to each thread, such that
        the total memory limit will be `per_cpu_memory_limit * number_of_threads`."""
        return self._per_thread_memory_limit

    @property
    def wallclock_time_limit(self):
        """str: The maximum amount of wall clock time that a worker can run for. This should
        be a string of the form `HH:MM` where HH is the number of hours and MM the number of minutes"""
        return self._wallclock_time_limit

    def __init__(self, number_of_threads=1, number_of_gpus=0, preferred_gpu_toolkit=None,
                 per_thread_memory_limit=1*unit.gigabytes, wallclock_time_limit="01:00"):
        """Constructs a new ComputeResources object.

        Notes
        -----
        Both the requested `number_of_threads` and the `number_of_gpus` must be less than
        or equal to the number of threads (/cpus/cores) and GPUs available to each compute
        node in the cluster respectively, such that a single worker is able to be accommodated
        by a single compute node.

        Parameters
        ----------
        per_thread_memory_limit: simtk.Quantity
            The maximum amount of memory available to each thread.
        wallclock_time_limit: str
            The maximum amount of wall clock time that a worker can run for. This should
            be a string of the form `HH:MM` where HH is the number of hours and MM the number
            of minutes
        """

        super().__init__(number_of_threads, number_of_gpus, preferred_gpu_toolkit)

        self._per_thread_memory_limit = per_thread_memory_limit
        self._wallclock_time_limit = wallclock_time_limit

        assert self._per_thread_memory_limit is not None

        assert (isinstance(self._per_thread_memory_limit, unit.Quantity) and
                unit.get_base_units(unit.byte)[-1] == unit.get_base_units(self._per_thread_memory_limit.units)[-1])

        assert self._per_thread_memory_limit > 0 * unit.byte

        assert wallclock_time_limit is not None
        assert isinstance(wallclock_time_limit, str) and len(wallclock_time_limit) > 0

        wallclock_pattern = re.compile(r'\d\d:\d\d')
        assert wallclock_pattern.match(wallclock_time_limit) is not None

    def dict(self):
        return self.__getstate__()

    def __getstate__(self):

        base_dict = super(QueueWorkerResources, self).__getstate__()

        base_dict.update({
            'per_thread_memory_limit': self.number_of_threads,
            'wallclock_time_limit': self.number_of_threads,
        })

        return base_dict

    def __setstate__(self, state):
        super(QueueWorkerResources, self).__setstate__(state)

        self._per_thread_memory_limit = state['per_thread_memory_limit']
        self._wallclock_time_limit = state['wallclock_time_limit']

    def __eq__(self, other):
        return super(QueueWorkerResources, self).__eq__(other) and \
               self.per_thread_memory_limit == other.per_thread_memory_limit and \
               self.wallclock_time_limit == other.wallclock_time_limit

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
