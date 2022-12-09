"""
Defines the base API for the openff-evaluator task calculation backend.
"""
import abc
import re
from enum import Enum

from openff.units import unit


class ComputeResources:
    """An object which stores how many of each type of computational
    resource (threads or gpu's) is available to a calculation worker.
    """

    class GPUToolkit(Enum):
        """An enumeration of the different GPU toolkits to
        make available to different calculations.
        """

        CUDA = "CUDA"
        OpenCL = "OpenCL"
        auto = "auto"

    class GPUPrecision(Enum):
        """An enumeration of the different precision for GPU calculations."""

        single = "single"
        mixed = "mixed"
        double = "double"

    @property
    def number_of_threads(self):
        """int: The number of threads available to a calculation worker."""
        return self._number_of_threads

    @property
    def number_of_gpus(self):
        """int: The number of GPUs available to a calculation worker."""
        return self._number_of_gpus

    @property
    def preferred_gpu_toolkit(self):
        """ComputeResources.GPUToolkit: The preferred toolkit to use when running on GPUs."""
        return self._preferred_gpu_toolkit

    @property
    def preferred_gpu_precision(self):
        """ComputeResources.GPUPrecision: The preferred precision level to use when running on GPUs."""
        return self._preferred_gpu_precision

    @property
    def gpu_device_indices(self):
        """str: The indices of the GPUs to run on. This is purely an internal
        implementation detail and should not be relied upon externally."""
        return self._gpu_device_indices

    def __init__(
        self,
        number_of_threads=1,
        number_of_gpus=0,
        preferred_gpu_toolkit=GPUToolkit.auto,
        preferred_gpu_precision=GPUPrecision.mixed,
    ):
        """Constructs a new ComputeResources object.

        Parameters
        ----------
        number_of_threads: int
            The number of threads available to a calculation worker.
        number_of_gpus: int
            The number of GPUs available to a calculation worker.
        preferred_gpu_toolkit: ComputeResources.GPUToolkit, optional
            The preferred toolkit to use when running on GPUs.
        preferred_gpu_precision: ComputeResources.GPUPrecision, optional
            The preferred precision level to use when runnin on GPUs.
        """

        self._number_of_threads = number_of_threads
        self._number_of_gpus = number_of_gpus

        self._preferred_gpu_toolkit = preferred_gpu_toolkit
        self._preferred_gpu_precision = preferred_gpu_precision
        # A workaround for when using a local cluster backend which is
        # strictly for internal purposes only for now.
        self._gpu_device_indices = None

        assert self._number_of_threads >= 0
        assert self._number_of_gpus >= 0

        assert self._number_of_threads > 0 or self._number_of_gpus > 0

        if self._number_of_gpus > 0:
            assert self._preferred_gpu_toolkit is not None
            assert self._number_of_gpus == 1, f'only 1 gpu per worker is supported'

    def __getstate__(self):
        return {
            "number_of_threads": self.number_of_threads,
            "number_of_gpus": self.number_of_gpus,
            "preferred_gpu_toolkit": self.preferred_gpu_toolkit,
            "preferred_gpu_precision": self.preferred_gpu_precision,
            "_gpu_device_indices": self._gpu_device_indices,
        }

    def __setstate__(self, state):

        self._number_of_threads = state["number_of_threads"]
        self._number_of_gpus = state["number_of_gpus"]
        self._preferred_gpu_toolkit = state["preferred_gpu_toolkit"]
        self._preferred_gpu_precision = state["preferred_gpu_precision"]
        self._gpu_device_indices = state["_gpu_device_indices"]

    def __eq__(self, other):
        return (
            self.number_of_threads == other.number_of_threads
            and self.number_of_gpus == other.number_of_gpus
            and self.preferred_gpu_toolkit == other.preferred_gpu_toolkit
            and self.preferred_gpu_precision == other.preferred_gpu_precision
        )

    def __ne__(self, other):
        return not self.__eq__(other)


class QueueWorkerResources(ComputeResources):
    """An extended resource object with properties specific
    to calculations which will run on queue based resources,
    such as LSF, PBS or SLURM.
    """

    @property
    def per_thread_memory_limit(self):
        """openmm.unit.Quantity: The maximum amount of memory available to each thread, such that
        the total memory limit will be `per_cpu_memory_limit * number_of_threads`."""
        return self._per_thread_memory_limit

    @property
    def wallclock_time_limit(self):
        """str: The maximum amount of wall clock time that a worker can run for. This should
        be a string of the form `HH:MM` where HH is the number of hours and MM the number of minutes"""
        return self._wallclock_time_limit

    def __init__(
        self,
        number_of_threads=1,
        number_of_gpus=0,
        preferred_gpu_toolkit=None,
        preferred_gpu_precision=None,
        per_thread_memory_limit=1 * unit.gigabytes,
        wallclock_time_limit="01:00",
    ):
        """Constructs a new ComputeResources object.

        Notes
        -----
        Both the requested `number_of_threads` and the `number_of_gpus` must be less than
        or equal to the number of threads (/cpus/cores) and GPUs available to each compute
        node in the cluster respectively, such that a single worker is able to be accommodated
        by a single compute node.

        Parameters
        ----------
        per_thread_memory_limit: openmm.unit.Quantity
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

        assert (
            isinstance(self._per_thread_memory_limit, unit.Quantity)
            and unit.get_base_units(unit.byte)[-1]
            == unit.get_base_units(self._per_thread_memory_limit.units)[-1]
        )

        assert self._per_thread_memory_limit > 0 * unit.byte

        assert wallclock_time_limit is not None
        assert isinstance(wallclock_time_limit, str) and len(wallclock_time_limit) > 0

        wallclock_pattern = re.compile(r"\d\d:\d\d")
        assert wallclock_pattern.match(wallclock_time_limit) is not None

    def __getstate__(self):

        base_dict = super(QueueWorkerResources, self).__getstate__()

        base_dict.update(
            {
                "per_thread_memory_limit": self.number_of_threads,
                "wallclock_time_limit": self.number_of_threads,
            }
        )

        return base_dict

    def __setstate__(self, state):
        super(QueueWorkerResources, self).__setstate__(state)

        self._per_thread_memory_limit = state["per_thread_memory_limit"]
        self._wallclock_time_limit = state["wallclock_time_limit"]

    def __eq__(self, other):
        return (
            super(QueueWorkerResources, self).__eq__(other)
            and self.per_thread_memory_limit == other.per_thread_memory_limit
            and self.wallclock_time_limit == other.wallclock_time_limit
        )

    def __ne__(self, other):
        return not self.__eq__(other)


class CalculationBackend(abc.ABC):
    """An abstract base representation of an openff-evaluator calculation backend. A backend is
    responsible for coordinating, distributing and running calculations on the
    available hardware. This may range from a single machine to a multinode cluster,
    but *not* across multiple cluster or physical locations.

    Notes
    -----
    All estimator backend classes must inherit from this class, and must implement the
    `start`, `stop`, and `submit_task` method.
    """

    @property
    def started(self):
        """bool: Returns whether this backend has been started yet."""
        return self._started

    def __init__(self, number_of_workers=1, resources_per_worker=None):

        """Constructs a new CalculationBackend object.

        Parameters
        ----------
        number_of_workers : int
            The number of works to run the calculations on. One worker
            can perform a single task (e.g run a simulation) at once.
        resources_per_worker: ComputeResources, optional
            The number of resources to request per worker.
        """

        if resources_per_worker is None:
            resources_per_worker = ComputeResources()

        self._number_of_workers = number_of_workers
        self._resources_per_worker = resources_per_worker

        self._started = False

    def start(self):
        """Start the calculation backend."""

        if self._started:
            raise RuntimeError("The backend has already been started.")

        self._started = True

    @abc.abstractmethod
    def stop(self):
        """Stop the calculation backend."""
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):

        if self._started:
            self.stop()
