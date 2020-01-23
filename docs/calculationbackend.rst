.. |calculation_backend|    replace:: :py:class:`~propertyestimator.backends.CalculationBackend`
.. |compute_resources|      replace:: :py:class:`~propertyestimator.backends.ComputeResources`

Calculation Backends
====================

A |calculation_backend| is an object used to distribute calculation tasks across available compute resources. This is
possible through specific backends which integrate with libraries such as `multiprocessing <https://docs.python.org/3.7
/library/multiprocessing.html>`__, `dask <https://distributed.dask.org/en/latest/>`_, `parsl <https://parsl-project.org
/>`_ and `cerlery <http://www.celeryproject.org/>`_.

Each backend is responsible for creating *compute workers*. A compute worker is an entity which has a set amount of
dedicated compute resources available to it and which can execute python functions using those resources. Calculation
backends may spawn multiple workers such that many tasks and calculations can be performed simultaneously.

A compute worker can be as simple as a new `multiprocessing <https://docs.python.org/3.7/library/multiprocessing.html#
the-process-class>`__ ``Process`` or something more complex like a `dask worker <https://distributed.dask.org/en/latest/
worker.html>`_. The resources available to a worker are described by the |compute_resources| object.

|calculation_backend| classes have a relatively simple structure::

    class MyCalculationBackend(CalculationBackend):

        def __init__(self, number_of_workers, resources_per_worker):
            ...

        def start(self):
            ...

        def stop(self):
            ...

        def submit_task(self, function, *args, **kwargs):
            ...

By default they implement a constructor which takes as input the number of workers that the backend should initially
spawn as well as the compute resources which are available to each. They must further implement:

* a ``start`` method which spawns the initial set of compute workers.
* a ``stop`` method which should kill all workers spawned by the backend as well as cleanup any temporary worker files.
* a ``submit_task`` method which takes a function to be execute by a worker, and a set of ``args`` and ``kwargs`` to
  pass to that function.

The ``submit_task`` must run asynchronously and return an `asyncio <https://docs.python.org/3/library/asyncio-future.
html>`_ ``Future`` object (or an object which implements the same API) when called, which can then be queried for when
the task has completed.

All calculation backends are implemented as context managers such that they can be used as::

    with MyCalculationBackend(number_of_workers=..., resources_per_worker...) as backend:
        backend.submit_task

where the ``start`` and ``stop`` methods will be called automatically.
