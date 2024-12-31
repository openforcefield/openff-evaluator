"""
Defines the base API for defining new openff-evaluator estimation layers.
"""

import abc
import collections
import logging
from os import path

from openff.units import unit

from openff.evaluator.attributes import (
    UNDEFINED,
    Attribute,
    AttributeClass,
    PlaceholderValue,
)
from openff.evaluator.datasets import PhysicalProperty
from openff.evaluator.storage.data import BaseStoredData, StoredSimulationData
from openff.evaluator.utils.exceptions import EvaluatorException

logger = logging.getLogger(__name__)


def return_args(*args, **_):
    return args


class CalculationLayerResult(AttributeClass):
    """The result of attempting to estimate a property using
    a `CalculationLayer`.
    """

    physical_property = Attribute(
        docstring="The estimated property (if the layer was successful).",
        type_hint=PhysicalProperty,
        optional=True,
    )
    data_to_store = Attribute(
        docstring="Paths to the data objects to store.",
        type_hint=list,
        default_value=[],
    )

    exceptions = Attribute(
        docstring="Any exceptions raised by the layer while estimating the "
        "property.",
        type_hint=list,
        default_value=[],
    )

    def validate(self, attribute_type=None):
        super(CalculationLayerResult, self).validate(attribute_type)

        assert all(isinstance(x, (tuple, list)) for x in self.data_to_store)
        assert all(len(x) == 2 for x in self.data_to_store)
        assert all(all(isinstance(y, str) for y in x) for x in self.data_to_store)

        assert all(isinstance(x, EvaluatorException) for x in self.exceptions)


class CalculationLayerSchema(AttributeClass):
    """A schema which encodes the options that a `CalculationLayer`
    should use when estimating a given class of physical properties.
    """

    absolute_tolerance = Attribute(
        docstring="The absolute uncertainty that the property should "
        "be estimated to within. This attribute is mutually exclusive "
        "with the `relative_tolerance` attribute.",
        type_hint=unit.Quantity,
        default_value=UNDEFINED,
        optional=True,
    )
    relative_tolerance = Attribute(
        docstring="The relative uncertainty that the property should "
        "be estimated to within, i.e `relative_tolerance * "
        "measured_property.uncertainty`. This attribute is mutually "
        "exclusive with the `absolute_tolerance` attribute.",
        type_hint=float,
        default_value=UNDEFINED,
        optional=True,
    )

    def validate(self, attribute_type=None):
        if (
            self.absolute_tolerance != UNDEFINED
            and self.relative_tolerance != UNDEFINED
        ):
            raise ValueError(
                "Only one of `absolute_tolerance` and `relative_tolerance` "
                "can be set."
            )

        super(CalculationLayerSchema, self).validate(attribute_type)


class CalculationLayer(abc.ABC):
    """An abstract representation of a calculation layer whose goal is
    to estimate a set of physical properties using a single approach,
    such as a layer which employs direct simulations to estimate properties,
    or one which reweights cached simulation data to the same end.
    """

    @classmethod
    @abc.abstractmethod
    def required_schema_type(cls):
        """Returns the type of `CalculationLayerSchema` required by
        this layer.

        Returns
        -------
        type of CalculationLayerSchema
            The required schema type.
        """
        raise NotImplementedError()

    @staticmethod
    def _await_results(
        layer_name,
        calculation_backend,
        storage_backend,
        batch,
        callback,
        submitted_futures,
        synchronous=False,
    ):
        """A helper method to handle passing the results of this layer back to
        the main thread.

        Parameters
        ----------
        layer_name: str
            The name of the layer processing the results.
        calculation_backend: CalculationBackend
            The backend to the submit the calculations to.
        storage_backend: StorageBackend
            The backend used to store / retrieve data from previous calculations.
        batch: Batch
            The request object which spawned the awaited results.
        callback: function
            The function to call when the backend returns the results (or an error).
        submitted_futures: list of dask.distributed.Future
            A list of the futures returned by the backed when submitting the calculation.
        synchronous: bool
            If true, this function will block until the calculation has completed.
        """
        print("callback")
        print(submitted_futures)

        callback_future = calculation_backend.submit_task(
            return_args, *submitted_futures
        )
        print(callback_future)
        callback_future.result()

        def callback_wrapper(results_future):
            CalculationLayer._process_results(
                results_future, batch, layer_name, storage_backend, callback
            )

        if synchronous:
            callback_wrapper(callback_future)
        else:
            callback_future.add_done_callback(callback_wrapper)

    @classmethod
    def _store_cached_output(cls, batch, returned_output, storage_backend):
        """Stores any cached pieces of simulation data using a storage backend.

        Parameters
        ----------
        batch: Batch
            The request which generated the cached data.
        returned_output: CalculationLayerResult
            The layer result which contains the cached data.
        storage_backend: StorageBackend
            The backend to use to store the cached data.
        """

        for data_tuple in returned_output.data_to_store:
            data_object_path, data_directory_path = data_tuple

            # Make sure the data directory / file to store actually exists
            if not path.isdir(data_directory_path) or not path.isfile(data_object_path):
                logger.info(
                    f"Invalid data directory ({data_directory_path}) / "
                    f"file ({data_object_path})"
                )
                continue

            # Attach any extra metadata which is missing.
            data_object = BaseStoredData.from_json(data_object_path)

            if isinstance(data_object, StoredSimulationData):
                if isinstance(data_object.force_field_id, PlaceholderValue):
                    data_object.force_field_id = batch.force_field_id
                if isinstance(data_object.source_calculation_id, PlaceholderValue):
                    data_object.source_calculation_id = batch.id
                data_object.calculation_layer = cls.__name__

            storage_backend.store_object(data_object, data_directory_path)

    @staticmethod
    def _process_results(results_future, batch, layer_name, storage_backend, callback):
        """Processes the results of a calculation layer, updates the server request,
        then passes it back to the callback ready for propagation to the next layer
        in the stack.

        Parameters
        ----------
        results_future: distributed.Future
            The future object which will hold the results.
        batch: Batch
            The batch which spawned the awaited results.
        layer_name: str
            The name of the layer processing the results.
        storage_backend: StorageBackend
            The backend used to store / retrieve data from previous calculations.
        callback: function
            The function to call when the backend returns the results (or an error).
        """
        # Wrap everything in a try catch to make sure the whole calculation backend /
        # server doesn't go down when an unexpected exception occurs.
        try:
            results = list(results_future.result())

            if len(results) > 0 and isinstance(results[0], collections.abc.Iterable):
                results = results[0]

            results_future.release()

            for returned_output in results:
                cls._process_single_result(returned_output, batch, storage_backend)

        except Exception as e:
            logger.exception(f"Error processing layer results for request {batch.id}")
            exception = EvaluatorException.from_exception(e)

            batch.exceptions.append(exception)

        callback(batch)

    @classmethod
    @abc.abstractmethod
    def _schedule_calculation(
        cls, calculation_backend, storage_backend, layer_directory, batch
    ):
        """The implementation of the `schedule_calculation` method which is responsible
        for handling the main layer logic.

        Parameters
        ----------
        calculation_backend: CalculationBackend
            The backend to the submit the calculations to.
        storage_backend: StorageBackend
            The backend used to store / retrieve data from previous calculations.
        layer_directory: str
            The directory in which to store all temporary calculation data from this
            layer.
        batch: Batch
            The batch of properties to estimate with the layer.

        Returns
        -------
        list of Future
            The future objects which will yield the finished `CalculationLayerResult`
            objects.
        """
        raise NotImplementedError()

    @classmethod
    def schedule_calculation(
        cls,
        calculation_backend,
        storage_backend,
        layer_directory,
        batch,
        callback,
        synchronous=False,
    ):
        """Submit the proposed calculation to the backend of choice.

        Parameters
        ----------
        calculation_backend: CalculationBackend
            The backend to the submit the calculations to.
        storage_backend: StorageBackend
            The backend used to store / retrieve data from previous calculations.
        layer_directory: str
            The directory in which to store all temporary calculation data from this
            layer.
        batch: Batch
            The batch of properties to estimate with the layer.
        callback: function
            The function to call when the backend returns the results (or an error).
        synchronous: bool
            If true, this function will block until the calculation has completed.
            This is mainly intended for debugging purposes.
        """
        futures = cls._schedule_calculation(
            calculation_backend, storage_backend, layer_directory, batch
        )

        cls._await_results(
            calculation_backend,
            storage_backend,
            batch,
            callback,
            futures,
            synchronous,
        )
