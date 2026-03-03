"""
Evaluator client side API.
"""

import copy
import json
import logging
import socket
import traceback
from collections import defaultdict
from enum import Enum
from time import sleep

from openff.evaluator.attributes import UNDEFINED, Attribute, AttributeClass
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.forcefield import (
    ForceFieldSource,
    FoyerForceFieldSource,
    LigParGenForceFieldSource,
    ParameterGradientKey,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.layers import (
    registered_calculation_layers,
    registered_calculation_schemas,
)
from openff.evaluator.layers.workflow import WorkflowCalculationSchema
from openff.evaluator.utils.exceptions import EvaluatorException
from openff.evaluator.utils.serialization import TypedJSONDecoder
from openff.evaluator.utils.tcp import (
    EvaluatorMessageTypes,
    pack_int,
    recvall,
    unpack_int,
)

logger = logging.getLogger(__name__)


class ConnectionOptions(AttributeClass):
    """The options to use when connecting to an `EvaluatorServer`"""

    server_address = Attribute(
        docstring="The address of the server to connect to.",
        type_hint=str,
        default_value="localhost",
    )
    server_port = Attribute(
        docstring="The port of the server to connect to.",
        type_hint=int,
        default_value=8000,
    )

    def __init__(self, server_address=None, server_port=None):
        """

        Parameters
        ----------
        server_address: str
            The address of the server to connect to.
        server_port: int
            The port of the server to connect to.
        """
        if server_address is not None:
            self.server_address = server_address
        if server_port is not None:
            self.server_port = server_port


class BatchMode(Enum):
    """The different modes in which a server can batch together properties
    to estimate.

    This enum may take values of

    * SameComponents: All properties measured for substances containing exactly
      the same components will be placed into a single batch. E.g. The density of
      a 80:20 and a 20:80 mix of ethanol and water would be batched together, but
      the density of pure ethanol and the density of pure water would be placed into
      separate batches.
    * SharedComponents: All properties measured for substances containing at least
      common component will be batched together. E.g.The densities of 80:20 and 20:80
      mixtures of ethanol and water, and the pure densities of ethanol and water would
      be batched together.
    * NoBatch: No batching will be performed. Each property will be estimated in a
      single, sequentially-increasing batch.

    Properties will only be marked as estimated by the server when all properties in a
    single batch are completed.
    """

    SameComponents = "SameComponents"
    SharedComponents = "SharedComponents"
    NoBatch = "NoBatch"


class Request(AttributeClass):
    """An estimation request which has been sent to a `EvaluatorServer`
    instance.

    This object can be used to query and retrieve the results of the
    request when finished, or be stored to retrieve the request at some
    point in the future."""

    id = Attribute(
        docstring="The unique id assigned to this request by the server.", type_hint=str
    )
    connection_options = Attribute(
        docstring="The options used to connect to the server handling the request.",
        type_hint=ConnectionOptions,
    )

    def __init__(self, client=None):
        """
        Parameters
        ----------
        client: EvaluatorClient, optional
            The client which submitted this request.
        """

        if client is not None:
            self.connection_options = ConnectionOptions()
            self.connection_options.server_address = client.server_address
            self.connection_options.server_port = client.server_port

        self._client = client

    def results(self, synchronous=False, polling_interval=5):
        """Attempt to retrieve the results of the request from the
        server.

        If the method is run synchronously it will block the main
        thread either all of the requested properties have been
        estimated, or an exception is returned.

        Parameters
        ----------
        synchronous: bool
            If `True`, this method will block the main thread until
            the server either returns a result or an error.
        polling_interval: float
            If running synchronously, this is the time interval (seconds)
            between checking if the calculation has finished. This will
            be ignored if running asynchronously.

        Returns
        -------
        RequestResult, optional
            Returns the current results of the request. This may
            be `None` if any unexpected exceptions occurred while
            retrieving the estimate.
        EvaluatorException, optional
            The exception raised will trying to retrieve the result
            if any.
        """
        if (
            self._client is None
            or self._client.server_address != self._client.server_address
            or self._client.server_port != self._client.server_port
        ):
            self.validate()
            self._client = EvaluatorClient(self.connection_options)

        return self._client.retrieve_results(self.id, synchronous, polling_interval)

    def __str__(self):
        return f"Request id={self.id}"

    def __repr__(self):
        return f"<{str(self)}>"


class RequestOptions(AttributeClass):
    """The options to use when requesting a set of physical
    properties be estimated by the server.
    """

    calculation_layers = Attribute(
        docstring="The calculation layers which may be used to "
        "estimate the set of physical properties. The order in which "
        "the layers appears in this list determines the order in which "
        "the layers will attempt to estimate the data set.",
        type_hint=list,
        default_value=["ReweightingLayer", "SimulationLayer"],
    )
    calculation_schemas = Attribute(
        docstring="The schemas that each calculation layer should "
        "use when estimating the set of physical properties. The "
        "dictionary should be of the form [property_type][layer_type].",
        type_hint=dict,
        optional=True,
    )

    batch_mode = Attribute(
        docstring="The way in which the server should batch together "
        "properties to estimate. Properties will only be marked as finished "
        "when all properties in a single batch are completed.",
        type_hint=BatchMode,
        default_value=BatchMode.SharedComponents,
        optional=True,
    )

    def add_schema(self, layer_type, property_type, schema):
        """A convenience function for adding a calculation schema
        to the schema dictionary.

        Parameters
        ----------
        layer_type: str or type of CalculationLayer
            The layer to associate the schema with.
        property_type: str or type of PhysicalProperty
            The class of property to associate the schema
            with.
        schema: CalculationSchema
            The schema to add.
        """

        # Validate the schema.
        schema.validate()

        # Make sure the schema is compatible with the layer.
        assert layer_type in registered_calculation_layers
        calculation_layer = registered_calculation_layers[layer_type]
        assert type(schema) is calculation_layer.required_schema_type()

        if isinstance(property_type, type):
            property_type = property_type.__name__

        if self.calculation_schemas == UNDEFINED:
            self.calculation_schemas = {}

        if property_type not in self.calculation_schemas:
            self.calculation_schemas[property_type] = {}
        if layer_type not in self.calculation_schemas[property_type]:
            self.calculation_schemas[property_type][layer_type] = {}

        self.calculation_schemas[property_type][layer_type] = schema

    def validate(self, attribute_type=None):
        super(RequestOptions, self).validate(attribute_type)

        assert all(isinstance(x, str) for x in self.calculation_layers)
        assert all(x in registered_calculation_layers for x in self.calculation_layers)

        if self.calculation_schemas != UNDEFINED:
            for property_type in self.calculation_schemas:
                assert isinstance(self.calculation_schemas[property_type], dict)

                for layer_type in self.calculation_schemas[property_type]:
                    assert layer_type in self.calculation_layers
                    calculation_layer = registered_calculation_layers[layer_type]

                    schema = self.calculation_schemas[property_type][layer_type]
                    required_type = calculation_layer.required_schema_type()
                    assert isinstance(schema, required_type)


class RequestResult(AttributeClass):
    """The current results of an estimation request - these
    results may be partial if the server hasn't yet completed
    the request.
    """

    queued_properties = Attribute(
        docstring="The set of properties which have yet to be, or "
        "are currently being estimated.",
        type_hint=PhysicalPropertyDataSet,
        default_value=PhysicalPropertyDataSet(),
    )

    estimated_properties = Attribute(
        docstring="The set of properties which have been successfully estimated.",
        type_hint=PhysicalPropertyDataSet,
        default_value=PhysicalPropertyDataSet(),
    )
    unsuccessful_properties = Attribute(
        docstring="The set of properties which could not be successfully estimated.",
        type_hint=PhysicalPropertyDataSet,
        default_value=PhysicalPropertyDataSet(),
    )

    equilibrated_properties = Attribute(
        docstring="The set of properties which have been successfully equilibrated.",
        type_hint=PhysicalPropertyDataSet,
        default_value=PhysicalPropertyDataSet(),
    )

    exceptions = Attribute(
        docstring="The set of properties which have yet to be, or "
        "are currently being estimated.",
        type_hint=list,
        default_value=[],
    )

    def validate(self, attribute_type=None):
        super(RequestResult, self).validate(attribute_type)
        assert all((isinstance(x, EvaluatorException) for x in self.exceptions))


class EvaluatorClient:
    """The object responsible for connecting to, and submitting
    physical property estimation requests to an `EvaluatorServer`.

    Examples
    --------
    These examples assume that an `EvaluatorServer` has been set up
    and is running (either synchronously or asynchronously). This
    server can be connect to be creating an `EvaluatorClient`:

    >>> from openff.evaluator.client import EvaluatorClient
    >>> client = EvaluatorClient()

    If the `EvaluatorServer` is not running on the local machine, you will
    need to specify its address and the port that it is listening on:

    >>> from openff.evaluator.client import ConnectionOptions
    >>>
    >>> connection_options = ConnectionOptions(server_address='server_address',
    >>>                                        server_port=8000)
    >>> client = EvaluatorClient(connection_options)

    To asynchronously submit a request to the running server using the default
    estimation options:

    >>> # Load in the data set of properties which will be used for comparisons
    >>> from openff.evaluator.datasets.thermoml import ThermoMLDataSet
    >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
    >>>
    >>> # Filter the dataset to only include densities measured between 130-260 K
    >>> from openff.units import unit
    >>> from openff.evaluator.properties import Density
    >>>
    >>> data_set.filter_by_property_types(Density)
    >>> data_set.filter_by_temperature(
    >>>     min_temperature=130*unit.kelvin,
    >>>     max_temperature=260*unit.kelvin
    >>> )
    >>>
    >>> # Load in the force field parameters
    >>> from openff.evaluator.forcefield import SmirnoffForceFieldSource
    >>> force_field_source = SmirnoffForceFieldSource.from_path('openff-2.2.1.offxml')
    >>>
    >>> # Submit the estimation request to a running server.
    >>> request = client.request_estimate(data_set, force_field_source)

    The status of the request can be asynchronously queried by calling

    >>> results = request.results()

    or the main thread can be blocked until the results are
    available by calling

    >>> results = request.results(synchronous=True)

    How the property set will be estimated can easily be controlled by passing a
    `RequestOptions` object to the estimate commands.

    The calculations layers which will be used to estimate the properties can be
    controlled for example like so:

    >>> from openff.evaluator.layers.reweighting import ReweightingLayer
    >>> from openff.evaluator.layers.simulation import SimulationLayer
    >>>
    >>> options = RequestOptions(calculation_layers=[
    >>>     "ReweightingLayer",
    >>>     "SimulationLayer"
    >>> ])
    >>>
    >>> request = client.request_estimate(data_set, force_field_source, options)

    Options for how properties should be estimated can be set on a per property, and per layer
    basis by providing a calculation schema to the options object.

    >>> from openff.evaluator.properties import DielectricConstant
    >>>
    >>> # Generate a schema to use when estimating densities directly
    >>> # from simulations.
    >>> density_simulation_schema = Density.default_simulation_schema()
    >>> # Generate a schema to use when estimating dielectric constants
    >>> # from cached simulation data.
    >>> dielectric_reweighting_schema = DielectricConstant.default_reweighting_schema()
    >>>
    >>> options.workflow_options = {
    >>>     'Density': {'SimulationLayer': density_simulation_schema},
    >>>     'Dielectric': {'SimulationLayer': dielectric_reweighting_schema}
    >>> }
    >>>
    >>> client.request_estimate(
    >>>     data_set,
    >>>     force_field_source,
    >>>     options,
    >>> )

    The gradients of the observables of interest with respect to a number of chosen
    parameters can be requested by passing a `parameter_gradient_keys` parameter.
    In the below example, gradients will be calculated with respect to both the
    bond length parameter for the [#6:1]-[#8:2] chemical environment, and the bond
    angle parameter for the [*:1]-[#8:2]-[*:3] chemical environment:

    >>> from openff.evaluator.forcefield import ParameterGradientKey
    >>>
    >>> parameter_gradient_keys = [
    >>>     ParameterGradientKey('Bonds', '[#6:1]-[#8:2]', 'length')
    >>>     ParameterGradientKey('Angles', '[*:1]-[#8:2]-[*:3]', 'angle')
    >>> ]
    >>>
    >>> client.request_estimate(
    >>>     data_set,
    >>>     force_field_source,
    >>>     options,
    >>>     parameter_gradient_keys
    >>> )
    """

    class _Submission(AttributeClass):
        """The data packet encoding an estimation request which will be sent to
        the server.
        """

        dataset = Attribute(
            docstring="The set of properties to estimate.",
            type_hint=PhysicalPropertyDataSet,
        )
        options = Attribute(
            docstring="The options to use when estimating the dataset.",
            type_hint=RequestOptions,
        )
        force_field_source = Attribute(
            docstring="The force field parameters to estimate the dataset using.",
            type_hint=ForceFieldSource,
        )
        parameter_gradient_keys = Attribute(
            docstring="A list of the parameters that the physical properties "
            "should be differentiated with respect to.",
            type_hint=list,
        )

        def validate(self, attribute_type=None):
            super(EvaluatorClient._Submission, self).validate(attribute_type)
            assert all(
                isinstance(x, ParameterGradientKey)
                for x in self.parameter_gradient_keys
            )

    @property
    def server_address(self):
        """str: The address of the server that this client is connected to."""
        return self._connection_options.server_address

    @property
    def server_port(self):
        """int: The port of the server that this client is connected to."""
        return self._connection_options.server_port

    def __init__(self, connection_options=None):
        """
        Parameters
        ----------
        connection_options: ConnectionOptions, optional
            The options used when connecting to the calculation
            server. If `None`, default options are used.
        """

        if connection_options is None:
            connection_options = ConnectionOptions()

        if connection_options.server_address is None:
            raise ValueError(
                "The address of the server which will run"
                "these calculations must be given."
            )

        self._connection_options = connection_options

    @staticmethod
    def default_request_options(data_set, force_field_source):
        """Returns the default `RequestOptions` options used
        to estimate a set of properties if `None` are provided.

        Parameters
        ----------
        data_set: PhysicalPropertyDataSet
            The data set which would be estimated.
        force_field_source: ForceFieldSource
            The force field parameters which will be used by the
            request.

        Returns
        -------
        RequestOptions
            The default options.
        """
        options = RequestOptions()
        EvaluatorClient._populate_request_options(options, data_set, force_field_source)

        return options

    @staticmethod
    def _default_protocol_replacements(force_field_source):
        """Returns the default set of protocols in a workflow to replace
        with different types. This is mainly to handle replacing the base
        force field assignment protocol with one specific to the force field
        source.

        Parameters
        ----------
        force_field_source: ForceFieldSource
            The force field parameters which will be used by the
            request.

        Returns
        -------
        dict of str and str
            A map between the type of protocol to replace, and the type of
            protocol to use in its place.
        """

        replacements = {}

        if isinstance(force_field_source, SmirnoffForceFieldSource):
            replacements["BaseBuildSystem"] = "BuildSmirnoffSystem"
        elif isinstance(force_field_source, LigParGenForceFieldSource):
            replacements["BaseBuildSystem"] = "BuildLigParGenSystem"
        elif isinstance(force_field_source, TLeapForceFieldSource):
            replacements["BaseBuildSystem"] = "BuildTLeapSystem"
        elif isinstance(force_field_source, FoyerForceFieldSource):
            replacements["BaseBuildSystem"] = "BuildFoyerSystem"

        return replacements

    @staticmethod
    def _populate_request_options(options, data_set, force_field_source):
        """Populates any missing attributes of a `RequestOptions`
        object with default values registered via the plug-in
        system.

        Parameters
        ----------
        options: RequestOptions
            The object to populate with defaults.
        data_set: PhysicalPropertyDataSet
            The data set to be estimated using the options.
        force_field_source: ForceFieldSource
            The force field parameters which will be used by the
            request.
        """

        # Retrieve the types of properties in the data set.
        property_types = data_set.property_types

        if options.calculation_schemas == UNDEFINED:
            options.calculation_schemas = defaultdict(dict)

        properties_without_schemas = set(property_types)

        for property_type in options.calculation_schemas:
            if property_type not in properties_without_schemas:
                continue

            properties_without_schemas.remove(property_type)

        # Assign default calculation schemas in the cases where the user
        # hasn't provided one.
        for calculation_layer in options.calculation_layers:
            for property_type in property_types:
                # Check if the user has already provided a schema.
                existing_schema = options.calculation_schemas.get(
                    property_type, {}
                ).get(calculation_layer, None)

                if existing_schema is not None:
                    continue

                # Check if this layer has any registered schemas.
                if calculation_layer not in registered_calculation_schemas:
                    continue

                default_layer_schemas = registered_calculation_schemas[
                    calculation_layer
                ]

                # Check if this property type has any registered schemas for
                # the given calculation layer.
                if property_type not in default_layer_schemas:
                    continue

                # noinspection PyTypeChecker
                default_schema = default_layer_schemas[property_type]

                if callable(default_schema):
                    default_schema = default_schema()

                # Mark this property as having at least one registered
                # calculation schema.
                if property_type in properties_without_schemas:
                    properties_without_schemas.remove(property_type)

                if property_type not in options.calculation_schemas:
                    options.calculation_schemas[property_type] = {}

                options.calculation_schemas[property_type][
                    calculation_layer
                ] = default_schema

        # Make sure all property types have at least one registered
        # calculation schema.
        if len(properties_without_schemas) >= 1:
            type_string = ", ".join(properties_without_schemas)

            raise ValueError(
                f"No calculation schema could be found for "
                f"the {type_string} properties."
            )

        # Perform any protocol type replacements
        replacement_types = EvaluatorClient._default_protocol_replacements(
            force_field_source
        )

        for calculation_layer in options.calculation_layers:
            for property_type in property_types:
                # Check if the user has already provided a schema.
                if (
                    property_type not in options.calculation_schemas
                    or calculation_layer
                    not in options.calculation_schemas[property_type]
                ):
                    continue

                schema = options.calculation_schemas[property_type][calculation_layer]

                if not isinstance(schema, WorkflowCalculationSchema):
                    continue

                workflow_schema = schema.workflow_schema
                workflow_schema.replace_protocol_types(replacement_types)

    def request_estimate(
        self,
        property_set,
        force_field_source,
        options=None,
        parameter_gradient_keys=None,
    ):
        """Submits a request for the `EvaluatorServer` to attempt to estimate
        the data set of physical properties using the specified force field
        parameters according to the provided options.

        Parameters
        ----------
        property_set : PhysicalPropertyDataSet
            The set of properties to estimate.
        force_field_source : ForceFieldSource or openff.toolkit.typing.engines.smirnoff.ForceField
            The force field parameters to estimate the properties using.
        options : RequestOptions, optional
            A set of evaluator options. If `None` default options
            will be used (see `default_request_options`).
        parameter_gradient_keys: list of ParameterGradientKey, optional
            A list of the parameters that the physical properties should
            be differentiated with respect to.

        Returns
        -------
        Request
            An object which will provide access to the
            results of this request.
        EvaluatorException, optional
            Any exceptions raised while attempting the submit the request.
        """
        from openff.toolkit.typing.engines import smirnoff

        if property_set is None or force_field_source is None:
            raise ValueError(
                "Both a data set and force field source must be "
                "present to compute physical properties."
            )

        if parameter_gradient_keys is None:
            parameter_gradient_keys = []

        # Handle the conversion of a SMIRNOFF force field object
        # for backwards compatibility.
        if isinstance(force_field_source, smirnoff.ForceField):
            force_field_source = SmirnoffForceFieldSource.from_object(
                force_field_source
            )

        # Fill in any missing options with default values
        if options is None:
            options = self.default_request_options(property_set, force_field_source)
        else:
            options = copy.deepcopy(options)
            self._populate_request_options(options, property_set, force_field_source)

        # Make sure the options are valid.
        options.validate()

        # Build the submission object.
        submission = EvaluatorClient._Submission()
        submission.dataset = property_set
        submission.force_field_source = force_field_source
        submission.options = options
        submission.parameter_gradient_keys = parameter_gradient_keys

        # Ensure the submission is valid.
        submission.validate()

        # Send the submission to the server.
        request_id, error = self._send_calculations_to_server(submission)

        # Build the object which represents this request.
        request_object = None

        if error is None:
            request_object = Request(self)
            request_object.id = request_id

        return request_object, error

    def retrieve_results(self, request_id, synchronous=False, polling_interval=5):
        """Retrieves the current results of a request from the server.

        Parameters
        ----------
        request_id: str
            The server assigned id of the request.
        synchronous: bool
            If true, this method will block the main thread until the server
            either returns a result or an error.
        polling_interval: float
            If running synchronously, this is the time interval (seconds)
            between checking if the request has completed.

        Returns
        -------
        RequestResult, optional
            Returns the current results of the request. This may
            be `None` if any unexpected exceptions occurred while
            retrieving the estimate.
        EvaluatorException, optional
            The exception raised will trying to retrieve the result,
            if any.
        """

        # If running asynchronously, just return whatever the server
        # sends back.

        if synchronous is False:
            return self._send_query_to_server(request_id)

        assert polling_interval >= 0

        response = None
        error = None

        should_run = True

        while should_run:
            if polling_interval > 0:
                sleep(polling_interval)

            response, error = self._send_query_to_server(request_id)

            if (
                isinstance(response, RequestResult)
                and len(response.queued_properties) > 0
            ):
                logger.info(
                    f"{request_id} --- "
                    f"# queued_properties: {len(response.queued_properties):4d} "
                    f"# estimated_properties: {len(response.estimated_properties):4d} "
                    f"# unsuccessful_properties: {len(response.unsuccessful_properties):4d} "
                    f"# equilibrated_properties: {len(response.equilibrated_properties):4d} "
                    f"# exceptions: {len(response.exceptions):4d}"
                )
                continue

            logger.info(f"The server has completed request {request_id}.")
            should_run = False

        return response, error

    def _send_calculations_to_server(self, submission):
        """Attempts to connect to the calculation server, and
        submit the requested calculations.

        Parameters
        ----------
        submission: _Submission
            The jobs to submit.

        Returns
        -------
        str, optional:
           The id which the server has assigned the submitted calculations.
           This can be used to query the server for when the calculation
           has completed.

           Returns None if the calculation could not be submitted.
        EvaluatorException, optional
            Any exceptions raised while attempting the submit the request.
        """

        # Attempt to establish a connection to the server.
        connection_settings = (
            self._connection_options.server_address,
            self._connection_options.server_port,
        )

        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect(connection_settings)

        request_id = None

        try:
            # Encode the submission json into an encoded
            # packet ready to submit to the server.
            message_type = pack_int(EvaluatorMessageTypes.Submission)

            encoded_json = submission.json().encode()
            length = pack_int(len(encoded_json))

            connection.sendall(message_type + length + encoded_json)

            # Wait for confirmation that the server has received
            # the jobs.
            header = recvall(connection, 4)
            length = unpack_int(header)[0]

            # Decode the response from the server. If everything
            # went well, this should be the id of the submitted
            # calculations.
            encoded_json = recvall(connection, length)
            request_id, error = json.loads(encoded_json.decode(), cls=TypedJSONDecoder)

        except Exception as e:
            trace = traceback.format_exception(None, e, e.__traceback__)
            error = EvaluatorException(message=trace)

        finally:
            if connection is not None:
                connection.close()

        # Return the ids of the submitted jobs.
        return request_id, error

    def _send_query_to_server(self, request_id):
        """Attempts to connect to the calculation server, and
        submit the requested calculations.

        Parameters
        ----------
        request_id: str
            The id of the job to query.

        Returns
        -------
        str, optional:
           The status of the submitted job.
           Returns None if the calculation has not yet completed.
        """
        server_response = None

        # Attempt to establish a connection to the server.
        connection_settings = (
            self._connection_options.server_address,
            self._connection_options.server_port,
        )

        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect(connection_settings)

        try:
            # Encode the request id into the message.
            message_type = pack_int(EvaluatorMessageTypes.Query)

            encoded_request_id = request_id.encode()
            length = pack_int(len(encoded_request_id))

            connection.sendall(message_type + length + encoded_request_id)

            # Wait for the server response.
            header = recvall(connection, 4)
            length = unpack_int(header)[0]

            # Decode the response from the server. If everything
            # went well, this should be the finished calculation.
            if length > 0:
                encoded_json = recvall(connection, length)
                server_response = encoded_json.decode()

        finally:
            if connection is not None:
                connection.close()

        response = None
        error = None

        if server_response is not None:
            response, error = json.loads(server_response, cls=TypedJSONDecoder)

        return response, error
