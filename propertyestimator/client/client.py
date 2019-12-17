"""
Property estimator client side API.
"""

import json
import logging
import traceback
import typing
from collections import defaultdict
from time import sleep

from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.tcpclient import TCPClient

from propertyestimator.attributes import UNDEFINED, Attribute, AttributeClass
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.layers import (
    registered_calculation_layers,
    registered_calculation_schemas,
)
from propertyestimator.layers.workflow import WorkflowCalculationSchema
from propertyestimator.utils.exceptions import EvaluatorException
from propertyestimator.utils.serialization import TypedBaseModel, TypedJSONDecoder
from propertyestimator.utils.tcp import (
    PropertyEstimatorMessageTypes,
    pack_int,
    unpack_int,
)


class ConnectionOptions(AttributeClass):
    """The options to use when connecting to an `EvaluatorServer`
    """

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
        polling_interval: int
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
        "estimate the set of physical properties.",
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

    def validate(self, attribute_type=None):

        super(RequestOptions, self).validate()

        assert all(isinstance(x, str) for x in self.calculation_layers)
        assert all(x in registered_calculation_layers for x in self.calculation_layers)

        if self.calculation_schemas != UNDEFINED:

            for property_type in self.calculation_layers:

                assert isinstance(self.calculation_layers[property_type], dict)

                for layer_type in self.calculation_layers[property_type]:

                    assert layer_type in self.calculation_layers
                    calculation_layer = registered_calculation_layers[layer_type]

                    schemas = self.calculation_layers[property_type][layer_type]
                    required_type = calculation_layer.required_schema_type()
                    assert all(isinstance(x, required_type) for x in schemas)


class RequestResult(AttributeClass):
    """The current results of an estimation request - these
    results may be partial if the server hasn't yet completed
    the request.
    """

    queued_properties = Attribute(
        docstring="The set of properties which have yet to be, or "
        "are currently being estimated.",
        type_hint=typing.Union[PhysicalPropertyDataSet, None],
        default_value=None,
    )

    estimated_properties = Attribute(
        docstring="The set of properties which have been successfully estimated.",
        type_hint=typing.Union[PhysicalPropertyDataSet, None],
        default_value=None,
    )
    unsuccessful_properties = Attribute(
        docstring="The set of properties which could not be successfully estimated.",
        type_hint=typing.Union[PhysicalPropertyDataSet, None],
        default_value=None,
    )

    exceptions = Attribute(
        docstring="The set of properties which have yet to be, or "
        "are currently being estimated.",
        type_hint=list,
        default_value=[],
    )

    def validate(self, attribute_type=None):

        super(RequestResult, self).validate()

        assert all((isinstance(x, EvaluatorException) for x in self.exceptions))
        assert (
            self.queued_properties is not None
            or self.estimated_properties is not None
            or self.unsuccessful_properties is not None
        )


class EvaluatorClient:
    """The object responsible for connecting to, and submitting
    physical property estimation requests to an `EvaluatorServer`.

    Examples
    --------
    Setting up the client instance:

    >>> from propertyestimator.client import EvaluatorClient
    >>> property_estimator = EvaluatorClient()

    If the EvaluatorServer is not running on the local machine, you will
    need to specify its address and the port that it is listening on:

    >>> from propertyestimator.client import ConnectionOptions
    >>>
    >>> connection_options = ConnectionOptions(server_address='server_address',
    >>>                                                         server_port=8000)
    >>> property_estimator = EvaluatorClient(connection_options)

    To asynchronously submit a request to the running server using the default estimator
    options:

    >>> # Load in the data set of properties which will be used for comparisons
    >>> from propertyestimator.datasets.thermoml import ThermoMLDataSet
    >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
    >>> # Filter the dataset to only include densities measured between 130-260 K
    >>> from propertyestimator import unit
    >>> from propertyestimator.properties import Density
    >>>
    >>> data_set.filter_by_property_types(Density)
    >>> data_set.filter_by_temperature(min_temperature=130*unit.kelvin, max_temperature=260*unit.kelvin)
    >>>
    >>> # Load in the force field parameters
    >>> from openforcefield.typing.engines import smirnoff
    >>> from propertyestimator.forcefield import SmirnoffForceFieldSource
    >>> smirnoff_force_field = smirnoff.ForceField('smirnoff99Frosst-1.1.0.offxml')
    >>> force_field_source = SmirnoffForceFieldSource.from_object(smirnoff_force_field)
    >>>
    >>> request = property_estimator.request_estimate(data_set, force_field_source)

    The status of the request can be asynchronously queried by calling

    >>> results = request.results()

    or the main thread can be blocked until the results are
    available by calling

    >>> results = request.results(synchronous=True)

    How the property set will be estimated can easily be controlled by passing a
    RequestOptions object to the estimate commands.

    The calculations layers which will be used to estimate the properties can be
    controlled for example like so:

    >>> from propertyestimator.layers.reweighting import ReweightingLayer
    >>> from propertyestimator.layers.simulation import SimulationLayer
    >>>
    >>> options = RequestOptions(allowed_calculation_layers = [ReweightingLayer,
    >>>                                                                  SimulationLayer])
    >>>
    >>> request = property_estimator.request_estimate(data_set, force_field_source, options)

    Options for how properties should be estimated can be set on a per property, and per layer
    basis. For example, the relative uncertainty that properties should estimated to within by
    the SimulationLayer can be set as:

    >>> from propertyestimator.workflow import WorkflowOptions
    >>>
    >>> workflow_options = WorkflowOptions(WorkflowOptions.ConvergenceMode.RelativeUncertainty,
    >>>                                    relative_uncertainty_fraction=0.1)
    >>> options.workflow_options = {
    >>>     'Density': {'SimulationLayer': workflow_options},
    >>>     'Dielectric': {'SimulationLayer': workflow_options}
    >>> }

    Or alternatively, as absolute uncertainty tolerance can be set as:

    >>> density_options = WorkflowOptions(WorkflowOptions.ConvergenceMode.AbsoluteUncertainty,
    >>>                                   absolute_uncertainty=0.0002 * unit.gram / unit.milliliter)
    >>> dielectric_options = WorkflowOptions(WorkflowOptions.ConvergenceMode.AbsoluteUncertainty,
    >>>                                      absolute_uncertainty=0.02 * unit.dimensionless)
    >>>
    >>> options.workflow_options = {
    >>>     'Density': {'SimulationLayer': density_options},
    >>>     'Dielectric': {'SimulationLayer': dielectric_options}
    >>> }

    The gradients of the observables of interest with respect to a number of chosen
    parameters can be requested by passing a `parameter_gradient_keys` parameter.
    In the below example, gradients will be calculated with respect to both the
    bond length parameter for the [#6:1]-[#8:2] chemical environment, and the bond
    angle parameter for the [*:1]-[#8:2]-[*:3] chemical environment:

    >>> from propertyestimator.forcefield import ParameterGradientKey
    >>>
    >>> parameter_gradient_keys = [
    >>>     ParameterGradientKey('Bonds', '[#6:1]-[#8:2]', 'length')
    >>>     ParameterGradientKey('Angles', '[*:1]-[#8:2]-[*:3]', 'angle')
    >>> ]
    >>>
    >>> request = property_estimator.request_estimate(data_set, force_field_source, options, parameter_gradient_keys)
    >>>
    """

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
        self._tcp_client = TCPClient()

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
        force_field_source : ForceFieldSource or openforcefield.typing.engines.smirnoff.ForceField
            The force field parameters to estimate the properties using.
        options : RequestOptions, optional
            A set of estimator options. If `None` default options
            will be used.
        parameter_gradient_keys: list of ParameterGradientKey, optional
            A list of the parameters that the physical properties should
            be differentiated with respect to.

        Returns
        -------
        Request
            An object which will provide access the the results of the request.
        """
        from openforcefield.typing.engines import smirnoff

        if property_set is None or force_field_source is None:

            raise ValueError(
                "Both a data set and force field source must be "
                "present to compute physical properties."
            )

        if options is None:
            options = RequestOptions()

        if isinstance(force_field_source, smirnoff.ForceField):
            # Handle conversion of the force field object for
            # backwards compatibility.
            force_field_source = SmirnoffForceFieldSource.from_object(
                force_field_source
            )

        if len(options.allowed_calculation_layers) == 0:
            raise ValueError("A submission contains no allowed calculation layers.")

        properties_list = []
        property_types = set()

        # Refactor the properties into a list, and extract the types
        # of properties to be estimated (e.g 'Denisty', 'DielectricConstant').
        for substance_tag in property_set.properties:

            for physical_property in property_set.properties[substance_tag]:

                properties_list.append(physical_property)
                property_types.add(physical_property.__class__.__name__)

        # type_name = type(physical_property).__name__
        #
        # if type_name not in registered_properties:
        #     raise ValueError(
        #         f"The property estimator does not support {type_name} properties."
        #     )
        #
        # if type_name in property_types:
        #     continue

        if options.workflow_options is None:
            options.workflow_options = defaultdict(dict)
        if options.workflow_schemas is None:
            options.workflow_schemas = defaultdict(dict)

        properties_without_schemas = {*property_types}

        # Assign default calculation schemas in the cases where the user
        # hasn't provided one, and validate all of the schemas to be used
        # in the estimation.
        for property_type in property_types:

            for calculation_layer in options.allowed_calculation_layers:

                if (
                    calculation_layer not in registered_calculation_schemas
                    or property_type
                    not in registered_calculation_schemas[calculation_layer]
                ):
                    continue

                if property_type in properties_without_schemas:
                    # Mark this property as having at least one registered
                    # calculation schema.
                    properties_without_schemas.remove(property_type)

                # TODO: When refactoring the main data models clean-up this mess.
                # Set a default schema with default options if none have been
                # provided.
                if (
                    property_type not in options.workflow_options
                    or calculation_layer not in options.workflow_schemas[property_type]
                    or options.workflow_schemas[property_type][calculation_layer]
                    is None
                ):

                    default_schema = registered_calculation_schemas[calculation_layer][
                        property_type
                    ]

                    if callable(default_schema):
                        default_schema = default_schema()

                    options.workflow_schemas[property_type][
                        calculation_layer
                    ] = default_schema

                if (
                    property_type not in options.workflow_options
                    or calculation_layer not in options.workflow_options[property_type]
                    or options.workflow_options[property_type][calculation_layer]
                    is None
                ):

                    workflow_options = options.workflow_schemas[property_type][
                        calculation_layer
                    ].workflow_options

                    options.workflow_options[property_type][
                        calculation_layer
                    ] = workflow_options

                calculation_schema = options.workflow_schemas[property_type][
                    calculation_layer
                ]
                calculation_schema.validate()

                if not isinstance(calculation_schema, WorkflowCalculationSchema):
                    continue

                # Handle the cases where some protocol types should be replaced with
                # others.
                # TODO: Fix.
                calculation_schema.workflow_schema.replace_protocol_types(
                    calculation_schema.workflow_options.protocol_replacements
                )

        submission = _Submission(
            properties=properties_list,
            force_field_source=force_field_source,
            options=options,
            parameter_gradient_keys=parameter_gradient_keys,
        )

        request_id = IOLoop.current().run_sync(
            lambda: self._send_calculations_to_server(submission)
        )

        request_object = EvaluatorClient.Request(
            request_id, self._connection_options, self
        )

        return request_object

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

            return IOLoop.current().run_sync(
                lambda: self._send_query_to_server(request_id)
            )

        assert polling_interval >= 0

        response = None
        should_run = True

        while should_run:

            if polling_interval > 0:
                sleep(polling_interval)

            response = IOLoop.current().run_sync(
                lambda: self._send_query_to_server(request_id)
            )

            if (
                isinstance(response, RequestResult)
                and len(response.queued_properties) > 0
            ):
                continue

            logging.info(f"The server has completed request {request_id}.")
            should_run = False

        return response

    async def _send_calculations_to_server(self, submission):
        """Attempts to connect to the calculation server, and
        submit the requested calculations.

        Notes
        -----

        This method is based on the StackOverflow response from
        A. Jesse Jiryu Davis: https://stackoverflow.com/a/40257248

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
        """
        request_id = None

        try:

            # Attempt to establish a connection to the server.
            stream = await self._tcp_client.connect(
                self._connection_options.server_address,
                self._connection_options.server_port,
            )

            stream.set_nodelay(True)

            # Encode the submission json into an encoded
            # packet ready to submit to the server.
            message_type = pack_int(PropertyEstimatorMessageTypes.Submission)

            encoded_json = submission.json().encode()
            length = pack_int(len(encoded_json))

            await stream.write(message_type + length + encoded_json)

            # Wait for confirmation that the server has received
            # the jobs.
            header = await stream.read_bytes(4)
            length = unpack_int(header)[0]

            # Decode the response from the server. If everything
            # went well, this should be the id of the submitted
            # calculations.
            encoded_json = await stream.read_bytes(length)
            request_id = json.loads(encoded_json.decode())

            stream.close()
            self._tcp_client.close()

        except StreamClosedError as e:

            # Handle no connections to the server gracefully.
            logging.info(
                f"Error connecting to {self.server_address}:{self.server_port}. "
                f"Please ensure the server is running and that the server address "
                f"/ port is correct."
            )

            formatted_exception = traceback.format_exception(None, e, e.__traceback__)
            logging.info(formatted_exception)

        # Return the ids of the submitted jobs.
        return request_id

    async def _send_query_to_server(self, request_id):
        """Attempts to connect to the calculation server, and
        submit the requested calculations.

        Notes
        -----

        This method is based on the StackOverflow response from
        A. Jesse Jiryu Davis: https://stackoverflow.com/a/40257248

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

        try:

            # Attempt to establish a connection to the server.
            stream = await self._tcp_client.connect(
                self._connection_options.server_address,
                self._connection_options.server_port,
            )

            stream.set_nodelay(True)

            # Encode the request id into the message.
            message_type = pack_int(PropertyEstimatorMessageTypes.Query)

            encoded_request_id = request_id.encode()
            length = pack_int(len(encoded_request_id))

            await stream.write(message_type + length + encoded_request_id)

            # Wait for the server response.
            header = await stream.read_bytes(4)
            length = unpack_int(header)[0]

            # Decode the response from the server. If everything
            # went well, this should be the finished calculation.
            if length > 0:

                encoded_json = await stream.read_bytes(length)
                server_response = encoded_json.decode()

            stream.close()
            self._tcp_client.close()

        except StreamClosedError as e:

            # Handle no connections to the server gracefully.
            logging.info(
                "Error connecting to {}:{} : {}. Please ensure the server is running and"
                "that the server address / port is correct.".format(
                    self._connection_options.server_address,
                    self._connection_options.server_port,
                    e,
                )
            )

        response, error = None

        if server_response is not None:
            response, error = json.loads(server_response, cls=TypedJSONDecoder)

        return response, error


class _Submission(TypedBaseModel):
    """Represents a set of properties to be estimated by the server backend,
    the parameters which will be used to estimate them, and options about
    how the properties will be estimated.

    Warnings
    --------
    This class is still heavily under development and is subject to rapid changes.

    Attributes
    ----------
    properties: list of PhysicalProperty
        The list of physical properties to estimate.
    options: RequestOptions
        The options which control how the `properties` are estimated.
    force_field_source: ForceFieldSource
        The source of the force field parameters used during the calculations.
    """

    def __init__(
        self,
        properties=None,
        force_field_source=None,
        options=None,
        parameter_gradient_keys=None,
    ):
        """Constructs a new _Submission object.

        Parameters
        ----------
        properties: list of PhysicalProperty
            The list of physical properties to estimate.
        options: RequestOptions
            The options which control how the `properties` are estimated.
        force_field_source: ForceFieldSource
            The source of the force field parameters used during the calculations.
        parameter_gradient_keys: list of ParameterGradientKey
            A list of references to all of the parameters which all observables
            should be differentiated with respect to.
        """
        self.properties = properties or []
        self.options = options

        self.force_field_source = force_field_source

        self.parameter_gradient_keys = (
            [] if parameter_gradient_keys is None else parameter_gradient_keys
        )

    def __getstate__(self):

        return {
            "properties": self.properties,
            "options": self.options,
            "force_field_source": self.force_field_source,
            "parameter_gradient_keys": self.parameter_gradient_keys,
        }

    def __setstate__(self, state):

        self.properties = state["properties"]
        self.options = state["options"]

        self.force_field_source = state["force_field_source"]
        self.parameter_gradient_keys = state["parameter_gradient_keys"]
