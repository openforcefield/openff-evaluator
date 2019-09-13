"""
Property estimator client side API.
"""

import json
import logging
from time import sleep

from simtk import unit
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.tcpclient import TCPClient

from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.layers import ReweightingLayer, SimulationLayer
from propertyestimator.properties.plugins import registered_properties
from propertyestimator.utils.serialization import TypedBaseModel
from propertyestimator.utils.tcp import PropertyEstimatorMessageTypes, pack_int, unpack_int
from propertyestimator.workflow import WorkflowOptions


class PropertyEstimatorOptions(TypedBaseModel):
    """Represents the options options that can be passed to the
    property estimation server backend.

    Warnings
    --------
    * This class is still heavily under development and is subject to rapid changes.

    Attributes
    ----------
    allowed_calculation_layers: list of str or list of class
        A list of allowed calculation layers. The order of the layers in the list is the order
        that the calculator will attempt to execute the layers in.
    workflow_schemas: dict of str and dict of str and WorkflowSchema
        A dictionary of the WorkflowSchema which will be used to calculate any properties.
        The dictionary key represents the type of property the schema will calculate. The
        dictionary will be automatically populated with defaults if no entries are added.
    workflow_options: dict of str and dict of str and WorkflowOptions, optional
        The set of options which will be used when setting up the default estimation
        workflows, where the string key here is the property for which the options apply.
        As an example, the target (relative or absolute) uncertainty of each property may be set
        using these options.

        If None, a set of defaults will be applied when the properties are sent to a server for
        estimation. The current set of defaults will ensure that properties are estimated with an
        uncertainty which is less than or equal to the experimental uncertainty of a property.
    allow_protocol_merging: bool, default = True
        If true, allows individual identical steps in a property estimation workflow to be merged.
    """

    def __init__(self, allowed_calculation_layers=None,
                 allow_protocol_merging=True):
        """Constructs a new PropertyEstimatorOptions object.

        Parameters
        ----------
        allowed_calculation_layers: list of str or list of class
            A list of allowed calculation layers. The order of the layers in the list is the order
            that the calculator will attempt to execute the layers in.

            If None, all registered calculation layers are set as allowed.
        allow_protocol_merging: bool, default = True
            If true, allows individual identical steps in a property estimation workflow to be merged.
        """

        if allowed_calculation_layers is None:

            self.allowed_calculation_layers = [
                ReweightingLayer.__name__,
                SimulationLayer.__name__
            ]

        else:

            self.allowed_calculation_layers = []

            for allowed_layer in allowed_calculation_layers:

                if isinstance(allowed_layer, str):
                    self.allowed_calculation_layers.append(allowed_layer)
                else:
                    self.allowed_calculation_layers.append(allowed_layer.__name__)

        self.workflow_schemas = {}
        self.workflow_options = {}

        self.allow_protocol_merging = allow_protocol_merging

    def __getstate__(self):

        return {
            'allowed_calculation_layers': self.allowed_calculation_layers,

            'workflow_schemas': self.workflow_schemas,
            'workflow_options': self.workflow_options,

            'allow_protocol_merging': self.allow_protocol_merging
        }

    def __setstate__(self, state):

        self.allowed_calculation_layers = state['allowed_calculation_layers']

        self.workflow_schemas = state['workflow_schemas']
        self.workflow_options = state['workflow_options']

        self.allow_protocol_merging = state['allow_protocol_merging']


class PropertyEstimatorSubmission(TypedBaseModel):
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
    options: PropertyEstimatorOptions
        The options which control how the `properties` are estimated.
    force_field_source: ForceFieldSource
        The source of the force field parameters used during the calculations.
    """
    def __init__(self, properties=None, force_field_source=None, options=None, parameter_gradient_keys=None):
        """Constructs a new PropertyEstimatorSubmission object.

        Parameters
        ----------
        properties: list of PhysicalProperty
            The list of physical properties to estimate.
        options: PropertyEstimatorOptions
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

        self.parameter_gradient_keys = [] if parameter_gradient_keys is None else parameter_gradient_keys

    def __getstate__(self):

        return {
            'properties': self.properties,
            'options': self.options,

            'force_field_source': self.force_field_source,

            'parameter_gradient_keys': self.parameter_gradient_keys
        }

    def __setstate__(self, state):

        self.properties = state['properties']
        self.options = state['options']

        self.force_field_source = state['force_field_source']
        self.parameter_gradient_keys = state['parameter_gradient_keys']


class PropertyEstimatorResult(TypedBaseModel):
    """Represents the results of attempting to estimate a set of physical
    properties using the property estimator server backend.

    Warnings
    --------
    This class is still heavily under development and is subject to rapid changes.

    Attributes
    ----------
    id: str
        The unique id assigned to this result set by the server.
    queued_properties: dict of str and PhysicalProperty
        A dictionary of the properties which have yet to be estimated by
        the server.
    estimated_properties: dict of str and PhysicalProperty
        A dictionary of the properties which were successfully estimated, where
        the dictionary key is the unique id of the property being estimated.
    unsuccessful_properties: dict of str and PhysicalProperty
        A dictionary of the properties which could not be estimated by the server.
    exceptions: list of PropertyEstimatorException
        A list of the exceptions that were raised when unsuccessfully carrying out this
        estimation request.
    """

    def __init__(self, result_id=''):
        """Constructs a new PropertyEstimatorResult object.

        Parameters
        ----------
        result_id: str
            The unique id assigned to this result set by the server.
        """

        self.id = result_id

        self.queued_properties = {}

        self.estimated_properties = {}
        self.unsuccessful_properties = {}

        self.exceptions = []

    def __getstate__(self):

        return {
            'id:': self.id,

            'queued_properties': self.queued_properties,

            'estimated_properties': self.estimated_properties,
            'unsuccessful_properties': self.unsuccessful_properties,

            'exceptions': self.exceptions,
        }

    def __setstate__(self, state):

        self.id = state['id:']

        self.queued_properties = state['queued_properties']

        self.estimated_properties = state['estimated_properties']
        self.unsuccessful_properties = state['unsuccessful_properties']

        self.exceptions = state['exceptions']


class ConnectionOptions(TypedBaseModel):
    """The set of options to use when connecting to a
    `PropertyEstimatorServer`

    Attributes
    ----------
    server_address: str
        The address of the server to connect to.
    server_port: int
        The port number that the server is listening on.

    Warnings
    --------
    This class is still heavily under development and is subject to rapid changes.
    """

    server_address: str = 'localhost'
    server_port: int = 8000

    def __init__(self, server_address='localhost', server_port=8000):
        """Constructs a new ConnectionOptions object.

        Parameters
        ----------
        server_address: str
            The address of the server to connect to.
        server_port: int
            The port number that the server is listening on.
        """

        self.server_address = server_address
        self.server_port = server_port

    def __getstate__(self):

        return {
            'server_address': self.server_address,
            'server_port': self.server_port,
        }

    def __setstate__(self, state):

        self.server_address = state['server_address']
        self.server_port = state['server_port']


class PropertyEstimatorClient:
    """The PropertyEstimatorClient is the main object that users of the
    property estimator will interface with. It is responsible for requesting
    that a PropertyEstimatorServer estimates a set of physical properties,
    as well as querying for when those properties have been estimated.

    The PropertyEstimatorClient supports two main workflows: one where
    a PropertyEstimatorServer lives on a remote supercomputing cluster
    where all of the expensive calculations will be run, and one where
    the users local machine acts as both the server and the client, and
    all calculations will be performed locally.

    Warnings
    --------
    While the API of this class in now close to being final, the internals and implementation
    are still heavily under development and is subject to rapid changes.

    Examples
    --------

    Setting up the client instance:

    >>> from propertyestimator.client import PropertyEstimatorClient
    >>> property_estimator = PropertyEstimatorClient()

    If the PropertyEstimatorServer is not running on the local machine, you will
    need to specify its address and the port that it is listening on:

    >>> from propertyestimator.client import ConnectionOptions
    >>>
    >>> connection_options = ConnectionOptions(server_address='server_address',
    >>>                                                         server_port=8000)
    >>> property_estimator = PropertyEstimatorClient(connection_options)

    To asynchronously submit a request to the running server using the default estimator
    options:

    >>> # Load in the data set of properties which will be used for comparisons
    >>> from propertyestimator.datasets import ThermoMLDataSet
    >>> data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
    >>> # Filter the dataset to only include densities measured between 130-260 K
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
    PropertyEstimatorOptions object to the estimate commands.

    The calculations layers which will be used to estimate the properties can be
    controlled for example like so:

    >>> from propertyestimator.layers import ReweightingLayer, SimulationLayer
    >>>
    >>> options = PropertyEstimatorOptions(allowed_calculation_layers = [ReweightingLayer,
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

    >>> from propertyestimator.properties import ParameterGradientKey
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
        return self._connection_options.server_address

    @property
    def server_port(self):
        return self._connection_options.server_port

    class Request:
        """An object representation of a estimation request which has
        been sent to a `PropertyEstimatorServer` instance. This object
        can be used to query and retrieve the results of the request, or
        be stored to retrieve the request at some point in the future."""

        @property
        def id(self):
            """str: The id of the submitted request."""
            return self._id

        @property
        def server_address(self):
            """str: The address of the server that the request was sent to."""
            return self._server_address

        @property
        def server_port(self):
            """The port that the server is listening on."""
            return self._server_port

        def __init__(self, request_id, connection_options, client=None):
            """Constructs a new Request object.

            Parameters
            ----------
            request_id: str
                The id of the submitted request.
            connection_options: ConnectionOptions
                The options that were used to connect to the server that the request was sent to.
            client: PropertyEstimatorClient, optional
                The client that was used to submit the request.
            """
            self._id = request_id

            self._server_address = connection_options.server_address
            self._server_port = connection_options.server_port

            self._client = client

            if client is None:

                connection_options = ConnectionOptions(
                    server_address=connection_options.server_address,
                    server_port=connection_options.server_port)

                self._client = PropertyEstimatorClient(connection_options)

        def __str__(self):

            return 'EstimateRequest id: {} server_address: {} server_port: {}'.format(self._id,
                                                                                      self._server_address,
                                                                                      self._server_port)

        def __repr__(self):
            return '<EstimateRequest id: {} server_address: {} server_port: {}>'.format(self._id,
                                                                                        self._server_address,
                                                                                        self._server_port)

        def json(self):
            """Returns a JSON representation of the `Request` object.

            Returns
            -------
            str:
                The JSON representation of the `Request` object.
            """

            return json.dumps({
                'id': self._id,
                'server_address': self._id,
                'server_port': self._id
            })

        @classmethod
        def from_json(cls, json_string):
            """Creates a new `Request` object from a JSON representation.

            Parameters
            ----------
            json_string: str
                The JSON representation of the `Request` object.

            Returns
            -------
            str:
                The created `Request` object.
            """
            json_dict = json.loads(json_string)

            return cls(json_dict['id'],
                       json_dict['server_address'],
                       json_dict['server_port'])

        def results(self, synchronous=False, polling_interval=5):
            """Retrieve the results of an estimate request.

            Parameters
            ----------
            synchronous: bool
                If true, this method will block the main thread until the server
                either returns a result or an error.
            polling_interval: int
                If running synchronously, this is the time interval (seconds) between
                checking if the calculation has finished.

            Returns
            -------
            PropertyEstimatorResult or PropertyEstimatorException:
                Returns either the results of the requested estimate, or any
                exceptions which were raised.

                If the method is run synchronously then this method will block the main
                thread until all of the requested properties have been estimated, or
                an exception is returned.
            """
            return self._client._retrieve_estimate(self._id, synchronous, polling_interval)

    def __init__(self, connection_options=ConnectionOptions()):
        """Constructs a new PropertyEstimatorClient object.

        Parameters
        ----------
        connection_options: ConnectionOptions
            The options used when connecting to the calculation server.
        """

        self._connection_options = connection_options

        if connection_options.server_address is None:

            raise ValueError('The address of the server which will run'
                             'these calculations must be given.')

        self._tcp_client = TCPClient()

    def request_estimate(self, property_set, force_field_source, options=None, parameter_gradient_keys=None):
        """Requests that a PropertyEstimatorServer attempt to estimate the
        provided property set using the supplied force field and estimator options.

        Parameters
        ----------
        property_set : PhysicalPropertyDataSet
            The set of properties to attempt to estimate.
        force_field_source : ForceFieldSource or openforcefield.typing.engines.smirnoff.ForceField
            The source of the force field parameters to use for the calculations.
        options : PropertyEstimatorOptions, optional
            A set of estimator options. If None, default options
            will be used.
        parameter_gradient_keys: list of ParameterGradientKey, optional
            A list of references to all of the parameters which all observables
            should be differentiated with respect to.

        Returns
        -------
        PropertyEstimatorClient.Request
            An object which will provide access the the results of the request.
        """
        from openforcefield.typing.engines import smirnoff

        if property_set is None or force_field_source is None:

            raise ValueError('Both a data set and force field source must be '
                             'present to compute physical properties.')

        if options is None:
            options = PropertyEstimatorOptions()

        if isinstance(force_field_source, smirnoff.ForceField):
            force_field_source = SmirnoffForceFieldSource.from_object(force_field_source)

        if len(options.allowed_calculation_layers) == 0:
            raise ValueError('A submission contains no allowed calculation layers.')

        properties_list = []
        property_types = set()

        # Refactor the properties into a list, and extract the types
        # of properties to be estimated (e.g 'Denisty', 'DielectricConstant').
        for substance_tag in property_set.properties:

            for physical_property in property_set.properties[substance_tag]:

                properties_list.append(physical_property)

                type_name = type(physical_property).__name__

                if type_name not in registered_properties:
                    raise ValueError(f'The property estimator does not support {type_name} properties.')

                if type_name in property_types:
                    continue

                property_types.add(type_name)

        if options.workflow_options is None:
            options.workflow_options = {}

        # Assign default workflows in the cases where the user hasn't
        # provided one, and validate all of the workflows to be used
        # in the estimation.
        for type_name in property_types:

            if type_name not in options.workflow_schemas:
                options.workflow_schemas[type_name] = {}

            if type_name not in options.workflow_options:
                options.workflow_options[type_name] = {}

            for calculation_layer in options.allowed_calculation_layers:

                property_type = registered_properties[type_name]()

                if (calculation_layer not in options.workflow_options[type_name] or
                    options.workflow_options[type_name][calculation_layer] is None):

                    options.workflow_options[type_name][calculation_layer] = WorkflowOptions()

                if (calculation_layer not in options.workflow_schemas[type_name] or
                    options.workflow_schemas[type_name][calculation_layer] is None):

                    default_schema = property_type.get_default_workflow_schema(
                        calculation_layer, options.workflow_options[type_name][calculation_layer])

                    options.workflow_schemas[type_name][calculation_layer] = default_schema

                workflow = options.workflow_schemas[type_name][calculation_layer]

                if workflow is None:
                    # Not all properties may support every calculation layer.
                    continue

                # Will raise the correct exception for non-valid interfaces.
                workflow.validate_interfaces()

                # Enforce the global option of whether to allow merging or not.
                for protocol_schema_name in workflow.protocols:

                    protocol_schema = workflow.protocols[protocol_schema_name]

                    if not options.allow_protocol_merging:
                        protocol_schema.inputs['.allow_merging'] = False

        submission = PropertyEstimatorSubmission(properties=properties_list,
                                                 force_field_source=force_field_source,
                                                 options=options,
                                                 parameter_gradient_keys=parameter_gradient_keys)

        request_id = IOLoop.current().run_sync(lambda: self._send_calculations_to_server(submission))

        request_object = PropertyEstimatorClient.Request(request_id,
                                                         self._connection_options,
                                                         self)

        return request_object

    def _retrieve_estimate(self, request_id, synchronous=False, polling_interval=5):
        """A method to retrieve the status of a requested estimate from the server.

        Parameters
        ----------
        request_id: str
            The id of the estimate request which was returned by the server
            upon making the request.
        synchronous: bool
            If true, this method will block the main thread until the server
            either returns a result or an error.
        polling_interval: int
            If running synchronously, this is the time interval (seconds) between
            checking if the calculation has finished.

        Returns
        -------
        PropertyEstimatorResult or PropertyEstimatorException:
            Returns either the results of the requested estimate, or any
            exceptions which were raised.

            If the method is run synchronously then this method will block the main
            thread until all of the requested properties have been estimated, or
            an exception is returned.
        """

        # If running asynchronously, just return whatever the server
        # sends back.
        if synchronous is False:
            return IOLoop.current().run_sync(lambda: self._send_query_server(request_id))

        assert polling_interval >= 0

        response = None
        should_run = True

        while should_run:

            if polling_interval > 0:
                sleep(polling_interval)

            response = IOLoop.current().run_sync(lambda: self._send_query_server(request_id))

            if isinstance(response, PropertyEstimatorResult) and len(response.queued_properties) > 0:
                continue

            logging.info(f'The server has completed request {request_id}.')
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
        submission: PropertyEstimatorSubmission
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
            logging.info("Attempting Connection to {}:{}".format(self._connection_options.server_address,
                                                                 self._connection_options.server_port))

            stream = await self._tcp_client.connect(self._connection_options.server_address,
                                                                 self._connection_options.server_port)

            logging.info("Connected to {}:{}".format(self._connection_options.server_address,
                                                                 self._connection_options.server_port))

            stream.set_nodelay(True)

            # Encode the submission json into an encoded
            # packet ready to submit to the server. The
            # Length of the packet is encoded in the first
            # four bytes.
            message_type = pack_int(PropertyEstimatorMessageTypes.Submission)

            encoded_json = submission.json().encode()
            length = pack_int(len(encoded_json))

            await stream.write(message_type + length + encoded_json)

            logging.info("Sent calculations to {}:{}. Waiting for a response from"
                         " the server...".format(self._connection_options.server_address,
                                                 self._connection_options.server_port))

            # Wait for confirmation that the server has submitted
            # the jobs. The first four bytes of the response should
            # be the length of the message being sent.
            header = await stream.read_bytes(4)
            length = unpack_int(header)[0]

            # Decode the response from the server. If everything
            # went well, this should be a list of ids of the submitted
            # calculations.
            encoded_json = await stream.read_bytes(length)
            request_id = json.loads(encoded_json.decode())

            logging.info('Received job id from server: {}'.format(request_id))
            stream.close()
            self._tcp_client.close()

        except StreamClosedError as e:

            # Handle no connections to the server gracefully.
            logging.info("Error connecting to {}:{} : {}. Please ensure the server is running and"
                         "that the server address / port is correct.".format(self._connection_options.server_address,
                                                                             self._connection_options.server_port, e))

        # Return the ids of the submitted jobs.
        return request_id

    async def _send_query_server(self, request_id):
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
            stream = await self._tcp_client.connect(self._connection_options.server_address,
                                                    self._connection_options.server_port)

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
            logging.info("Error connecting to {}:{} : {}. Please ensure the server is running and"
                         "that the server address / port is correct.".format(self._connection_options.server_address,
                                                                             self._connection_options.server_port, e))

        if server_response is not None:
            server_response = TypedBaseModel.parse_json(server_response)

        # Return the ids of the submitted jobs.
        return server_response
