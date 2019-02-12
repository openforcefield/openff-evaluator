"""
Property estimator client side API.
"""

import json
import logging
from time import sleep
from typing import Dict, List

from pydantic import BaseModel, ValidationError
from simtk import unit
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.tcpclient import TCPClient

from propertyestimator.layers import SurrogateLayer, ReweightingLayer, SimulationLayer
from propertyestimator.properties import PhysicalProperty
from propertyestimator.properties.plugins import registered_properties
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import serialize_quantity, PolymorphicDataType, serialize_force_field
from propertyestimator.utils.tcp import PropertyEstimatorMessageTypes, pack_int, unpack_int
from propertyestimator.workflow import WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath


class PropertyEstimatorOptions(BaseModel):
    """Represents the options options that can be passed to the
    property estimation server backend.

    Warnings
    --------
    The `gradient_properties` property is not implemented yet, and is meant only
    as a placeholder for future api development.

    Attributes
    ----------
    allowed_calculation_layers: :obj:`list` of :obj:`str`
        A list of allowed calculation layers. The order of the layers in the list is the order
        that the calculator will attempt to execute the layers in.
    workflow_schemas: :obj:`dict` of :obj:`str` and :obj:`WorkflowSchema`
        A dictionary of the WorkflowSchema which will be used to calculate any properties.
        The dictionary key represents the type of property the schema will calculate. The
        dictionary will be automatically populated with defaults if no entries are added.
    relative_uncertainty_tolerance: :obj:`float`, default = 1.0
        Controls the desired uncertainty of any calculated properties. The estimator will
        attempt to estimate all properties to within an uncertainity equal to:

        `target_uncertainty = relative_uncertainty_tolerance * experimental_uncertainty`
    allow_protocol_merging: :obj:`bool`, default = True
        If true, allows individual identical steps in a property estimation workflow to be merged.

    gradient_properties: :obj:`list` of :obj:`str`
        A list of the types of properties to calculate gradients for. As an example
        setting this to ['Density'] would calculate the gradients of any estimated
        densities.
    """
    allowed_calculation_layers: List[str] = [
        SurrogateLayer.__name__,
        ReweightingLayer.__name__,
        SimulationLayer.__name__
    ]

    workflow_schemas: Dict[str, WorkflowSchema] = {}

    relative_uncertainty_tolerance: float = 1.0
    allow_protocol_merging: bool = True

    gradient_properties: List[str] = []

    class Config:

        arbitrary_types_allowed = True

        json_encoders = {
            unit.Quantity: lambda v: serialize_quantity(v),
            ProtocolPath: lambda v: v.full_path,
            PolymorphicDataType: lambda value: PolymorphicDataType.serialize(value)
        }


class PropertyEstimatorSubmission(BaseModel):
    """Represents a set of properties to be estimated by the server backend,
    the parameters which will be used to estimate them, and options about
    how the properties will be estimated.

    Attributes
    ----------
    properties: :obj:`list` of :obj:`PhysicalProperty`
        The list of physical properties to estimate.
    options: :obj:`PropertyEstimatorOptions`
        The options which control how the `properties` are estimated.
    force_field: :obj:`dict` of :obj:`int` and :obj:`str`
        The force field parameters used during the calculations. These should be
        obtained by calling `serialize_force_field` on a `ForceField` object.
    """
    properties: List[PhysicalProperty] = []
    options: PropertyEstimatorOptions = None

    force_field: Dict[int, str] = None

    class Config:

        arbitrary_types_allowed = True

        json_encoders = {
            unit.Quantity: lambda v: serialize_quantity(v),
            ProtocolPath: lambda v: v.full_path,
            PolymorphicDataType: lambda value: PolymorphicDataType.serialize(value)
        }


class PropertyEstimatorResult(BaseModel):
    """Represents the results of attempting to estimate a set of physical
    properties using the property estimator server backend.

    Attributes
    ----------
    id: :obj:`str`
        The unique id assigned to this result set by the server.
    estimated_properties: :obj:`dict` of :obj:`str` and :obj:`PhysicalProperty`
        A dictionary of the properties which were successfully estimated, where
        the dictionary key is the unique id of the property being estimated.
    unsuccessful_properties: :obj:`dict` of :obj:`str` and :obj:`PhysicalProperty`
        A dictionary of the properties which could not be estimated. The dictionary
        key is the unique id of the property which could not be estimated.
    force_field_id:
        The server assigned id of the parameter set used in the calculation.
    """
    id: str

    estimated_properties: Dict[str, PhysicalProperty] = {}
    unsuccessful_properties: Dict[str, PropertyEstimatorException] = {}

    force_field_id: str = None

    class Config:
        arbitrary_types_allowed = True

        json_encoders = {
            unit.Quantity: lambda v: serialize_quantity(v),
            ProtocolPath: lambda v: v.full_path,
            PolymorphicDataType: lambda value: PolymorphicDataType.serialize(value)
        }


class PropertyEstimatorClient:
    """The :obj:`PropertyEstimatorClient` is the main object that users of the
    property estimator will interface with. It is responsible for requesting
    that a :obj:`PropertyEstimatorServer` estimates a set of physical properties,
    as well as querying for when those properties have been estimated.

    The :obj:`PropertyEstimatorClient` supports two main workflows: one where
    a :obj:`PropertyEstimatorServer` lives on a remote supercomputing cluster
    where all of the expensive calculations will be run, and one where
    the users local machine acts as both the server and the client, and
    all calculations will be performed locally.

    Examples
    --------

    Setting up the client instance:

    >>> from propertyestimator.client import PropertyEstimatorClient
    >>> property_estimator = PropertyEstimatorClient()

    If the :obj:`PropertyEstimatorServer` is not running on the local machine, you will
    need to specify its address and the port that it is listening on:

    >>> property_estimator = PropertyEstimatorClient(server_address='server_address',
    >>>                                              port=8000)

    To asynchronously submit a request to the running server using the default estimator
    options:

    >>> # Load in the data set of properties which will be used for comparisons
    >>> from propertyestimator.datasets import ThermoMLDataSet
    >>> data_set = ThermoMLDataSet.from_doi_list('10.1016/j.jct.2016.10.001')
    >>> # Filter the dataset to only include densities measured between 130-260 K
    >>> from propertyestimator.properties import Density
    >>>
    >>> data_set.filter_by_properties(types=[Density.__name__])
    >>> data_set.filter_by_temperature(min_temperature=130*unit.kelvin, max_temperature=260*unit.kelvin)
    >>>
    >>> # Load initial parameters
    >>> from openforcefield.typing.engines.smirnoff import ForceField
    >>> parameters = ForceField(['smirnoff99Frosst.offxml'])
    >>>
    >>> ticket_id = property_estimator.estimate(data_set, parameters)

    The status of the request can be asynchronously queried by calling

    >>> results = property_estimator.retrieve_estimate(ticket_id)

    or the main thread can be blocked until the results are
    available by calling

    >>> results = property_estimator.retrieve_estimate(ticket_id, synchronous=True)

    Alternatively, the estimate request can be ran synchronously by
    setting the synchronous parameter to True

    >>> results = property_estimator.estimate(data_set, parameters, synchronous=True)

    How the property set will be estimated can easily be controlled by passing a
    :obj:`PropertyEstimatorOptions` object to the estimate commands.

    The calculations layers which will be used to estimate the properties can be
    controlled for example like so:

    >>> options = PropertyEstimatorOptions(allowed_calculation_layers = [ReweightingLayer.__name__,
    >>>                                                                  SimulationLayer.__name__])
    >>>
    >>> ticket_id = property_estimator.estimate(data_set, parameters, options)

    As can the uncertainty tolerance:

    >>> options = PropertyEstimatorOptions(relative_uncertainty_tolerance = 0.1)
    """

    def __init__(self, server_address='localhost', port=8000):
        """Constructs a new PropertyEstimatorClient object.

        Parameters
        ----------
        server_address: :obj:`str`
            The address of the calculation server.
        port: :obj:`int`
            The port that the server is listening on.
        """

        self._server_address = server_address

        if server_address is None:

            raise ValueError('The address of the server which will run'
                             'these calculations must be given.')

        self._port = port
        self._tcp_client = TCPClient()

    def estimate(self, property_set, force_field, options=None, synchronous=False, polling_interval=5):
        """A blocking convenience method for requesting that a :obj:`PropertyEstimatorServer`
        attempt to estimate the provided property set using the supplied force field and
        estimator options.

        Parameters
        ----------
        property_set : :obj:`PhysicalPropertyDataSet`
            The set of properties to attempt to estimate.
        force_field : :obj:`ForceField`
            The OpenFF force field to use for the calculations.
        options : :obj:`PropertyEstimatorOptions`, optional
            A set of estimator options. If None, default options
            will be used.
        synchronous : :obj:`bool`, default=False
            If True, this will be a blocking call which will not return
            until the server has finished estimated all of the properties.
            If False, this method will return a unique id which can be used
            to query / retrieve the results of the submitted estimate request.
        polling_interval: :obj:`int`
            If running synchronously, this is the time interval (seconds) between
            checking if the calculation has finished.

        Returns
        -------
        If the synchronous option is True, either a PropertyEstimatorResult or a
        PropertyCalculatorException will be returned depending on whether the
        server returned an error. If the synchronous option is false, a unique
        :obj`str` id which can be used to query / retrieve the results of the
        submitted estimate request.
        """
        if property_set is None or force_field is None:

            raise ValueError('Both a data set and parameter set must be '
                             'present to compute physical properties.')

        if options is None:
            options = PropertyEstimatorOptions()

        properties_list = []

        for substance_tag in property_set.properties:

            for physical_property in property_set.properties[substance_tag]:

                properties_list.append(physical_property)

                type_name = type(physical_property).__name__

                if type_name not in registered_properties:

                    raise ValueError('The property estimator does not support {} '
                                     'properties.'.format(type_name))

                if type_name in options.workflow_schemas:
                    continue

                property_type = registered_properties[type_name]()

                options.workflow_schemas[type_name] = \
                    property_type.get_default_calculation_schema()

        for property_schema_name in options.workflow_schemas:

            options.workflow_schemas[property_schema_name].validate_interfaces()

            for protocol_schema_name in options.workflow_schemas[property_schema_name].protocols:

                protocol_schema = options.workflow_schemas[property_schema_name].protocols[protocol_schema_name]
                protocol_schema.inputs['.allow_merging'] = PolymorphicDataType(options.allow_protocol_merging)

        submission = PropertyEstimatorSubmission(properties=properties_list,
                                                 force_field=serialize_force_field(force_field),
                                                 options=options)

        ticket_id = IOLoop.current().run_sync(lambda: self._send_calculations_to_server(submission))

        if synchronous is False:
            return ticket_id

        return self.retrieve_estimate(ticket_id, True, polling_interval)

    def retrieve_estimate(self, ticket_id, synchronous=False, polling_interval=5):
        """A method to retrieve the status of a requested estimate from the server.

        Parameters
        ----------
        ticket_id: :obj:`str`
            The id of the estimate request which was returned by the server
            upon making the request.
        synchronous: :obj:`bool`
            If true, this method will block the main thread until the server
            either returns a result or an error.
        polling_interval: :obj:`int`
            If running synchronously, this is the time interval (seconds) between
            checking if the calculation has finished.

        Returns
        -------
        If the method is run synchronously then this method will block the main
        thread until the server returns either an error (:obj:`PropertyCalculatorException`),
        or the results of the requested estimate (:obj:`PropertyEstimatorResult`).

        If the method is run asynchronously, it will return None if the
        estimate is still queued for calculation on the server, or either
        an error (:obj:`PropertyCalculatorException`) or results
        (:obj:`PropertyEstimatorResult`) object.
        """

        # If running asynchronously, just return whatever the server
        # sends back.
        if synchronous is False:
            return IOLoop.current().run_sync(lambda: self._send_query_server(ticket_id))

        assert polling_interval >= 5

        response = None
        should_run = True

        while should_run:

            sleep(polling_interval)

            response = IOLoop.current().run_sync(lambda: self._send_query_server(ticket_id))

            if response is None:
                continue

            logging.info('The server has returned a response.')
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
        ticket_id = None

        try:

            # Attempt to establish a connection to the server.
            logging.info("Attempting Connection to {}:{}".format(self._server_address, self._port))
            stream = await self._tcp_client.connect(self._server_address, self._port)
            logging.info("Connected to {}:{}".format(self._server_address, self._port))

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
                         " the server...".format(self._server_address, self._port))

            # Wait for confirmation that the server has submitted
            # the jobs. The first four bytes of the response should
            # be the length of the message being sent.
            header = await stream.read_bytes(4)
            length = unpack_int(header)[0]

            # Decode the response from the server. If everything
            # went well, this should be a list of ids of the submitted
            # calculations.
            encoded_json = await stream.read_bytes(length)
            ticket_id = json.loads(encoded_json.decode())

            logging.info('Received job id from server: {}'.format(ticket_id))
            stream.close()
            self._tcp_client.close()

        except StreamClosedError as e:

            # Handle no connections to the server gracefully.
            logging.info("Error connecting to {}:{} : {}. Please ensure the server is running and"
                         "that the server address / port is correct.".format(self._server_address, self._port, e))

        # Return the ids of the submitted jobs.
        return ticket_id

    async def _send_query_server(self, ticket_id):
        """Attempts to connect to the calculation server, and
        submit the requested calculations.

        Notes
        -----

        This method is based on the StackOverflow response from
        A. Jesse Jiryu Davis: https://stackoverflow.com/a/40257248

        Parameters
        ----------
        ticket_id: str
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
            logging.info("Attempting Connection to {}:{}".format(self._server_address, self._port))
            stream = await self._tcp_client.connect(self._server_address, self._port)
            logging.info("Connected to {}:{}".format(self._server_address, self._port))

            stream.set_nodelay(True)

            # Encode the ticket id into the message.
            message_type = pack_int(PropertyEstimatorMessageTypes.Query)

            encoded_ticket_id = ticket_id.encode()
            length = pack_int(len(encoded_ticket_id))

            await stream.write(message_type + length + encoded_ticket_id)

            logging.info("Querying the server {}:{}...".format(self._server_address, self._port))

            # Wait for the server response.
            header = await stream.read_bytes(4)
            length = unpack_int(header)[0]

            # Decode the response from the server. If everything
            # went well, this should be the finished calculation.
            if length > 0:

                encoded_json = await stream.read_bytes(length)
                server_response = encoded_json.decode()

            logging.info('Received response from server of length {}'.format(length))

            stream.close()
            self._tcp_client.close()

        except StreamClosedError as e:

            # Handle no connections to the server gracefully.
            logging.info("Error connecting to {}:{} : {}. Please ensure the server is running and"
                         "that the server address / port is correct.".format(self._server_address, self._port, e))

        if server_response is not None:

            try:
                server_response = PropertyEstimatorResult.parse_raw(server_response)
            except ValidationError:
                server_response = PropertyEstimatorException.parse_raw(server_response)

        # Return the ids of the submitted jobs.
        return server_response
