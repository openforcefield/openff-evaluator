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
    relative_uncertainty: :obj:`float`, default = 1.0
        Controls the desired uncertainty of any calculated properties. The estimator will
        attempt to estimate all properties to within an uncertainity equal to:

        `target_uncertainty = relative_uncertainty * experimental_uncertainty`
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

    relative_uncertainty: float = 1.0
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
    parameter_set: :obj:`dict` of :obj:`int` and :obj:`str`
        The force field parameters used during the calculations. These should be
        obtained by calling `serialize_force_field` on a `ForceField` object.
    """
    properties: List[PhysicalProperty] = []
    options: PropertyEstimatorOptions = None

    parameter_set: Dict[int, str] = None

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
    >>> # Create a property estimator
    >>> from propertyestimator.client import PropertyEstimatorClient
    >>> property_estimator = PropertyEstimatorClient()

    If the :obj:`PropertyEstimatorServer` is not running on the local machine, you will
    need to specify its address and the port that it is listening on:

    >>> property_estimator = PropertyEstimatorClient(server_address='server_address',
    >>>                                              port=8000)

    To submit a request to the running server using the default estimator options:

    >>> ticket_id = property_estimator.request_estimate(data_set, parameters)

    The status of the request can be queried by calling

    >>> results = property_estimator.query_estimate(ticket_id)

    Both the `request_estimate` and `query_estimate` methods are non-blocking,
    and will return even if the server hasn't yet finished estimating the property
    set. To perform a blocking property estimation request, use instead:

    >>> results = property_estimator.estimate(data_set, parameters)

    or the local version if you wish the properties to be estimated on your local
    machine:

    >>> results = property_estimator.estimate_locally(data_set, parameters)

    How the property set will be estimated can easily be controlled by passing a
    :obj:`PropertyEstimatorOptions` object to the estimate commands.

    The calculations layers which will be used to estimate the properties can be
    controlled for example like so:

    >>> options = PropertyEstimatorOptions(allowed_calculation_layers = [ReweightingLayer.__name__,
    >>>                                                                  SimulationLayer.__name__])
    >>>
    >>> ticket_id = property_estimator.request_estimate(data_set, parameters, options)

    As can the uncertainty tolerance:

    >>> options = PropertyEstimatorOptions(relative_uncertainty = 0.1)
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

    def request_estimate(self, property_set, force_field, options=None):
        """Sends a request to the :obj:`PropertyEstimatorServer` to estimate the provided
        property set using the supplied force field and estimator options.

        Parameters
        ----------
        property_set : :obj:`PhysicalPropertyDataSet`
            The set of properties to attempt to estimate.
        force_field : :obj:`ForceField`
            The OpenFF force field to use for the calculations.
        options : :obj:`PropertyEstimatorOptions`, optional
            A set of estimator options. If None, default options
            will be used.

        Returns
        -------
        :obj:`list` of :obj:`str`:
            A list unique ids which can be used to retrieve the submitted calculations
            when they have finished running.
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

                protocol_schema = options.workflow_schemas[
                    property_schema_name].protocols[protocol_schema_name]

                protocol_schema.inputs['.allow_merging'] = PolymorphicDataType(options.allow_protocol_merging)

        submission = PropertyEstimatorSubmission(properties=properties_list,
                                                 parameter_set=serialize_force_field(force_field),
                                                 options=options)

        # For now just do a blocking submit to the server.
        ticket_ids = IOLoop.current().run_sync(lambda: self._send_calculations_to_server(submission))

        return ticket_ids

    def query_estimate(self, ticket_id):
        """A method to retrieve the status (e.g. still running, finished) of
        a requested property estimation.

        Parameters
        ----------
        ticket_id: :obj:`str`
            The id of the estimate request which was returned by the server
            upon making the request.

        Returns
        -------
        :obj:`PropertyEstimatorResult` or :obj:`PropertyCalculatorException`, optional:
            The status requested estimation. Returns None if the estimation has
            not yet completed.
        """

        # For now just do a blocking submit to the server.
        result = IOLoop.current().run_sync(lambda: self._send_query_server(ticket_id))
        return result

    def wait_for_estimate(self, ticket_id, interval=1):
        """
        Synchronously wait for the result of a calculation

        Parameters
        ----------
        ticket_id: str
            The id of the calculation to wait for.
        interval: int
            The time interval (seconds) between checking if the calculation has finished.

        Returns
        -------
        PropertyEstimatorResult or PropertyCalculatorException, optional:
           The result of the submitted job. Returns None if the calculation has
           not yet completed.
        """
        assert interval >= 1

        response = None
        should_run = True

        while should_run:

            sleep(interval)

            response = IOLoop.current().run_sync(lambda: self._send_query_server(ticket_id))

            if response is None:
                continue

            logging.info('The server has returned a response.')
            should_run = False

        return response

    def estimate(self, property_set, force_field, options=None):
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

        Returns
        -------
        PropertyEstimatorResult or PropertyCalculatorException, optional:
           The result of the submitted job. Returns None if the calculation has
           not yet completed.
        """
        ticket_id = self.request_estimate(property_set, force_field, options)
        return self.wait_for_estimate(ticket_id)

    def estimate_locally(self, property_set, force_field, options=None,
                         calculation_backend=None, storage_backend=None):

        """A blocking convenience method for setting up a local :obj:`PropertyEstimatorServer`,
        and using this to attempt to estimate the provided property set using the supplied
        force field and estimator options.

        Parameters
        ----------
        property_set : :obj:`PhysicalPropertyDataSet`
            The set of properties to attempt to estimate.
        force_field : :obj:`ForceField`
            The OpenFF force field to use for the calculations.
        options : :obj:`PropertyEstimatorOptions`, optional
            A set of estimator options. If None, default options
            will be used.
        calculation_backend : :obj:`PropertyEstimatorBackend`
            The calculation backend to use. By default, a local dask cluster backend
            will be used:

            `calculation_backend = DaskLocalClusterBackend(1, 1, PropertyEstimatorBackendResources(1, 0))`

        storage_backend : :obj:`PropertyEstimatorStorage`
            The backend to use when storing the results of the calculations. By default,
            a local file storage backend will be used:

            `storage_backend = LocalFileStorage(root_directory='stored_data')`

        Returns
        -------
        :obj:`PropertyEstimatorResult` or :obj:`PropertyCalculatorException`, optional:
           The result of the submitted job. Returns None if the calculation has
           not yet completed.
        """

        from propertyestimator.backends import DaskLocalClusterBackend, PropertyEstimatorBackendResources
        calculation_backend = calculation_backend or DaskLocalClusterBackend(1, 1,
                                                                             PropertyEstimatorBackendResources(1, 0))

        from propertyestimator.storage import LocalFileStorage
        storage_backend = storage_backend or LocalFileStorage()

        from propertyestimator.server import PropertyEstimatorServer
        property_server = PropertyEstimatorServer(calculation_backend,
                                                    storage_backend,
                                                    working_directory='estimator_working_directory')

        ticket_id = self.request_estimate(property_set, force_field, options)
        result = self.wait_for_estimate(ticket_id)

        return result  # property_server.finished_calculations[ticket_id]

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
