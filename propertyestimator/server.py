"""
Property calculator 'server' side API.
"""

import json
import logging
import uuid
from os import path, makedirs

from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.tcpserver import TCPServer

from propertyestimator.client import PropertyEstimatorSubmission, PropertyEstimatorResult
from propertyestimator.layers import available_layers
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import TypedBaseModel
from propertyestimator.utils.tcp import PropertyEstimatorMessageTypes, pack_int, unpack_int


class PropertyEstimatorServer(TCPServer):
    """The object responsible for coordinating all properties estimations to to
    be ran using the property estimator, in addition to deciding at which fidelity
    a property will be calculated.

    It acts as a server, which receives submitted jobs from clients
    launched via the property estimator.

    Warnings
    --------
    This class is still heavily under development and is subject to rapid changes.

    Notes
    -----
    Methods to handle the TCP messages are based on the StackOverflow response from
    A. Jesse Jiryu Davis: https://stackoverflow.com/a/40257248

    Examples
    --------
    Setting up a general server instance using a dask LocalCluster backend:

    >>> # Create the backend which will be responsible for distributing the calculations
    >>> from propertyestimator.backends import DaskLocalClusterBackend, ComputeResources
    >>> calculation_backend = DaskLocalClusterBackend(1)
    >>>
    >>> # Calculate the backend which will be responsible for storing and retrieving
    >>> # the data from previous calculations
    >>> from propertyestimator.storage import LocalFileStorage
    >>> storage_backend = LocalFileStorage()
    >>>
    >>> # Create the server to which all estimation requests will be submitted
    >>> from propertyestimator.server import PropertyEstimatorServer
    >>> property_server = PropertyEstimatorServer(calculation_backend, storage_backend)
    >>>
    >>> # Instruct the server to listen for incoming requests
    >>> property_server.start_listening_loop()
    """

    class ServerEstimationRequest(TypedBaseModel):
        """Represents a request for the server to estimate a set of properties. Such requests
        are expected to only estimate properties for a single system (e.g. fixed components
        in a fixed ratio)
        """

        def __init__(self, estimation_id='', queued_properties=None, options=None, force_field_id=None):
            """Constructs a new ServerEstimationRequest object.

            Parameters
            ----------
            estimation_id: str
                A unique id assigned to this estimation request.
            queued_properties: list of PhysicalProperty, optional
                A list of physical properties waiting to be estimated.
            options: PropertyEstimatorOptions, optional
                The options used to estimate the properties.
            force_field_id: str
                The unique server side id of the force field parameters used to estimate the properties.
            """
            self.id = estimation_id

            self.queued_properties = queued_properties or []

            self.estimated_properties = {}
            self.unsuccessful_properties = {}

            self.exceptions = []

            self.options = options

            self.force_field_id = force_field_id

        def __getstate__(self):
            return {
                'id': self.id,

                'queued_properties': self.queued_properties,

                'estimated_properties': self.estimated_properties,
                'unsuccessful_properties': self.unsuccessful_properties,

                'exceptions': self.exceptions,

                'options': self.options,

                'force_field_id': self.force_field_id,
            }

        def __setstate__(self, state):
            self.id = state['id']

            self.queued_properties = state['queued_properties']

            self.estimated_properties = state['estimated_properties']
            self.unsuccessful_properties = state['unsuccessful_properties']

            self.exceptions = state['exceptions']

            self.options = state['options']

            self.force_field_id = state['force_field_id']

    def __init__(self, calculation_backend, storage_backend,
                 port=8000, working_directory='working-data'):
        """Constructs a new PropertyEstimatorServer object.

        Parameters
        ----------
        calculation_backend: PropertyEstimatorBackend
            The backend to use for executing calculations.
        storage_backend: PropertyEstimatorStorage
            The backend to use for storing information from any calculations.
        port: int
            The port on which to listen for incoming client requests.
        working_directory: str
            The local directory in which to store all local, temporary calculation data.
        """

        assert calculation_backend is not None and storage_backend is not None

        self._calculation_backend = calculation_backend
        self._storage_backend = storage_backend

        self._port = port

        self._working_directory = working_directory

        if not path.isdir(self._working_directory):
            makedirs(self._working_directory)

        self._queued_calculations = {}
        self._finished_calculations = {}

        # Each client request id (i.e an id relating to a client requesting
        # that an entire data set of properties is estimated) is matched to
        # a set of server set request ids.
        #
        # The main difference is that on the server, a request to estimate
        # an entire data set is split into multiple requests to estimate
        # properties per substance.
        self._server_request_ids_per_client_id = {}

        super().__init__()

        self.bind(self._port)
        self.start(1)

        calculation_backend.start()

    async def _handle_job_submission(self, stream, address, message_length):
        """An asynchronous routine for handling the receiving and processing
        of job submissions from a client.

        Parameters
        ----------
        stream: IOStream
            An IO stream used to pass messages between the
            server and client.
        address: str
            The address from which the request came.
        message_length: int
            The length of the message being received.
        """

        logging.info('Received estimation request from {}'.format(address))

        # Read the incoming request from the server. The first four bytes
        # of the response should be the length of the message being sent.

        # Decode the client submission json.
        encoded_json = await stream.read_bytes(message_length)
        json_model = encoded_json.decode()

        # TODO: Add exception handling so the server can gracefully reject bad json.
        client_data_model = PropertyEstimatorSubmission.parse_json(json_model)

        client_request_id = str(uuid.uuid4())

        while client_request_id in self._server_request_ids_per_client_id:
            client_request_id = str(uuid.uuid4())

        self._server_request_ids_per_client_id[client_request_id] = []

        # Pass the ids of the submitted requests back to the
        # client.
        encoded_job_ids = json.dumps(client_request_id).encode()
        length = pack_int(len(encoded_job_ids))

        await stream.write(length + encoded_job_ids)

        logging.info('Request id sent to the client ({}): {}'.format(address, client_request_id))

        server_requests, request_ids_to_launch = self._prepare_server_requests(client_data_model,
                                                                               client_request_id)

        # Keep track of which server request ids belong to which client
        # request id.
        for request_id in request_ids_to_launch:
            self._schedule_server_request(server_requests[request_id])

    async def _handle_job_query(self, stream, message_length):
        """An asynchronous routine for handling the receiving and processing
        of job queries from a client

        Parameters
        ----------
        stream: IOStream
            An IO stream used to pass messages between the
            server and client.
        message_length: int
            The length of the message being received.
        """

        encoded_request_id = await stream.read_bytes(message_length)
        client_request_id = encoded_request_id.decode()

        response = None

        if client_request_id not in self._server_request_ids_per_client_id:

            response = PropertyEstimatorException(directory='',
                                                  message=f'The {client_request_id} request id was not found '
                                                          f'on the server.')

        else:
            response = self._query_client_request_status(client_request_id)

        encoded_response = response.json().encode()
        length = pack_int(len(encoded_response))

        await stream.write(length + encoded_response)

    async def handle_stream(self, stream, address):
        """A routine to handle incoming requests from
        a property estimator TCP client.

        Notes
        -----
        This method is based on the StackOverflow response from
        A. Jesse Jiryu Davis: https://stackoverflow.com/a/40257248

        Parameters
        ----------
        stream: IOStream
            An IO stream used to pass messages between the
            server and client.
        address: str
            The address from which the request came.
        """
        # logging.info("Incoming connection from {}".format(address))

        try:
            while True:

                # Receive an introductory message with the message type.
                packed_message_type = await stream.read_bytes(4)
                message_type_int = unpack_int(packed_message_type)[0]

                packed_message_length = await stream.read_bytes(4)
                message_length = unpack_int(packed_message_length)[0]

                # logging.info('Introductory packet recieved: {} {}'.format(message_type_int, message_length))

                message_type = None

                try:
                    message_type = PropertyEstimatorMessageTypes(message_type_int)
                    # logging.info('Message type: {}'.format(message_type))

                except ValueError as e:

                    logging.info('Bad message type recieved: {}'.format(e))

                    # Discard the unrecognised message.
                    if message_length > 0:
                        await stream.read_bytes(message_length)

                    continue

                if message_type is PropertyEstimatorMessageTypes.Submission:
                    await self._handle_job_submission(stream, address, message_length)
                elif message_type is PropertyEstimatorMessageTypes.Query:
                    await self._handle_job_query(stream, message_length)

        except StreamClosedError:

            # Handle client disconnections gracefully.
            # logging.info("Lost connection to {}:{} : {}.".format(address, self._port, e))
            pass

    def _find_server_estimation_request(self, request):
        """Checks whether the server is currently, or has previously completed
        a request to estimate a set of properties for a particular substance
        using the same force field parameters and estimation options.

        Parameters
        ----------
        request: PropertyEstimatorServer.ServerEstimationRequest
            The request to check for.

        Returns
        -------
        str, optional
            The id of the existing request if one exists, otherwise None.
        """

        cached_request_id = request.id

        for existing_id in self._queued_calculations:

            request.id = existing_id

            if request.json() != self._queued_calculations[existing_id].json():
                continue

            request.id = cached_request_id
            return existing_id

        for existing_id in self._finished_calculations:

            request.id = existing_id

            if request.json() != self._finished_calculations[existing_id].json():
                continue

            request.id = cached_request_id
            return existing_id

        request.id = cached_request_id
        return None

    def _prepare_server_requests(self, client_data_model, client_request_id):
        """Turns a client estimation submission request into a form more useful
        to the server, namely a list of properties to estimate separated by
        system composition.

        Parameters
        ----------
        client_data_model: PropertyEstimatorSubmission
            The client data model.
        client_request_id: str
            The id that was assigned to the client request.

        Returns
        -------
        dict of str and PropertyEstimatorServer.ServerEstimationRequest
            A list of the requests to be calculated by the server.
        list of str
            The ids of the requests which haven't already been launched by
            the server.
        """

        force_field = client_data_model.force_field
        force_field_id = self._storage_backend.has_force_field(force_field)

        if force_field_id is None:

            force_field_id = str(uuid.uuid4())
            self._storage_backend.store_force_field(force_field_id, force_field)

        server_requests = {}

        # Split the full list of properties into lists partitioned by
        # substance.
        properties_by_substance = {}

        for physical_property in client_data_model.properties:

            if physical_property.substance.identifier not in properties_by_substance:
                properties_by_substance[physical_property.substance.identifier] = []

            properties_by_substance[physical_property.substance.identifier].append(physical_property)

        for substance_identifier in properties_by_substance:

            calculation_id = str(uuid.uuid4())

            # Make sure we don't somehow generate the same uuid
            # twice (although this is very unlikely to ever happen).
            while (calculation_id in self._queued_calculations or
                   calculation_id in self._finished_calculations):

                calculation_id = str(uuid.uuid4())

            properties_to_estimate = properties_by_substance[substance_identifier]

            request = self.ServerEstimationRequest(estimation_id=calculation_id,
                                                   queued_properties=properties_to_estimate,
                                                   options=client_data_model.options,
                                                   force_field_id=force_field_id)

            server_requests[calculation_id] = request

        request_ids_to_launch = []

        # Make sure this request is not already in the queue / has
        # already been completed, and if not add it to the list of
        # things to be queued.
        for server_request_id in server_requests:

            server_request = server_requests[server_request_id]
            existing_id = self._find_server_estimation_request(server_request)

            if existing_id is None:

                request_ids_to_launch.append(server_request_id)
                existing_id = server_request_id

                self._queued_calculations[server_request_id] = server_request

            self._server_request_ids_per_client_id[client_request_id].append(existing_id)

        return server_requests, request_ids_to_launch

    def _query_client_request_status(self, client_request_id):
        """Queries the current status of a client request by querying
        the state of the individual server requests it was split into.

        Parameters
        ----------
        client_request_id: str
            The id of the client request to query.

        Returns
        -------
        PropertyEstimatorResult
            The current results of the client request.
        """

        request_results = PropertyEstimatorResult(result_id=client_request_id)

        for server_request_id in self._server_request_ids_per_client_id[client_request_id]:

            server_request = None

            if server_request_id in self._queued_calculations:
                server_request = self._queued_calculations[server_request_id]

            elif server_request_id in self._finished_calculations:

                server_request = self._finished_calculations[server_request_id]

                if len(server_request.queued_properties) > 0:

                    return PropertyEstimatorException(message=f'An internal error occurred - the {server_request_id} '
                                                              f'was prematurely marked us finished.')

            else:

                return PropertyEstimatorException(message=f'An internal error occurred - the {server_request_id} '
                                                          f'request was not found on the server.')

            for physical_property in server_request.queued_properties:

                substance_id = physical_property.substance.identifier

                if substance_id not in request_results.queued_properties:
                    request_results.queued_properties[substance_id] = []

                request_results.queued_properties[substance_id].append(physical_property)

            for substance_id in server_request.unsuccessful_properties:

                physical_property = server_request.unsuccessful_properties[substance_id]

                if substance_id not in request_results.unsuccessful_properties:
                    request_results.unsuccessful_properties[substance_id] = []

                request_results.unsuccessful_properties[substance_id].append(physical_property)

            for substance_id in server_request.estimated_properties:

                physical_property = server_request.estimated_properties[substance_id]

                if substance_id not in request_results.estimated_properties:
                    request_results.estimated_properties[substance_id] = []

                request_results.estimated_properties[substance_id].append(physical_property)

            request_results.exceptions.extend(server_request.exceptions)

        return request_results

    def _schedule_server_request(self, server_request):
        """Schedules the estimation of the requested properties.

        This method will recursively cascade through all allowed calculation
        layers or until all properties have been calculated.

        Parameters
        ----------
        server_request : PropertyEstimatorServer.ServerEstimationRequest
            The object containing instructions about which calculations
            should be performed.
        """

        if len(server_request.options.allowed_calculation_layers) == 0 or \
           len(server_request.queued_properties) == 0:

            # Move any remaining properties to the unsuccessful list.
            for physical_property in server_request.queued_properties:

                substance_id = physical_property.substance.identifier

                if substance_id not in server_request.unsuccessful_properties:
                    server_request.unsuccessful_properties[substance_id] = []

                server_request.unsuccessful_properties[substance_id].append(physical_property)

                server_request.queued_properties = []

            self._queued_calculations.pop(server_request.id)
            self._finished_calculations[server_request.id] = server_request

            logging.info(f'Finished server request {server_request.id}')
            return

        current_layer_type = server_request.options.allowed_calculation_layers.pop(0)

        if current_layer_type not in available_layers:

            # Kill all remaining properties if we reach an unsupported calculation layer.
            error_object = PropertyEstimatorException(message=f'The {current_layer_type} layer is not '
                                                              f'supported by / available on the server.')

            server_request.exceptions.append(error_object)

            server_request.options.allowed_calculation_layers.append(current_layer_type)
            server_request.queued_properties = []

            self._schedule_server_request(server_request)
            return

        logging.info(f'Launching server request {server_request.id} using the {current_layer_type} layer')

        layer_directory = path.join(self._working_directory, current_layer_type, server_request.id)

        if not path.isdir(layer_directory):
            makedirs(layer_directory)

        current_layer = available_layers[current_layer_type]

        current_layer.schedule_calculation(self._calculation_backend,
                                           self._storage_backend,
                                           layer_directory,
                                           server_request,
                                           self._schedule_server_request)

    def start_listening_loop(self):
        """Starts the main (blocking) server IOLoop which will run until
        the user kills the process.
        """
        logging.info('Server listening at port {}'.format(self._port))

        try:
            IOLoop.current().start()
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops the property calculation server and it's
        provided backend.
        """
        self._calculation_backend.stop()
        IOLoop.current().stop()
