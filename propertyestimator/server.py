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

    class SubstanceEstimationRequest(TypedBaseModel):
        """Represents a set of properties for a single substance to be estimated by the
        server, along with the options which should be used when running the calculations.
        """

        def __init__(self, estimation_id='', queued_properties=None, options=None, force_field_id=None):
            """Constructs a new SubstanceEstimationRequest object.

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

            self.substance_identifier = None

            if len(self.queued_properties) > 0:
                self.substance_identifier = self.queued_properties[0].substance.identifier

            self.estimated_properties = {}
            self.unsuccessful_properties = {}

            self.options = options

            self.force_field_id = force_field_id

        def __getstate__(self):
            return {
                'id': self.id,
                'substance_identifier': self.substance_identifier,

                'queued_properties': self.queued_properties,

                'estimated_properties': self.estimated_properties,
                'unsuccessful_properties': self.unsuccessful_properties,

                'options': self.options,

                'force_field_id': self.force_field_id,
            }

        def __setstate__(self, state):
            self.id = state['id']
            self.substance_identifier = state['substance_identifier']

            self.queued_properties = state['queued_properties']

            self.estimated_properties = state['estimated_properties']
            self.unsuccessful_properties = state['unsuccessful_properties']

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

        assert self._calculation_backend is not None and self._storage_backend is not None

        self._calculation_backend = calculation_backend
        self._storage_backend = storage_backend

        self.working_directory = working_directory

        if not path.isdir(self.working_directory):
            makedirs(self.working_directory)

        self._port = port

        self._queued_calculations = {}
        self._finished_calculations = {}

        self._property_request_ids_per_id = {}

        self._periodic_loops = []

        super().__init__()

        # self.listen(self._port)
        self.bind(self._port)
        self.start(1)

        calculation_backend.start()

    def _find_calculation_request(self, request):
        """Checks whether the server is currently, or has previously completed
        a given request.

        Parameters
        ----------
        request: PropertyEstimatorServer.SubstanceEstimationRequest
            The request to check for

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
            The length of the message being recieved.
        """

        logging.info('Received job request from {}'.format(address))

        # Read the incoming request from the server. The first four bytes
        # of the response should be the length of the message being sent.

        # Decode the client submission json.
        encoded_json = await stream.read_bytes(message_length)
        json_model = encoded_json.decode()

        # TODO: Add exception handling so the server can gracefully reject bad json.
        client_data_model = PropertyEstimatorSubmission.parse_json(json_model)
        per_substance_requests = self._prepare_per_substance_requests(client_data_model)

        unique_id = str(uuid.uuid4())
        self._property_request_ids_per_id[unique_id] = []

        request_ids_to_launch = []

        # Make sure this request is not already in the queue / has
        # already been completed, and if not add it to the list of
        # things to be queued.
        for per_substance_request_id in per_substance_requests:

            per_substance_request = per_substance_requests[per_substance_request_id]
            existing_id = self._find_calculation_request(per_substance_request)

            if existing_id is None:

                request_ids_to_launch.append(per_substance_request.id)
                existing_id = per_substance_request_id.id

                self._queued_calculations[existing_id] = per_substance_request

            self._property_request_ids_per_id[unique_id].append(existing_id)

        # Pass the ids of the submitted calculations back to the
        # client.
        encoded_job_ids = json.dumps(unique_id).encode()
        length = pack_int(len(encoded_job_ids))

        await stream.write(length + encoded_job_ids)

        logging.info('Jobs ids sent to the client ({}): {}'.format(address, unique_id))

        for request_id in request_ids_to_launch:
            self._schedule_calculation(per_substance_requests[request_id])

    async def _handle_job_query(self, stream, message_length):
        """An asynchronous routine for handling the receiving and processing
        of job queries from a client

        Parameters
        ----------
        stream: IOStream
            An IO stream used to pass messages between the
            server and client.
        message_length: int
            The length of the message being recieved.
        """

        # logging.info('Received job query from {}'.format(address))

        encoded_request_id = await stream.read_bytes(message_length)
        request_id = encoded_request_id.decode()

        # logging.info('Looking up request id {}'.format(request_id))

        response = None

        if request_id not in self._property_request_ids_per_id:

            response = PropertyEstimatorException(directory='',
                                                  message='The {} request id was not found '
                                                           'on the server.'.format(request_id)).json()

        elif request_id in self._finished_calculations:
            response = self._finished_calculations[request_id].json()

        else:
            response = ''

        encoded_response = response.encode()
        length = pack_int(len(encoded_response))

        await stream.write(length + encoded_response)

        # logging.info('Job results sent to the client {}: {}'.format(address, response))

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

    def _prepare_per_substance_requests(self, client_data_model):
        """Turns a client estimation submission request into a form more useful
        to the server, namely a list of properties to estimate separated by
        system composition.

        Parameters
        ----------
        client_data_model: PropertyEstimatorSubmission
            The client data model.

        Returns
        -------
        dict of str and PropertyEstimatorServer.SubstanceEstimationRequest
            A list of the requests to be calculated by the server.
        """

        force_field = client_data_model.force_field

        force_field_id = self._storage_backend.has_force_field(force_field)

        if force_field_id is None:

            force_field_id = str(uuid.uuid4())
            self._storage_backend.store_force_field(force_field_id, force_field)

        return_data = {}

        for substance_identifier in client_data_model.properties:

            calculation_id = str(uuid.uuid4())

            # Make sure we don't somehow generate the same uuid
            # twice (although this is very unlikely to ever happen).
            while calculation_id in self._queued_calculations or calculation_id in self._finished_calculations:
                calculation_id = str(uuid.uuid4())

            properties_to_estimate = client_data_model.properties[substance_identifier]

            request = self.SubstanceEstimationRequest(estimation_id=calculation_id,
                                                      queued_properties=properties_to_estimate,
                                                      options=client_data_model.options,
                                                      force_field_id=force_field_id)

            return_data[substance_identifier] = request

        return return_data

    def _gather_finished_calculations(self, request_id):

        # output_model = PropertyEstimatorResult(result_id=server_data_model.id)
        #
        # output_model.estimated_properties = server_data_model.estimated_properties
        # output_model.unsuccessful_properties = server_data_model.unsuccessful_properties

        return None

    def _schedule_calculation(self, data_model):
        """Schedules the calculation of the given properties using the passed
        parameters.

        This method will recursively cascade through all allowed calculation
        layers or until all properties have been calculated.

        Parameters
        ----------
        data_model : PropertyEstimatorServer.SubstanceEstimationRequest
            The object containing instructions about which calculations
            should be performed.
        """

        if len(data_model.options.allowed_calculation_layers) == 0 or \
           len(data_model.queued_properties) == 0:

            self._queued_calculations.pop(data_model.id)
            self._finished_calculations[data_model.id] = data_model

            logging.info(f'Finished calculation {data_model.id}')
            return

        current_layer_type = data_model.options.allowed_calculation_layers.pop(0)

        if current_layer_type not in available_layers:

            # Kill all remaining properties if we reach an unsupported calculation layer.
            error_object = PropertyEstimatorException(message=f'The {current_layer_type} layer is not '
                                                              f'supported by the server.')

            for queued_calculation in data_model.queued_properties:
                data_model.unsuccessful_properties[queued_calculation.id] = error_object

            data_model.options.allowed_calculation_layers.append(current_layer_type)
            data_model.queued_properties = []

            self._schedule_calculation(data_model)
            return

        logging.info(f'Launching calculation {data_model.id} using the {current_layer_type} layer')

        layer_directory = path.join(self.working_directory, current_layer_type, data_model.id)

        if not path.isdir(layer_directory):
            makedirs(layer_directory)

        current_layer = available_layers[current_layer_type]

        current_layer.schedule_calculation(self._calculation_backend,
                                           self._storage_backend,
                                           layer_directory,
                                           data_model,
                                           self._schedule_calculation)

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

        for periodic_loop in self._periodic_loops:
            periodic_loop.stop()

        IOLoop.current().stop()
