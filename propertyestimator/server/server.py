"""
The core functionality of the 'server' side of the
evaluator framework.
"""
import copy
import json
import logging
import os
import select
import socket
import threading
import traceback
import uuid

from propertyestimator.attributes import Attribute, AttributeClass
from propertyestimator.client import EvaluatorClient, RequestOptions, RequestResult
from propertyestimator.datasets import PhysicalProperty
from propertyestimator.forcefield import ParameterGradientKey
from propertyestimator.layers import registered_calculation_layers
from propertyestimator.storage import LocalFileStorage
from propertyestimator.utils.exceptions import EvaluatorException
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator.utils.tcp import (
    PropertyEstimatorMessageTypes,
    pack_int,
    recvall,
    unpack_int,
)

logger = logging.getLogger(__name__)


class Batch(AttributeClass):
    """Represents a batch of physical properties which are being estimated by
    the server for a given set of force field parameters.

    The expectation is that this object will be passed between calculation layers,
    whereby each layer will attempt to estimate each of the `queued_properties`.
    Those properties which can be estimated will be moved to the `estimated_properties`
    set, while those that couldn't will remain in the `queued_properties` set ready
    for the next layer.
    """

    id = Attribute(
        docstring="The unique id of this batch.",
        type_hint=str,
        default_value=lambda: str(uuid.uuid4()).replace("-", ""),
    )

    force_field_id = Attribute(
        docstring="The id of the force field being used to estimate"
        "this batch of properties.",
        type_hint=str,
    )
    options = Attribute(
        docstring="The options being used to estimate this batch.",
        type_hint=RequestOptions,
    )
    parameter_gradient_keys = Attribute(
        docstring="The parameters that this batch of physical properties "
        "should be differentiated with respect to.",
        type_hint=list,
    )

    queued_properties = Attribute(
        docstring="The set of properties which have yet to be estimated.",
        type_hint=list,
        default_value=[],
    )
    estimated_properties = Attribute(
        docstring="The set of properties which have been successfully estimated.",
        type_hint=list,
        default_value=[],
    )
    unsuccessful_properties = Attribute(
        docstring="The set of properties which have been could not be estimated.",
        type_hint=list,
        default_value=[],
    )
    exceptions = Attribute(
        docstring="The set of properties which have yet to be, or "
        "are currently being estimated.",
        type_hint=list,
        default_value=[],
    )

    def validate(self, attribute_type=None):
        super(Batch, self).validate(attribute_type)

        assert all(isinstance(x, PhysicalProperty) for x in self.queued_properties)
        assert all(isinstance(x, PhysicalProperty) for x in self.estimated_properties)
        assert all(
            isinstance(x, PhysicalProperty) for x in self.unsuccessful_properties
        )
        assert all(isinstance(x, EvaluatorException) for x in self.exceptions)
        assert all(
            isinstance(x, ParameterGradientKey) for x in self.parameter_gradient_keys
        )


class EvaluatorServer:
    """The object responsible for coordinating all properties estimations to
    be ran using the property estimator.

    This server is responsible for receiving estimation requests from the client,
    determining which calculation layer to use to launch the request, and
    distributing that estimation across the available compute resources.

    Notes
    -----
    Every client request is split into logical chunk batches. This enables batches
    of related properties (e.g. all properties for CO) to be estimated in one go
    (or one task graph in the case of workflow based layers) and returned when ready,
    rather than waiting for the full data set to complete.

    Examples
    --------
    Setting up a general server instance using a dask based calculation backend,
    and a local file storage backend:

    >>> # Create the backend which will be responsible for distributing the calculations
    >>> from propertyestimator.backends import DaskLocalCluster, ComputeResources
    >>> calculation_backend = DaskLocalCluster()
    >>> calculation_backend.start()
    >>>
    >>> # Create the server to which all estimation requests will be submitted
    >>> from propertyestimator.server import EvaluatorServer
    >>> property_server = EvaluatorServer(calculation_backend)
    >>>
    >>> # Instruct the server to listen for incoming requests
    >>> # This command will run until killed.
    >>> property_server.start()
    """

    def __init__(
        self,
        calculation_backend,
        storage_backend=None,
        port=8000,
        working_directory="working-data",
    ):
        """Constructs a new EvaluatorServer object.

        Parameters
        ----------
        calculation_backend: CalculationBackend
            The backend to use for executing calculations.
        storage_backend: StorageBackend, optional
            The backend to use for storing information from any calculations.
            If `None`, a default `LocalFileStorage` backend will be used.
        port: int
            The port on which to listen for incoming client requests.
        working_directory: str
            The local directory in which to store all local, temporary calculation data.
        """

        # Initialize the main 'server' attributes.
        self._port = port

        self._server_thread = None
        self._socket = None

        self._started = False
        self._stopped = True

        # Initialize the internal components.
        assert calculation_backend is not None and calculation_backend.started
        self._calculation_backend = calculation_backend

        if storage_backend is None:
            storage_backend = LocalFileStorage()

        self._storage_backend = storage_backend

        self._working_directory = working_directory
        os.makedirs(self._working_directory, exist_ok=True)

        self._queued_batches = {}
        self._finished_batches = {}

        self._batch_ids_per_client_id = {}

    def _query_request_status(self, client_request_id):
        """Queries the the current state of an estimation request
        and stores it in a `RequestResult`.

        Parameters
        ----------
        client_request_id: str
            The id of the request to query.

        Returns
        -------
        RequestResult
            The state of the request.
        EvaluatorException, optional
            The exception raised while retrieving the status,
            if any.
        """

        request_results = RequestResult()

        for batch_id in self._batch_ids_per_client_id[client_request_id]:

            # Find the batch.
            if batch_id in self._queued_batches:
                batch = self._queued_batches[batch_id]

            elif batch_id in self._finished_batches:

                batch = self._finished_batches[batch_id]

                if len(batch.queued_properties) > 0:

                    return (
                        None,
                        EvaluatorException(
                            message=f"An internal error occurred - the {batch_id} "
                            f"batch was prematurely marked us finished."
                        ),
                    )

            else:

                return (
                    None,
                    EvaluatorException(
                        message=f"An internal error occurred - the {batch_id} "
                        f"request was not found on the server."
                    ),
                )

            request_results.queued_properties.add_properties(*batch.queued_properties)
            request_results.unsuccessful_properties.add_properties(
                *batch.unsuccessful_properties
            )
            request_results.estimated_properties.add_properties(
                *batch.estimated_properties
            )
            request_results.exceptions.extend(batch.exceptions)

        return request_results, None

    def _prepare_batches(self, submission, request_id):
        """Turns an estimation request into chunked batches to
        calculate separately.

        This enables batches of related properties (e.g. all properties
        for CO) to be estimated in one go (or one task graph in the case
        of workflow based layers) and returned when ready, rather than waiting
        for the full data set to complete.

        Parameters
        ----------
        submission: EvaluatorClient._Submission
            The full request submission.
        request_id: str
            The id that was assigned to the request.

        Returns
        -------
        list of Batch
            A list of the batches to launch.
        """

        force_field_source = submission.force_field_source
        force_field_id = self._storage_backend.store_force_field(force_field_source)

        batches = []

        # Batch properties to be estimated for the same substance
        # into one chunk
        for substance in submission.dataset.substances:

            batch = Batch()
            batch.force_field_id = force_field_id

            # Make sure we don't somehow generate the same uuid
            # twice (although this is very unlikely to ever happen).
            while (
                batch.id in self._queued_batches or batch.id in self._finished_batches
            ):

                batch.id = str(uuid.uuid4()).replace("-", "")

            batch.queued_properties = [
                x for x in submission.dataset.properties_by_substance(substance)
            ]
            batch.options = RequestOptions.parse_json(submission.options.json())

            batch.parameter_gradient_keys = copy.deepcopy(
                submission.parameter_gradient_keys
            )

            batches.append(batch)

            self._queued_batches[batch.id] = batch
            self._batch_ids_per_client_id[request_id].append(batch.id)

        return batches

    def _launch_batch(self, batch):
        """Launch a batch of properties to estimate.

        This method will recursively cascade through all allowed calculation
        layers or until all properties have been calculated.

        Parameters
        ----------
        batch : Batch
            The batch to launch.
        """

        if (
            len(batch.options.calculation_layers) == 0
            or len(batch.queued_properties) == 0
        ):

            # Move any remaining properties to the unsuccessful list.
            batch.unsuccessful_properties = [*batch.queued_properties]
            batch.queued_properties = []

            self._queued_batches.pop(batch.id)
            self._finished_batches[batch.id] = batch

            logger.info(f"Finished server request {batch.id}")
            return

        current_layer_type = batch.options.calculation_layers.pop(0)

        if current_layer_type not in registered_calculation_layers:

            # Add an exception if we reach an unsupported calculation layer.
            error_object = EvaluatorException(
                message=f"The {current_layer_type} layer is not "
                f"supported by / available on the server."
            )

            batch.exceptions.append(error_object)
            self._launch_batch(batch)
            return

        logger.info(f"Launching batch {batch.id} using the {current_layer_type} layer")

        layer_directory = os.path.join(
            self._working_directory, current_layer_type, batch.id
        )
        os.makedirs(layer_directory, exist_ok=True)

        current_layer = registered_calculation_layers[current_layer_type]

        current_layer.schedule_calculation(
            self._calculation_backend,
            self._storage_backend,
            layer_directory,
            batch,
            self._launch_batch,
        )

    def _handle_job_submission(self, connection, address, message_length):
        """An asynchronous routine for handling the receiving and processing
        of job submissions from a client.

        Parameters
        ----------
        connection:
            An IO stream used to pass messages between the
            server and client.
        address: str
            The address from which the request came.
        message_length: int
            The length of the message being received.
        """

        logger.info("Received estimation request from {}".format(address))

        # Read the incoming request from the server. The first four bytes
        # of the response should be the length of the message being sent.

        # Decode the client submission json.
        encoded_json = recvall(connection, message_length)
        json_model = encoded_json.decode()

        request_id = None
        error = None

        try:

            # noinspection PyProtectedMember
            submission = EvaluatorClient._Submission.parse_json(json_model)
            submission.validate()

        except Exception as e:

            formatted_exception = traceback.format_exception(None, e, e.__traceback__)

            error = EvaluatorException(
                message=f"An exception occured when parsing "
                f"the submission: {formatted_exception}"
            )

            submission = None

        if error is None:

            while request_id is None or request_id in self._batch_ids_per_client_id:
                request_id = str(uuid.uuid4()).replace("-", "")

            self._batch_ids_per_client_id[request_id] = []

        # Pass the id of the submitted requests back to the client
        # as well as any error which may have occurred.
        return_packet = json.dumps((request_id, error), cls=TypedJSONEncoder)

        encoded_return_packet = return_packet.encode()
        length = pack_int(len(encoded_return_packet))

        connection.sendall(length + encoded_return_packet)

        if error is not None:
            # Exit early if there is an error.
            return

        # Batch the request into more managable chunks.
        batches = self._prepare_batches(submission, request_id)

        # Launch the batches
        for batch in batches:
            self._launch_batch(batch)

    def _handle_job_query(self, connection, message_length):
        """An asynchronous routine for handling the receiving and
        processing of request status queries from a client

        Parameters
        ----------
        connection:
            An IO stream used to pass messages between the
            server and client.
        message_length: int
            The length of the message being received.
        """

        encoded_request_id = recvall(connection, message_length)
        client_request_id = encoded_request_id.decode()

        response = None

        if client_request_id not in self._batch_ids_per_client_id:

            error = EvaluatorException(
                message=f"The request id ({client_request_id}) was not found "
                f"on the server.",
            )

        else:
            response, error = self._query_request_status(client_request_id)

        response_json = json.dumps((response, error), cls=TypedJSONEncoder)

        encoded_response = response_json.encode()
        length = pack_int(len(encoded_response))

        connection.sendall(length + encoded_response)

    def _handle_stream(self, connection, address):
        """A routine to handle incoming requests from
        a TCP client.
        """

        # Receive an introductory message with the message type.
        packed_message_type = recvall(connection, 4)
        message_type_int = unpack_int(packed_message_type)[0]

        packed_message_length = recvall(connection, 4)
        message_length = unpack_int(packed_message_length)[0]

        try:
            message_type = PropertyEstimatorMessageTypes(message_type_int)
        except ValueError as e:

            trace = traceback.format_exception(None, e, e.__traceback__)
            logger.info(f"Bad message type received: {trace}")

            # Discard the unrecognised message.
            if message_length > 0:
                recvall(connection, message_length)

            return

        if message_type is PropertyEstimatorMessageTypes.Submission:
            self._handle_job_submission(connection, address, message_length)
        elif message_type is PropertyEstimatorMessageTypes.Query:
            self._handle_job_query(connection, message_length)

    def _handle_connections(self):
        """Handles incoming client TCP connections.
        """
        to_read = [self._socket]

        try:

            while not self._stopped:

                ready, _, _ = select.select(to_read, [], [], 0.1)

                for data in ready:

                    if data == self._socket:
                        connection, address = self._socket.accept()
                        to_read.append(connection)

                    else:

                        connection = data
                        self._handle_stream(connection, connection.getpeername())
                        connection.close()

                        to_read.remove(data)

        except Exception:
            logger.exception(f"Fatal error in the main server loop")

    def start(self, asynchronous=False):
        """Instructs the server to begin listening for incoming
        requests from any `EvaluatorClients`.

        Parameters
        ----------
        asynchronous: bool
            If `True` the server will run on a separate thread in the background,
            returning control back to the main thread. Otherwise, this function
            will block the main thread until this server is killed.
        """

        if self._started:
            raise RuntimeError("The server has already been started.")

        logger.info("Server listening at port {}".format(self._port))
        self._started = True
        self._stopped = False

        # Create the TCP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("localhost", self._port))
        self._socket.listen(128)

        try:

            if asynchronous:

                self._server_thread = threading.Thread(
                    target=self._handle_connections, daemon=True
                )
                self._server_thread.start()

            else:
                self._handle_connections()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stops the property calculation server and it's
        provided backend.
        """
        if not self._started:
            raise ValueError("The server has not yet been started.")

        self._stopped = True
        self._started = False

        if self._server_thread is not None:

            self._server_thread.join()
            self._server_thread = None

        if self._socket is not None:

            self._socket.close()
            self._socket = None

    def __enter__(self):
        self.start(asynchronous=True)
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):

        if self._started and not self._stopped:
            self.stop()
