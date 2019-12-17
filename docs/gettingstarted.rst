Getting Started
===============

The ``propertyestimator`` currently exists as two key components:

* a client object which the user can use to request the estimation of data sets of
  physical properties.

* a server object which accepts requests from a client and performs the estimations.

.. warning:: These instructions are still a work in progress, and may not run as expected.

Creating an Estimator Server
----------------------------

The ``EvaluatorServer`` class creates objects that handle property estimation of all of the properties in a
dataset given a set.

Create the file ``run_server.py``. Tell server to log to file in case of failure::

    setup_timestamp_logging()

Create directory structure to store intermediary results::

    # Set the name of the directory in which all temporary files
    # will be generated.
    working_directory = 'working_directory'

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

Set up a calculation backend. Different backends will take different optional arguments, but here is
an example that will launch a backend with a single worker process::

    # Create a calculation backend to perform workflow
    # calculations on.
    calculation_backend = DaskLocalCluster(1)

Set up storage the storage backend which will cache any generated simulation data::

    # Create a backend to handle storing and retrieving
    # cached simulation data.
    storage_backend = LocalFileStorage()

Start the server running::

    # Create a server instance.
    property_server = server.EvaluatorServer(calculation_backend,
                                                     storage_backend,
                                                     working_directory=working_directory)

    # Tell the server to start listening for incoming
    # estimation requests.
    property_server.start_listening_loop()

To start the server, call the following command from the command line::

    python run_server.py

The server will wait for requests until killed.

Submitting Estimation Requests
------------------------------

Create the file ``run_client.py`` Load in the data set of properties to estimate, and the force field parameters to
use in the calculations::

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'))

    # Load in the force field to use.
    force_field_source = SmirnoffForceFieldSource.from_path('smirnoff99Frosst-1.1.0.offxml')

Create the client object and use it to send the estimation request to the server::

    # Create the client object.
    property_estimator = client.EvaluatorClient()
    # Submit the request to a running server.
    result = property_estimator.request_estimate(data_set, force_field_source)

Query the result until all of the properties have either been estimated or have errored::

    # Wait for the results synchronously.
    results = request.results(True)
    logging.info('The server has returned a response: {}'.format(result))

Save the results to a file::

    with open('results.json', 'w') as file:

        json_results = json.dump(results, file, sort_keys=True, indent=2,
                                 separators=(',', ': '), cls=TypedJSONEncoder)
