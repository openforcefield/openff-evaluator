Getting Started
===============

Overview - currently consists of two parts. A server which will handle all estimation requests,
coordinate with compute backend to perform calculations and cache data using a storage backend.

Creating an Estimator Server
----------------------------

The :obj:`PropertyEstimatorClient` class creates objects that handle property estimation of all of the properties in a dataset,
given a set or sets of parameters. The implementation will isolate the user from whatever backend (local machine,
HPC cluster, `XSEDE resources <http://xsede.org>`_, `Amazon EC2 <https://aws.amazon.com/ec2>`_) is being used to compute
the properties, as well as whether new simulations are being launched and analyzed or existing simulation data is being
reweighted.

Create file server.py. Tell server to log to file in case of failure

    setup_timestamp_logging()

Create dir structure::

    # Set the name of the directory in which all temporary files
    # will be generated.
    working_directory = 'working_directory'

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

Set up a calculation backend, different backends will take different optional arguments, but here is
an example that will launch and use 10 worker processes on a cluster::

    # Create a calculation backend to perform workflow
    # calculations on.
    calculation_backend = DaskLocalClusterBackend(1)

Set up storage::

    # Create a backend to handle storing and retrieving
    # cached simulation data.
    storage_backend = LocalFileStorage()

Start the server running::

    # Create a server instance.
    property_server = server.PropertyEstimatorServer(calculation_backend,
                                                     storage_backend,
                                                     working_directory=working_directory)

    # Tell the server to start listening for incoming
    # estimation requests.
    property_server.start_listening_loop()

Server will wait for requests until killed.

Submitting Estimation Requests
------------------------------

Here, ``dataset`` is a ``PhysicalPropertyDataset`` or subclass, and ``force_fields`` is a list containing
``ForceField`` objects used to parameterize the physical systems in the dataset.

This can be a single parameter set or multiple (usually closely related) parameter sets::

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'))

    # Load in the force field to use.
    force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

Create the client object::

    # Create the client object.
    property_estimator = client.PropertyEstimatorClient()
    # Submit the request to a running server, and wait for the results.
    result = property_estimator.request_estimate(data_set, force_field)

Query the result until all finished or errored::

    logging.info('The server has returned a response: {}'.format(result))

``PropertyEstimatorClient.computeProperties(...)`` returns a list of ``ComputedPhysicalProperty`` objects that provide access
to several pieces of information:

* ``property.value`` - the computed property value, with appropriate units
* ``property.uncertainty`` - the statistical uncertainty in the computed property
* ``property.parameters`` - a reference to the parameter set used to compute this property
* ``property.property`` - a reference to the corresponding ``MeasuredPhysicalProperty`` this property was computed for
