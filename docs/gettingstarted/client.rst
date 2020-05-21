.. |evaluator_server|           replace:: :py:class:`~openff.evaluator.server.EvaluatorServer`
.. |evaluator_client|           replace:: :py:class:`~openff.evaluator.client.EvaluatorClient`
.. |evaluator_exception|        replace:: :py:class:`~openff.evaluator.utils.exceptions.EvaluatorException`
.. |connection_options|         replace:: :py:class:`~openff.evaluator.client.ConnectionOptions`

.. |request|                    replace:: :py:class:`~openff.evaluator.client.Request`
.. |request_result|             replace:: :py:class:`~openff.evaluator.client.RequestResult`
.. |request_options|            replace:: :py:class:`~openff.evaluator.client.RequestOptions`

.. |force_field_source|         replace:: :py:class:`~openff.evaluator.forcefield.ForceFieldSource`

.. |request_estimate|           replace:: :py:meth:`~openff.evaluator.client.EvaluatorClient.request_estimate`
.. |default_request_options|    replace:: :py:meth:`~openff.evaluator.client.EvaluatorClient.default_request_options`

.. |calculation_layers|         replace:: :py:class:`~openff.evaluator.client.RequestOptions.calculation_layers`
.. |calculation_schemas|        replace:: :py:class:`~openff.evaluator.client.RequestOptions.calculation_schemas`

.. |future|                     replace:: :py:class:`~asyncio.Future`

.. |smirnoff_force_field_source|       replace:: :py:class:`~openff.evaluator.forcefield.SmirnoffForceFieldSource`
.. |lig_par_gen_force_field_source|    replace:: :py:class:`~openff.evaluator.forcefield.LigParGenForceFieldSource`
.. |tleap_force_field_source|          replace:: :py:class:`~openff.evaluator.forcefield.TLeapForceFieldSource`

.. |build_smirnoff_system|             replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildSmirnoffSystem`
.. |build_tleap_system|                replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildTLeapSystem`
.. |build_lig_par_gen_system|          replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildLigParGenSystem`

.. |workflow_calculation_schema|       replace:: :py:class:`~openff.evaluator.layers.workflow.WorkflowCalculationSchema`

Evaluator Client
================

The |evaluator_client| object is responsible for both submitting requests to estimate a data set of properties to
a running :doc:`server` instance, and for pulling back the results of those requests when complete.

An |evaluator_client| object may optionally be created using a set of |connection_options| which specifies the network
address of the running :doc:`server` instance to connect to::

    # Specify the address of a server running on the local machine.
    connection_options = ConnectionOptions(server_address="localhost", server_port=8000)
    # Create the client object
    evaluator_client = EvaluatorClient(connection_options)

Requesting Estimates
--------------------

The client can request the estimation of a data set of properties using the |request_estimate| function::

    # Specify the data set.
    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(...)

    # Specify the force field source.
    force_field = SmirnoffForceFieldSource.from_path("openff-1.0.0.offxml")

    # Specify some estimation options (optional).
    options = client.default_request_options(data_set, force_field)

    # Specify the parameters to differentiate with respect to (optional).
    gradient_keys = [
        ParameterGradientKey(tag="vdW", smirks="[#6X4:1]", attribute="epsilon")
    ]

    # Request the estimation of the data set.
    request, errors = evaluator_client.request_estimate(
        data_set,
        force_field,
        options,
        gradient_keys
    )

A request must at minimum specify:

.. rst-class:: spaced-list

    * the :doc:`data set <../datasets/physicalproperties>` of physical properties to estimate.
    * the :ref:`force field parameters <gettingstarted/client:Force Field Sources>` to estimate the data set using.

and may also optionally specify:

.. rst-class:: spaced-list

    * the :ref:`options <gettingstarted/client:Request Options>` to use when estimating the property set.
    * the parameters to differentiate each physical property estimate with respect to.

.. note:: Gradients can currently only be computed for requests using a `SMIRNOFF <https://open-forcefield-toolkit.
  readthedocs.io/en/latest/smirnoff.html>`_ based force field.

The |request_estimate| function returns back two objects:

.. rst-class:: spaced-list

    * a |request| object which can be used to retrieve the results of the request and,
    * an |evaluator_exception| object which will be populated if any errors occured while submitting the request.

The |request| object is similar to a |future| object, in that it is an object which can be used to query the current
status of a request either asynchronously::

    results = request.results(synchronous=False)

or synchronously::

    results = request.results(synchronous=True)

The results (which may currently be incomplete) are returned back as a |request_result| object.

The |request| object is fully JSON serializable::

    # Save the request to JSON
    request.json(file_path="request.json", format=True)
    # Load the request from JSON
    request = Request.from_json(file_path="request.json")

making it easy to keep track of any open requests.

Request Options
---------------

The |request_options| object allows greater control over how properties are estimated by the server. It currently allows
control over:

.. rst-class:: spaced-list

    * |calculation_layers|: The :doc:`calculation layers <../layers/calculationlayers>` which the server should attempt
      to use when estimating the data set. The order which the layers are specified in this list is the order which
      the server will attempt to use each layer.
    * |calculation_schemas|: The :ref:`calculation schemas <layers/calculationlayers:Defining a Calculation Layer>` to
      use for each allowed calculation layer per class of property. These will be automatically populated in the cases
      where no user specified schema is provided, and where a default schema has been registered with the plugin system
      for the particular layer and property type.

If no options are passed to |request_estimate| a default set will be generated through a call to
|default_request_options|. For more information about how default calculation schemas are registered, see the
:ref:`layers/calculationlayers:Default Schemas` section.

Force Field Sources
-------------------

Different force field representations (e.g. ``SMIRNOFF``, ``TLeap``, ``LigParGen``) are defined within the framework as
|force_field_source| objects. A force field source should specify *all* of the options which would be required by
a particular force field, such as the non-bonded cutoff or the charge scheme if not specified directly in the force
field itself.

Currently the framework has built in support for force fields applied via:

.. rst-class:: spaced-list

    * the `OpenFF toolkit <https://open-forcefield-toolkit.readthedocs.io/en/latest/>`_ (|smirnoff_force_field_source|).
    * the ``tleap`` program from the `AmberTools suite <https://ambermd.org/AmberTools.php>`_
      (|lig_par_gen_force_field_source|).
    * an instance of the `LigParGen server <http://zarbi.chem.yale.edu/ligpargen/>`_ (|lig_par_gen_force_field_source|).

The client will automatically adapt any of the built-in calculation schemas which are based off of the
|workflow_calculation_schema| to use the correct workflow protocol (|build_smirnoff_system|, |build_tleap_system| or
|build_lig_par_gen_system|) for the requested force field.
