Physical property datasets
==================================

.. warning:: This text is now out of date, but will be updated in future to reflect the
             latest version of the framework.

A ``PhysicalPropertyDataset`` is a collection of ``MeasuredPhysicalProperty`` objects that are related in some way.

.. code-block:: python

    dataset = PhysicalPropertyDataset([measurement1, measurement2])

The dataset is iterable:

.. code-block:: python

    dataset = PhysicalPropertyDataset([measurement1, measurement2])

    for measurement in dataset:
        print measurement.value

and has accessors to retrieve DOIs and references associated with measurements in the dataset:

.. code-block:: python

    # Print the DOIs associated with this dataset
    print(dataset.DOIs)

    # Print the references associated with this dataset
    print(dataset.references)

For convenience, you can retrieve the dataset as a `pandas DataFrame <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_:

.. code-block:: python

    dataset.to_pandas()

ThermoML datasets
-----------------

A ``ThermoMLDataset`` object represents a physical property dataset stored in the IUPAC-standard
`ThermoML <http://trc.nist.gov/ThermoMLRecommendations.pdf>`_) for specifying thermodynamic properties in XML format.
``ThermoMLDataset`` is a subclass of ``PhysicalPropertyDataset``, and provides the same API interface (in addition to
some ThermoML-specfic methods).

Direct access to the `NIST ThermoML Archive <http://trc.nist.gov/ThermoML.html>`_ is
supported for obtaining physical property measurements in this format directly from the NIST TRC repository.

For example, to retrieve `the ThermoML dataset <http://trc.boulder.nist.gov/ThermoML/10.1016/j.jct.2005.03.012>`_ that
accompanies `this paper <http://www.sciencedirect.com/science/article/pii/S0021961405000741>`_, we can simply use the
DOI ``10.1016/j.jct.2005.03.012`` as a key for creating a ``PhysicalPropertyDataset`` subclassed object from the
ThermoML Archive:

.. code-block:: python

    dataset = ThermoMLDataset(doi='10.1016/j.jct.2005.03.012')

You can also specify multiple ThermoML Archive keys to create a dataset from multiple ThermoML files:

.. code-block:: python

    thermoml_keys = ['10.1021/acs.jced.5b00365', '10.1021/acs.jced.5b00474']
    dataset = ThermoMLDataset(doi=thermoml_keys)

It is also possible to specify ThermoML datasets housed at other locations, such as

.. code-block:: python

    dataset = ThermoMLDataset(url='http://openforcefieldgroup.org/thermoml-datasets')

or

.. code-block:: python

    dataset = ThermoMLDataset(url='file:///Users/choderaj/thermoml')

or

.. code-block:: python

    dataset = ThermoMLDataset(doi=['10.1021/acs.jced.5b00365', '10.1021/acs.jced.5b00474'],
                              url='http://openforcefieldgroup.org/thermoml-datasets')

or from ThermoML and a different URL:

.. code-block:: python

    dataset = ThermoMLDataset(doi=thermoml_keys)
    dataset.retrieve(doi=local_keys, url='http://openforcefieldgroup.org/thermoml-datasets')

You can see which DOIs contribute to the current ``ThermoMLDataset`` with the convenience functions:

.. code-block:: python

    print(dataset.DOIs)

NIST has compiled a JSON frame of corrections to uncertainties.

These can be used to update or correct data uncertainties and discard outliers using ``applyNISTUncertainties()``:

.. code-block:: python

    # Modify uncertainties according to NIST evaluation
    dataset.apply_nist_uncertainties(nist_uncertainties, adjust_uncertainties=True, discard_outliers=True)

.. todo::

    * We should merge any other useful parts parts of the `ThermoPyL API <https://github.com/choderalab/thermopyl>`_
      in here.

Other datasets
--------------

In future, we will add interfaces to other online datasets, such as

* `BindingDB <https://www.bindingdb.org/bind/index.jsp>`_ for retrieving
  `host-guest binding affinity <https://www.bindingdb.org/bind/HostGuest.jsp>`_ datasets.
