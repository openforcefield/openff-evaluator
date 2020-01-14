Property Data Sets
==================

A ``PhysicalPropertyDataSet`` is a collection of ``PhysicalProperty`` objects. It can be easily stored as /
created from JSON::

    data_set = PhysicalPropertyDataset.from_json("data_set.json")

or for convenience, can be retrieved as a pandas `DataFrame <https://pandas.pydata.org/pandas-docs/stable/
generated/pandas.DataFrame.html>`_::

    data_set.to_pandas()

The framework implements specific data set objects for a number of open data sources, such as the ``ThermoMLDataSet``
which provides utilities for extracting the data from the `NIST ThermoML Archive <http://trc.nist.gov/ThermoML.html>`_
and converting it into the standard framework objects.

Physical Properties
-------------------

The ``PhysicalProperty`` object describes a measured property of substance, and is defined by a combination of a:

* ``Substance`` specifying the substance that the measurement was collected for.
* ``ThermodynamicState`` specifying the thermodynamic conditions under which the measurement was performed

and the measured value, as well as optionally

* the uncertainty in the value of the property.
* a list of ``ParameterGradient`` which defines the gradient of the property with respect to the model parameters
  if it was computationally estimated.
* a ``Source`` specifying the source (either experimental or computational) and provenance of the measurement.

Each type of property supported by the framework, such as a density of an enthalpy of vaporization, must have it's own
class representation which inherits from ``PhysicalProperty``::

    # Define the substance
    water = Substance.from_components("O")
    # Define thermodynamic state
    thermodynamic_state = ThermodynamicState(
        pressure=1.0*unit.atmospheres, temperature=298.15*unit.kelvin
    )

    # Define a density measurement
    density = Density(
        substance=substance,
        thermodynamic_state=thermodynamic_state,
        value=1.0*unit.gram/unit.millilitre,
        uncertainty=0.0001*unit.gram/unit.millilitre
    )

    # Add the property to a data set
    data_set = PhysicalPropertyDataset()
    data_set.add_properties(density)

Substances
----------

A ``Substance`` is defined by a number of components (which may have specific roles assigned to them such as
being solutes in the system) and the amount of each component in the substance.

To create a pure substance containing only water::

    water_substance = Substance.from_components("O")

To create binary mixture of water and methanol in a 20:80 ratio::

    binary_mixture = Substance()
    binary_mixture.add_component(Component(smiles="O"), MoleFraction(value=0.2))
    binary_mixture.add_component(Component(smiles="CO"), MoleFraction(value=0.8))

To create a substance of an infinitely dilute paracetamol solute dissolved in water::

    solution = Substance()
    solution.add_component(
        Component(smiles="O", role=Component.Role.Solvent), MoleFraction(value=1.0)
    )
    solution.add_component(
        Component(smiles="CC(=O)Nc1ccc(O)cc1", role=Component.Role.Solute), ExactAmount(value=1)
    )

Thermodynamic States
--------------------

A ``ThermodynamicState`` specifies a combination of the temperature and (optionally) the pressure at which a
measurement is performed::

    thermodynamic_state = ThermodynamicState(
        temperature=298.15*unit.kelvin, pressure=1.0*unit.atmosphere
    )

