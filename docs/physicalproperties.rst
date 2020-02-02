.. |physical_property_data_set|    replace:: :py:class:`~propertyestimator.datasets.PhysicalPropertyDataSet`
.. |physical_property|             replace:: :py:class:`~propertyestimator.datasets.PhysicalProperty`
.. |property_phase|                replace:: :py:class:`~propertyestimator.datasets.PropertyPhase`
.. |source|                        replace:: :py:class:`~propertyestimator.datasets.Source`

.. |thermoml_data_set|             replace:: :py:class:`~propertyestimator.datasets.thermoml.ThermoMLDataSet`

.. |substance|                     replace:: :py:class:`~propertyestimator.substances.Substance`
.. |thermodynamic_state|           replace:: :py:class:`~propertyestimator.thermodynamics.ThermodynamicState`
.. |force_field_source|            replace:: :py:class:`~propertyestimator.forcefield.ForceFieldSource`

.. |parameter_gradient|            replace:: :py:class:`~propertyestimator.forcefield.ParameterGradient`

.. |data_frame|                    replace:: :py:class:`~pandas.DataFrame`

Property Data Sets
==================

A |physical_property_data_set| is a collection of measured physical properties encapsulated as :ref:`physical property
<physicalproperties:Physical Properties>` objects. They may be created from scratch::

    # Define a density measurement
    density = Density(
        substance=Substance.from_components("O"),
        thermodynamic_state=ThermodynamicState(
            pressure=1.0*unit.atmospheres, temperature=298.15*unit.kelvin
        ),
        phase=PropertyPhase.Liquid,
        value=1.0*unit.gram/unit.millilitre,
        uncertainty=0.0001*unit.gram/unit.millilitre
    )

    # Add the property to a data set
    data_set = PhysicalPropertyDataset()
    data_set.add_properties(density)

are readily JSON (de)serializable::

    # Save the data set as a JSON file.
    data_set.json(file_path="data_set.json", format=True)
    # Load the data set from a JSON file
    data_set = PhysicalPropertyDataset.from_json(file_path="data_set.json")

and may be converted to pandas |data_frame| objects::

    data_set.to_pandas()

The framework implements specific data set objects for extracting data measurements directly from a number of open data
sources, such as the |thermoml_data_set| (see :doc:`thermomldatasets`) which provides utilities for extracting the data
from the `NIST ThermoML Archive <http://trc.nist.gov/ThermoML.html>`_ and converting it into the standard framework
objects.

Data set objects are directly iterable::

    for physical_property in data_set:
        ...

or can be iterated over for a specific substance::

    for physical_property in data_set.properties_by_substance(substance):
        ...

or for a specific type of property::

    for physical_property in data_set.properties_by_type("Density"):
        ...

Physical Properties
-------------------

The |physical_property| object is a base class for any object which describes a measured property of substance, and is
defined by a combination of:

* the observed value of the property.
* |substance| specifying the substance that the measurement was collected for.
* |property_phase| specifying the phase that the measurement was collected in.
* |thermodynamic_state| specifying the thermodynamic conditions under which the measurement was performed

as well as optionally

* the uncertainty in the value of the property.
* a list of |parameter_gradient| which defines the gradient of the property with respect to the model parameters
  if it was computationally estimated.
* a |source| specifying the source (either experimental or computational) and provenance of the measurement.

Each type of property supported by the framework, such as a density of an enthalpy of vaporization, must have it's own
class representation which inherits from |physical_property|::

    # Define a density measurement
    density = Density(
        substance=Substance.from_components("O"),
        thermodynamic_state=ThermodynamicState(
            pressure=1.0*unit.atmospheres, temperature=298.15*unit.kelvin
        ),
        phase=PropertyPhase.Liquid,
        value=1.0*unit.gram/unit.millilitre,
        uncertainty=0.0001*unit.gram/unit.millilitre
    )

Substances
----------

A |substance| is defined by a number of components (which may have specific roles assigned to them such as
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

Property Phases
---------------

The |property_phase| enum describes the possible phases which a measurement was performed in.

While the enum only has three defined phases (``Solid``, ``Liquid`` and ``Gas``), multiple phases can be formed by
OR'ing (|) multiple phases together. As an example, to define a phase for a liquid and gas coexisting::

    liquid_gas_phase = PropertyPhase.Liquid | PropertyPhase.Gas

Thermodynamic States
--------------------

A |thermodynamic_state| specifies a combination of the temperature and (optionally) the pressure at which a
measurement is performed::

    thermodynamic_state = ThermodynamicState(
        temperature=298.15*unit.kelvin, pressure=1.0*unit.atmosphere
    )

