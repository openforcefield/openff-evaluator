Physical property measurements
==================================

.. warning:: This text is now out of date, but will be updated in future to reflect the
             latest version of the framework.

`Physical property measurements <https://en.wikipedia.org/wiki/Physical_property>`_ are measured properties of
a substance that provide some information about the physical parameters that define the interactions within the
substance.

A physical property is defined by a combination of:

* A ``Mixture`` specifying the substance that the measurement was performed on
* A ``ThermodynamicState`` specifying the thermodynamic conditions under which the measurement was performed
* A ``PhysicalProperty`` is the physical property that was measured
* A ``MeasurementMethod`` specifying the kind of measurement that was performed

An example of each:

* ``Mixture``: a 0.8 mole fraction mixture of ethanol and water
* ``ThermodynamicState``: 298 kelvin, 1 atmosphere
* ``PhysicalProperty``: mass density
* ``MeasurementMethod``: vibrating tube method

Physical substances
-------------------

We generally use the concept of a liquid or gas ``Mixture``, which is a subclass of ``Substance``.

A simple liquid has only one component:

.. code-block:: python

    liquid = Mixture()
    liquid.add_component('water')

A binary mixture has two components:

.. code-block:: python

    binary_mixture = Mixture()
    binary_mixture.add_component('water', mole_fraction=0.2)
    binary_mixture.add_component('methanol') # assumed to be rest of mixture if no mole_fraction specified

A ternary mixture has three components:

.. code-block:: python

    ternary_mixture = Mixture()
    ternary_mixture.add_component('ethanol', mole_fraction=0.2)
    ternary_mixture.add_component('methanol', mole_fraction=0.2)
    ternary_mixture.add_component('water')

The infinite dilution of one solute within a solvent or mixture is also specified as a ``Mixture``, where the solute
has zero mole fraction:

.. code-block:: python

    infinite_dilution = Mixture()
    infinite_dilution.add_component('phenol', impurity=True) # infinite dilution; one copy only of the impurity
    infinite_dilution.add_component('water')

You can iterate over the components in a mixture:

.. code-block:: python

    for component in mixture.components:
        print (component.iupac_name, component.mole_fraction)

retrieve a component by name:

.. code-block:: python

    component = mixture.components['ethanol']

or get the number of components in a mixture:

.. code-block:: python

    ncomponents = mixture.ncomponents

or check if a component is an impurity:

.. code-block:: python

    if component.impurity == True:
        ...

Thermodynamic states
--------------------

A ``ThermodynamicState`` specifies a combination of thermodynamic parameters (e.g. temperature, pressure) at which a
measurement is performed.

.. code-block:: python

    from simtk import unit
    thermodynamic_state = ThermodynamicState(pressure=500*unit.kilopascals, temperature=298.15*unit.kelvin)

We use the ``simtk.unit`` unit system from `OpenMM <http://openmm.org>`_ for units (though we may later migrate to
`pint <https://pint.readthedocs.io>`_ for portability).

Physical property measurements
------------------------------

A ``MeasuredPhysicalProperty`` is a combination of ``Substance``, ``ThermodynamicState``, and a unit-bearing measured
property ``value`` and ``uncertainty``:

.. code-block:: python

    # Define mixture
    mixture = Mixture()
    mixture.addComponent('water', mole_fraction=0.2)
    mixture.addComponent('methanol')

    # Define thermodynamic state
    thermodynamic_state = ThermodynamicState(pressure=500*unit.kilopascals, temperature=298.15*unit.kelvin)

    # Define measurement
    measurement = ExcessMolarEnthalpy(substance, thermodynamic_state, value=83.3863244*unit.kilojoules_per_mole,
                                      uncertainty=0.1220794866*unit.kilojoules_per_mole)

The various properties are all subclasses of ``MeasuredPhysicalProperty`` and generally follow the ``<ePropName/>``
ThermoML tag names.

Some examples of ``MeasuredPhysicalProperty``:

* ``MassDensity`` - mass density
* ``ExcessMolarEnthalpy`` - excess partial apparent molar enthalpy
* ``HeatCapacity`` - molar heat capacity at constant pressure

A `roadmap of physical properties to be implemented <https://github.com/open-forcefield-group/open-forcefield-tools/wiki/Physical-Properties-for-Calculation>`_) is available.

Please raise an issue if your physical property of interest is not listed!

Each ``MeasuredPhysicalProperty`` has several properties:

* ``.substance`` - the ``Mixture`` for which the measurement was made
* ``.thermodynamic_state`` - the ``ThermodynamicState`` at which the measurement was made
* ``.measurement_method`` - the ``MeasurementMethod`` used to measure the physical property
* ``.value`` - the unit-bearing measurement value
* ``.uncertainty`` - the standard uncertainty of the measurement
* ``.reference`` - the literature reference (if available) for the measurement
* ``.DOI`` - the literature reference DOI (if available) for the measurement

The value, uncertainty, reference, and DOI do not necessarily need to be defined for a dataset in order for property
calculations to be performed.