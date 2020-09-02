.. |curation_component|                  replace:: :py:class:`~openff.evaluator.datasets.curation.components.CurationComponent`
.. |curation_component_schema|           replace:: :py:class:`~openff.evaluator.datasets.curation.components.CurationComponentSchema`

.. |curation_workflow|                   replace:: :py:class:`~openff.evaluator.datasets.curation.workflow.CurationWorkflow`
.. |curation_workflow_schema|            replace:: :py:class:`~openff.evaluator.datasets.curation.workflow.CurationWorkflowSchema`

.. |import_thermoml_data|                replace:: :py:class:`~openff.evaluator.datasets.curation.components.thermoml.ImportThermoMLData`
.. |import_free_solv|                    replace:: :py:class:`~openff.evaluator.datasets.curation.components.freesolv.ImportFreeSolv`

.. |convert_excess_density_data|         replace:: :py:class:`~openff.evaluator.datasets.curation.components.conversion.ConvertExcessDensityData`

.. |select_data_points|                  replace:: :py:class:`~openff.evaluator.datasets.curation.components.selection.SelectDataPoints`
.. |select_substances|                   replace:: :py:class:`~openff.evaluator.datasets.curation.components.selection.SelectSubstances`

.. |filter_duplicates|                   replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterDuplicates`
.. |filter_by_temperature|               replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByTemperature`
.. |filter_by_pressure|                  replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByPressure`
.. |filter_by_mole_fraction|             replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByMoleFraction`
.. |filter_by_racemic|                   replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByRacemic`
.. |filter_by_elements|                  replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByElements`
.. |filter_by_property_types|            replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByPropertyTypes`
.. |filter_by_stereochemistry|           replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByStereochemistry`
.. |filter_by_charged|                   replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByCharged`
.. |filter_by_ionic_liquid|              replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByIonicLiquid`
.. |filter_by_smiles|                    replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterBySmiles`
.. |filter_by_smirks|                    replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterBySmirks`
.. |filter_by_n_components|               replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByNComponents`
.. |filter_by_substances|                replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterBySubstances`
.. |filter_by_environments|              replace:: :py:class:`~openff.evaluator.datasets.curation.components.filtering.FilterByEnvironments`

.. |data_frame|                          replace:: :py:class:`~pandas.DataFrame`

Data Set Curation
=================

The framework offers a full suite of features to facilitate the curation of data sets of physical properties,
including:

.. rst-class:: spaced-list

    - a significant amount of data filters, including to filter by state, substance composition and chemical
      functionalities.

and components to

.. rst-class:: spaced-list

    - easily download and import the full :ref:`NIST ThermoML <datasets/thermomldatasets:ThermoML Archive>` and
      `FreeSolv <https://github.com/MobleyLab/FreeSolv>`_ archives .
    - select data points which were measured close to a set of target states, and which were measured for a diverse
      range of substances which contain specific functionalities.
    - convert between different compatible property types (e.g. convert density <-> excess molar volume data).

These features are implemented as |curation_component| objects, which take as input an associated
|curation_component_schema| which controls how the curation components should be applied to a particular data
set (or a data set which is being stored as pandas |data_frame| object).

An example of a curation component would be one that filters out data points which were measured outside of a
particular temperature range::

    # Filter data points measured at less than 290.0 K or greater than 320.0 K
    filtered_frame = FilterByTemperature.apply(
        data_frame,
        FilterByTemperatureSchema(minimum_temperature=290.0, maximum_temperature=320.0),
    )

Curation components can be conveniently chained together using a |curation_workflow| and an associated
|curation_workflow_schema| so as to easily curated full training and testing data sets::

    curation_schema = WorkflowSchema(
        component_schemas=[
            # Import the ThermoML archive.
            thermoml.ImportThermoMLDataSchema()
            # Filter out any measurements made for systems with more than two components
            filtering.FilterByNComponentsSchema(n_components=[1, 2]),
            # Remove any duplicate data.
            filtering.FilterDuplicatesSchema(),
            # Filter out data points measured away from ambient
            # and biologically relevant temperatures.
            filtering.FilterByTemperatureSchema(
                minimum_temperature=298.0, maximum_temperature=320.0
            ),
            # Retain only density and enthalpy of mixing data points.
            filtering.FilterByPropertyTypesSchema(
                property_types=["Density", "EnthalpyOfMixing"],
            ),
            # Select data points measured for alcohols, esters or mixtures of both.
            selection.SelectSubstancesSchema(
                target_environments=[
                    ChemicalEnvironment.Alcohol,
                    ChemicalEnvironment.CarboxylicAcidEster,
                ],
                n_per_environment=10,
            ),
        ]
    )

    data_frame = Workflow.apply(pandas.DataFrame(), curation)

Examples
--------

Data Extraction
"""""""""""""""

* |import_free_solv|: A component which will download the latest, full FreeSolv data
  set from the GitHub repository::

    from openff.evaluator.datasets.curation.components.freesolv import (
        ImportFreeSolv,
        ImportFreeSolvSchema,
    )

    # Import the full FreeSolv data set.
    data_frame = ImportFreeSolv.apply(pandas.DataFrame(), ImportFreeSolvSchema())

* |import_thermoml_data|: A component which will download all :ref:`supported data <datasets/thermomldatasets:Registering Properties>`
  from the NIST ThermoML Archive::

    from openff.evaluator.datasets.curation.components.thermoml import (
        ImportThermoMLData,
        ImportThermoMLDataSchema,
    )

    # Import all data collected from the IJT journal.
    data_frame = ImportThermoMLData.apply(
        pandas.DataFrame(), ImportThermoMLDataSchema(journal_names=["IJT"])
    )

Filtration
""""""""""

* |filter_duplicates|: A component to remove duplicate data points (within a specified precision) from a data set::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterDuplicates,
        FilterDuplicatesSchema,
    )

    filtered_frame = FilterDuplicates.apply(data_frame, FilterDuplicatesSchema())

* |filter_by_temperature|: A component which will filter out data points which were measured outside of a specified
  temperature range::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByTemperature,
        FilterByTemperatureSchema,
    )

    filtered_frame = FilterByTemperature.apply(
        data_frame,
        FilterByTemperatureSchema(minimum_temperature=290.0, maximum_temperature=320.0),
    )

* |filter_by_pressure|: A component which will filter out data points which were measured outside of a specified
  pressure range::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByPressure,
        FilterByPressureSchema,
    )

    filtered_frame = FilterByPressure.apply(
        data_frame,
        FilterByPressureSchema(minimum_pressure=100.0, maximum_pressure=140.0),
    )

* |filter_by_mole_fraction|: A component which will filter out data points which were measured outside of a specified
  mole fraction range::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByMoleFraction,
        FilterByMoleFractionSchema,
    )

    filtered_frame = FilterByMoleFraction.apply(
        data_frame, FilterByMoleFractionSchema(mole_fraction_ranges={2: [[(0.1, 0.3)]]})
    )

* |filter_by_racemic|: A component which will filter out data points which were measured for racemic mixtures::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByRacemic,
        FilterByRacemicSchema,
    )

    filtered_frame = FilterByRacemic.apply(data_frame, FilterByRacemicSchema())

* |filter_by_elements|: A component which will filter out data points which were measured for systems which contain
  specific elements::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByElements,
        FilterByElementsSchema,
    )

    filtered_frame = FilterByElements.apply(
        data_frame,
        FilterByElementsSchema(allowed_elements=["C", "O", "H"]),
    )

* |filter_by_property_types|: A component which will apply a filter which only retains properties of specified types::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByPropertyTypes,
        FilterByPropertyTypesSchema,
    )

    # Retain only density measurements made for either pure or binary systems.
    filtered_frame = FilterByPropertyTypes.apply(
        data_frame,
        FilterByPropertyTypesSchema(
            property_types=["Density"],
            n_components={"Density": [1, 2]},
        ),
    )

* |filter_by_stereochemistry|: A component which filters out data points measured for systems whereby the
  stereochemistry of a number of components is undefined::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByStereochemistry,
        FilterByStereochemistrySchema,
    )

    filtered_frame = FilterByStereochemistry.apply(
        data_frame, FilterByStereochemistrySchema()
    )

* |filter_by_charged|: A component which filters out data points measured for substance where any of the constituent
  components have a net non-zero charge.::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByCharged,
        FilterByChargedSchema,
    )

    filtered_frame = FilterByCharged.apply(data_frame, FilterByChargedSchema())

* |filter_by_ionic_liquid|: A component which filters out data points measured for substances which contain or are
  classed as an ionic liquids::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByIonicLiquid,
        FilterByIonicLiquidSchema,
    )

    filtered_frame = FilterByIonicLiquid.apply(data_frame, FilterByIonicLiquidSchema())

* |filter_by_smiles|: A component which filters the data set so that it only contains either a specific set of smiles,
  or does not contain any of a set of specifically excluded smiles::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterBySmiles,
        FilterBySmilesSchema,
    )

    filtered_frame = FilterBySmiles.apply(
        data_frame, FilterBySmilesSchema(smiles_to_include=["CCCO"]),
    )

* |filter_by_smirks|: A component which filters a data set so that it only contains measurements made for molecules
  which contain (or don't) a set of chemical environments represented by SMIRKS patterns::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterBySmirks,
        FilterBySmirksSchema,
    )

    filtered_frame = FilterBySmirks.apply(
        data_frame, FilterBySmirksSchema(smirks_to_include=["[#6a]"]),
    )

* |filter_by_n_components|: A component which filters out data points measured for systems with specified number of
  components::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByNComponents,
        FilterByNComponentsSchema,
    )

    filtered_frame = FilterByNComponents.apply(
        data_frame, FilterByNComponentsSchema(n_components=[1, 2])
    )

* |filter_by_substances|: A component which filters the data set so that it only contains properties measured for
  particular substances::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterBySubstances,
        FilterBySubstancesSchema,
    )

    filtered_frame = FilterBySubstances.apply(
        data_frame, FilterBySubstancesSchema(substances_to_include=[("CO", "C")])
    )

* |filter_by_environments|: A component which filters a data set so that it only contains measurements made for
  substances which contain specific chemical environments::

    from openff.evaluator.datasets.curation.components.filtering import (
        FilterByEnvironments,
        FilterByEnvironmentsSchema,
    )

    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            environments=[
                ChemicalEnvironment.Aqueous,
                ChemicalEnvironment.Alcohol,
                ChemicalEnvironment.Amine,
            ]
        ),
    )

Data Selection
""""""""""""""

* |select_substances|: A component for selecting data points which were measured for specified number
  of maximally diverse systems containing a specified set of chemical functionalities::

    # Select (if possible) data points which were measured for 10 different (and
    # structurally diverse) alcohols.
    schema = SelectSubstancesSchema(
        target_environments=[ChemicalEnvironment.Alcohol],
        n_per_environment=10,
    )

    data_frame = ConvertExcessDensityData.apply(data_frame, schema)

* |select_data_points|: A component for selecting a set of data points which are close to a particular set of
  states::

    # Select (if possible) density data points which were measured for pure systems
    # at close to 298.15 K and 308.15K
    schema = SelectDataPointsSchema(
        target_states=[
            TargetState(
                property_types=[("Density", 1)],
                states=[
                    State(temperature=298.15, pressure=101.325, mole_fractions=(1.0,),
                    State(temperature=308.15, pressure=101.325, mole_fractions=(1.0,),
                ],
            )
        ]
    )

    data_frame = ConvertExcessDensityData.apply(data_frame, schema)


Data Conversion
"""""""""""""""

* |convert_excess_density_data|: A component for converting binary mass density data to excess molar volume
  data and vice versa where pure density data measured for the components is
  available::

    from openff.evaluator.datasets.curation.components.conversion import (
        ConvertExcessDensityData,
        ConvertExcessDensityDataSchema,
    )

    converted_data_frame = ConvertExcessDensityData.apply(
        data_frame, ConvertExcessDensityDataSchema()
    )

