import pathlib
import shutil

import numpy as np
from openff.toolkit.topology import Topology
from openff.units import unit
from openff.utilities.utilities import get_data_dir_path
from openmm.openmm import System as OpenMMSystem

from openff.evaluator.datasets import PropertyPhase
from openff.evaluator.properties import EnthalpyOfMixing
from openff.evaluator.properties.enthalpy import EnthalpyOfMixing
from openff.evaluator.substances import Component, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.workflow import Workflow


class TestEnthalpyOfMixing:

    def test_expected_output_from_production_simulation(self, tmpdir):
        """
        First constructed to test expected mole fractions.

        See Issue #575
        """

        data_directory = pathlib.Path(
            get_data_dir_path(
                "test/example_properties/dhmix_triethanolamine", "openff.evaluator"
            )
        )

        default_schema = EnthalpyOfMixing.default_simulation_schema(n_molecules=1000)
        workflow_schema = default_schema.workflow_schema
        workflow_schema.replace_protocol_types(
            {"BaseBuildSystem": "BuildSmirnoffSystem"}
        )

        substance = Substance()
        substance.add_component(Component(smiles="O"), MoleFraction(0.5098))
        substance.add_component(Component(smiles="OCCN(CCO)CCO"), MoleFraction(0.4902))

        thermodynamic_state = ThermodynamicState(
            temperature=298 * unit.kelvin, pressure=1 * unit.atmosphere
        )
        physical_property = EnthalpyOfMixing(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            substance=substance,
            value=10.0 * unit.kilojoules_per_mole,
            uncertainty=1.0 * unit.kilojoules_per_mole,
        )

        metadata = Workflow.generate_default_metadata(
            physical_property, str(data_directory / "force-field.json")
        )
        uuid = "6421"
        workflow = Workflow.from_schema(
            workflow_schema, metadata=metadata, unique_id=uuid
        )

        abs_path = data_directory.resolve()
        with tmpdir.as_cwd():
            tmp_path = pathlib.Path(".")
            # copy data files over from data_directory
            for path in abs_path.iterdir():
                if path.name.startswith(uuid):
                    dest_dir = tmp_path / path.name
                    shutil.copytree(path, dest_dir)

            result = workflow.execute()

            # manually check mole fractions
            substance0 = workflow.protocols[
                "6421|build_coordinates_component_0"
            ].output_substance
            substance1 = workflow.protocols[
                "6421|build_coordinates_component_1"
            ].output_substance
            substance_mix = workflow.protocols[
                "6421|build_coordinates_mixture"
            ].output_substance

            assert len(substance0.amounts) == 1
            assert len(substance1.amounts) == 1
            assert len(substance_mix.amounts) == 2

            assert np.isclose(substance[0][r"O{solv}"].value, 1)
            assert np.isclose(substance[1][r"OCCN(CCO)CCO{solv}"].value, 1)
            assert np.isclose(substance_mix[0][r"O{solv}"].value, 0.51)
            assert np.isclose(substance_mix[1][r"OCCN(CCO)CCO{solv}"].value, 0.49)

            # check assignment and parameterization
            system0 = workflow.protocols[
                "6421|assign_parameters_component_0"
            ].parameterized_system
            system1 = workflow.protocols[
                "6421|assign_parameters_component_1"
            ].parameterized_system
            system_mix = workflow.protocols[
                "6421|assign_parameters_mixture"
            ].parameterized_system

            for system in [system0, system1, system_mix]:
                assert isinstance(system.topology, Topology)
                assert system.topology.n_molecules == 1000
                assert isinstance(system.system, OpenMMSystem)

            # check cg:= conditional_group. This is a group of protocols
            cg0 = workflow.protocols["6421|conditional_group_component_0"]
            cg1 = workflow.protocols["6421|conditional_group_component_1"]
            cg_mix = workflow.protocols["6421|conditional_group_mixture"]

            # check enthalpy is correct
            # note this is not re-simulated, but it *is* re-calculated
            enth0 = cg0.protocols["6421|extract_observable_component_0"].value
            enth1 = cg1.protocols["6421|extract_observable_component_1"].value
            enth_mix = cg_mix.protocols["6421|extract_observable_mixture"].value

            assert np.isclose(enth0.m_as(unit.kilojoules_per_mole), -44.879, atol=1e-3)
            assert np.isclose(enth1.m_as(unit.kilojoules_per_mole), 251.418, atol=1e-3)
            assert np.isclose(
                enth_mix.m_as(unit.kilojoules_per_mole), 98.855, atol=1e-3
            )

            # now check weighting by mole fraction
            wmf0 = cg0.protocols["6421|weight_by_mole_fraction_0"].weighted_value
            wmf1 = cg1.protocols["6421|weight_by_mole_fraction_1"].weighted_value

            assert np.isclose(
                wmf0.value.m_as(unit.kilojoules_per_mole), -22.888, atol=1e-3
            )
            assert np.isclose(
                wmf1.value.m_as(unit.kilojoules_per_mole), 123.195, atol=1e-3
            )

            # check final value
            assert np.isclose(
                result.value.value.m_as(unit.kilojoules_per_mole), -1.452, atol=1e-3
            )
