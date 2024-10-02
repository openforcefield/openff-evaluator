import json
import pathlib
import shutil

import numpy as np
import pytest
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit
from openff.utilities.utilities import get_data_dir_path
from openmm.openmm import System as OpenMMSystem

from openff.evaluator.datasets import PropertyPhase
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.properties import EnthalpyOfMixing
from openff.evaluator.protocols import analysis
from openff.evaluator.substances import Component, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.observables import Observable
from openff.evaluator.utils.serialization import TypedJSONEncoder
from openff.evaluator.workflow import Workflow


def _write_dummy_trajectory_file(directory):
    """Write a dummy trajectory file to disk.
    This gets around a check in simulation resumption
    that requires a trajectory file to exist and not be empty.
    """

    directory = pathlib.Path(directory)
    trajectory = directory / "trajectory.dcd"
    with trajectory.open("w") as file:
        file.write("dummy")


def _get_dummy_enthalpy_of_mixing(substance):
    """Generate a dummy EnthalpyOfMixing object for testing."""
    thermodynamic_state = ThermodynamicState(
        temperature=298 * unit.kelvin, pressure=1 * unit.atmosphere
    )
    return EnthalpyOfMixing(
        thermodynamic_state=thermodynamic_state,
        phase=PropertyPhase.Liquid,
        substance=substance,
        value=10.0 * unit.kilojoules_per_mole,
        uncertainty=1.0 * unit.kilojoules_per_mole,
    )


def _write_force_field(force_field: str = "openff-2.0.0.offxml"):
    """
    Write a force field file to disk.
    """
    ff = ForceField(force_field)
    with open("force-field.json", "w") as file:
        file.write(SmirnoffForceFieldSource.from_object(ff).json())


def _generate_dummy_observable(name):
    """Generate fake observable data for calculating with mole fractions"""
    observable = analysis.AverageObservable(name)
    observable.value = Observable(
        value=unit.Measurement(
            value=10 * unit.kilojoules_per_mole, error=1 * unit.kilojoules_per_mole
        )
    )
    return observable


class TestEnthalpyOfMixing:

    @pytest.mark.parametrize(
        "input_mole_fractions, output_mole_fractions",
        [
            [(0.5, 0.5), (0.5, 0.5)],
            [(0.1037, 0.8963), (0.10, 0.90)],
        ],
    )
    def test_mole_fractions_direct_simulation(
        self, input_mole_fractions, output_mole_fractions, tmpdir
    ):
        """
        This test *only* checks the part where mole fractions are weighted.

        It does the following:
        
        * creates an EnthalpyOfMixing target
        * executes the "build" where molecules are packed into a box with Packmol
        * creates some dummy observable data
        * calculates the mole fraction weighting of said data
        """
        # build our enthalpy of mixing target
        default_schema = EnthalpyOfMixing.default_simulation_schema(n_molecules=100)
        workflow_schema = default_schema.workflow_schema
        workflow_schema.replace_protocol_types(
            {"BaseBuildSystem": "BuildSmirnoffSystem"}
        )
        possible_smiles = ["O", "OCCN(CCO)CCO", "CO", "CCO"]

        substance = Substance()
        n_components = len(input_mole_fractions)
        for i in range(n_components):
            substance.add_component(
                Component(smiles=possible_smiles[i]),
                MoleFraction(input_mole_fractions[i]),
            )

        physical_property = _get_dummy_enthalpy_of_mixing(substance)
        with tmpdir.as_cwd():
            here = pathlib.Path(".")

            # generate force field and metadata
            _write_force_field()
            metadata = Workflow.generate_default_metadata(
                physical_property, "force-field.json"
            )
            uuid = "4000"

            # generate workflow
            workflow = Workflow.from_schema(
                workflow_schema, metadata=metadata, unique_id=uuid
            )
            workflow_graph = workflow.to_graph()
            protocol_graph = workflow_graph._protocol_graph
            
            # execute the build protocol, saving the output file paths
            parent_outputs = []
            for name, protocol in workflow_graph.protocols.items():
                if "build" in name:
                    build_path = name.replace("|", "_")
                    output = protocol_graph._execute_protocol(
                        build_path,
                        protocol,
                        True,
                        available_resources=None,
                        safe_exceptions=True,
                    )
                    # keep the output JSON files to pass in as input to mole fraction protocols
                    parent_outputs.append(output)

            for i in range(n_components):
                # generate fake data for the conditional groups
                base = f"extract_observable_component_{i}"
                path = (
                    here
                    / f"{uuid}_conditional_group_component_{i}"
                    / f"{uuid}_{base}"
                    / f"{uuid}|{base}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                name = f"extract_observable_component_{i}"
                observable = _generate_dummy_observable(name)
                with path.open("w") as file:
                    file.write(
                        json.dumps(
                            {
                                ".value": observable.value,
                                ".time_series_statistics": observable.time_series_statistics,
                            },
                            cls=TypedJSONEncoder,
                        )
                    )
                parent_outputs.append((f"{uuid}|{base}", str(path)))

                # now check mole fraction protocol execution
                cg = workflow_graph.protocols[f"{uuid}|conditional_group_component_{i}"]
                wmf = cg.protocols[f"{uuid}|weight_by_mole_fraction_{i}"]
                wmf_path = (
                    here
                    / f"{uuid}_conditional_group_component_{i}"
                    / f"{uuid}_weight_by_mole_fraction_{i}"
                )
                protocol_graph._execute_protocol(
                    str(wmf_path),
                    wmf,
                    True,
                    *parent_outputs,
                    available_resources=None,
                    safe_exceptions=True,
                )

                # check mole fraction substance directly
                assert np.isclose(
                    wmf.full_substance.amounts[possible_smiles[i] + "{solv}"][0].value,
                    output_mole_fractions[i],
                )

                # check weighted value
                assert np.isclose(
                    wmf.weighted_value.value.m_as(unit.kilojoules_per_mole),
                    output_mole_fractions[i] * 10,
                )

    def test_expected_output_from_production_simulation(self, tmpdir):
        """
        This is an integration test of sorts,
        constructed to test expected mole fractions. See Issue #575.

        This test steps through the EnthalpyOfMixing process.
        Some data is provided to avoid re-simulating long trajectories:

        - energy minimisations and equilibrations
        - the production simulation observables
        - the decorrelated trajectory and observables

        Note: the test is expected to take a few minutes to run.

        TODO: add further tests on earlier steps (e.g. parameterisation)

        """
        # locate our saved test data
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
        for schema in workflow_schema.protocol_schemas:
            # set conditional protocols to match the test data
            if "conditional" in schema.id:
                for protocol_name, protocol in schema.protocol_schemas.items():
                    if "simulation" in protocol_name:
                        protocol.inputs[".steps_per_iteration"] = 10000000
                        protocol.inputs[".output_frequency"] = 2000
                        protocol.inputs[".checkpoint_frequency"] = 10

        # build the enthalpy of mixing target
        substance = Substance()
        substance.add_component(Component(smiles="O"), MoleFraction(0.5098))
        substance.add_component(Component(smiles="OCCN(CCO)CCO"), MoleFraction(0.4902))

        physical_property = _get_dummy_enthalpy_of_mixing(substance)

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

            # write dummy trajectory files
            for suffix in ["component_0", "component_1", "mixture"]:
                _write_dummy_trajectory_file(
                    tmp_path
                    / f"{uuid}_conditional_group_{suffix}"
                    / f"{uuid}_production_simulation_{suffix}"
                )

            result = workflow.execute()

            # manually check mole fractions from built coordinates are as expected
            substance0 = workflow.protocols[
                f"{uuid}|build_coordinates_component_0"
            ].output_substance
            substance1 = workflow.protocols[
                f"{uuid}|build_coordinates_component_1"
            ].output_substance
            substance_mix = workflow.protocols[
                f"{uuid}|build_coordinates_mixture"
            ].output_substance

            assert len(substance0.amounts) == 1
            assert len(substance1.amounts) == 1
            assert len(substance_mix.amounts) == 2  # binary mixture

            assert np.isclose(substance0.amounts[r"O{solv}"][0].value, 1)
            assert np.isclose(substance1.amounts[r"OCCN(CCO)CCO{solv}"][0].value, 1)
            assert np.isclose(substance_mix.amounts[r"O{solv}"][0].value, 0.51)
            assert np.isclose(
                substance_mix.amounts[r"OCCN(CCO)CCO{solv}"][0].value, 0.49
            )

            # check assignment and parameterization
            system0 = workflow.protocols[
                f"{uuid}|assign_parameters_component_0"
            ].parameterized_system
            system1 = workflow.protocols[
                f"{uuid}|assign_parameters_component_1"
            ].parameterized_system
            system_mix = workflow.protocols[
                f"{uuid}|assign_parameters_mixture"
            ].parameterized_system

            for system in [system0, system1, system_mix]:
                assert isinstance(system.topology, Topology)
                assert system.topology.n_molecules == 1000
                assert isinstance(system.system, OpenMMSystem)

            # check cg:= conditional_group. This is a group of protocols
            cg0 = workflow.protocols[f"{uuid}|conditional_group_component_0"]
            cg1 = workflow.protocols[f"{uuid}|conditional_group_component_1"]
            cg_mix = workflow.protocols[f"{uuid}|conditional_group_mixture"]

            # check enthalpy is correct
            # note this is just read in
            enth0 = cg0.protocols[f"{uuid}|extract_observable_component_0"].value
            enth1 = cg1.protocols[f"{uuid}|extract_observable_component_1"].value
            enth_mix = cg_mix.protocols[f"{uuid}|extract_observable_mixture"].value

            assert np.isclose(
                enth0.value.m_as(unit.kilojoules_per_mole), -44.879, atol=1e-3
            )
            assert np.isclose(
                enth1.value.m_as(unit.kilojoules_per_mole), 251.418, atol=1e-3
            )
            assert np.isclose(
                enth_mix.value.m_as(unit.kilojoules_per_mole), 98.855, atol=1e-3
            )

            # now check weighting by mole fraction
            wmf0 = cg0.protocols[f"{uuid}|weight_by_mole_fraction_0"]
            wmf1 = cg1.protocols[f"{uuid}|weight_by_mole_fraction_1"]

            assert np.isclose(wmf0.full_substance.amounts[r"O{solv}"][0].value, 0.51)
            assert np.isclose(
                wmf1.full_substance.amounts[r"OCCN(CCO)CCO{solv}"][0].value, 0.49
            )

            assert np.isclose(
                wmf0.weighted_value.value.m_as(unit.kilojoules_per_mole),
                -22.888,
                atol=1e-3,
            )
            assert np.isclose(
                wmf1.weighted_value.value.m_as(unit.kilojoules_per_mole),
                123.195,
                atol=1e-3,
            )

            # check adding the two together
            add = workflow.protocols[f"{uuid}|add_component_observables"].result
            assert np.isclose(
                add.value.m_as(unit.kilojoules_per_mole), 100.307, atol=1e-3
            )

            # check final value
            assert np.isclose(
                result.value.value.m_as(unit.kilojoules_per_mole), -1.452, atol=1e-3
            )
