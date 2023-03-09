import shutil

from openff.toolkit.typing.engines.smirnoff import ForceField, ParameterList
from openff.units import unit
from openmm import unit as openmm_unit

from openff.evaluator.forcefield import ParameterGradientKey, SmirnoffForceFieldSource
from openff.evaluator.protocols.coordinates import BuildCoordinatesPackmol
from openff.evaluator.protocols.forcefield import BuildSmirnoffSystem
from openff.evaluator.protocols.openmm import OpenMMEnergyMinimisation, OpenMMSimulation
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import Ensemble, ThermodynamicState


def tip4p_force_field() -> ForceField:
    force_field = ForceField()

    constraint_handler = force_field.get_parameter_handler("Constraints")
    constraint_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1]",
            "distance": 0.9572 * openmm_unit.angstrom,
        }
    )
    constraint_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1:2]",
            "distance": 1.5139 * openmm_unit.angstrom,
        }
    )

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0]-[#1]",
            "epsilon": (
                78.0
                * openmm_unit.kelvin
                * openmm_unit.BOLTZMANN_CONSTANT_kB
                * openmm_unit.AVOGADRO_CONSTANT_NA
            ),
            "sigma": 3.154 * openmm_unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#1]-[#8X2H2+0:1]-[#1]",
            "epsilon": 0.0 * openmm_unit.kilojoules_per_mole,
            "sigma": 1.0 * openmm_unit.angstrom,
        }
    )

    force_field.get_parameter_handler("Electrostatics")
    force_field.get_parameter_handler(
        "ChargeIncrementModel",
        {"version": "0.3", "partial_charge_method": "formal_charge"},
    )

    virtual_site_handler = force_field.get_parameter_handler("VirtualSites")
    virtual_site_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "type": "DivalentLonePair",
            "distance": -0.15 * openmm_unit.angstrom,
            "outOfPlaneAngle": 0.0 * openmm_unit.degrees,
            "match": "once",
            "charge_increment1": 0.52 * openmm_unit.elementary_charge,
            "charge_increment2": 0.0 * openmm_unit.elementary_charge,
            "charge_increment3": 0.52 * openmm_unit.elementary_charge,
        }
    )
    # Currently required due to OpenFF issue #884
    virtual_site_handler._parameters = ParameterList(virtual_site_handler._parameters)

    return force_field


def main():
    force_field = tip4p_force_field()
    substance = Substance.from_components("O")

    with open("force-field.json", "w") as file:
        file.write(SmirnoffForceFieldSource.from_object(force_field).json())

    build_coordinates = BuildCoordinatesPackmol("")
    build_coordinates.substance = substance
    build_coordinates.max_molecules = 216
    build_coordinates.execute("build-coords")

    apply_parameters = BuildSmirnoffSystem("")
    apply_parameters.force_field_path = "force-field.json"
    apply_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
    apply_parameters.substance = substance
    apply_parameters.execute("apply-params")

    minimize = OpenMMEnergyMinimisation("")
    minimize.input_coordinate_file = build_coordinates.coordinate_file_path
    minimize.parameterized_system = apply_parameters.parameterized_system
    minimize.execute("minimize-coords")

    npt = OpenMMSimulation("")
    npt.input_coordinate_file = minimize.output_coordinate_file
    npt.parameterized_system = apply_parameters.parameterized_system
    npt.ensemble = Ensemble.NPT
    npt.thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )
    npt.steps_per_iteration = 500
    npt.total_number_of_iterations = 2
    npt.gradient_parameters = [
        ParameterGradientKey(
            tag="vdW", smirks="[#1:1]-[#8X2H2+0]-[#1]", attribute="epsilon"
        )
    ]
    npt.output_frequency = 50
    npt.execute("run-npt")

    shutil.copytree("run-npt", "run-npt-1")

    npt.total_number_of_iterations = 4
    npt.execute("run-npt")

    assert len(npt.observables) == 40
    assert len(npt.observables["PotentialEnergy"].gradients) == 1
    assert len(npt.observables["PotentialEnergy"].gradients[0]) == 40


if __name__ == "__main__":
    main()
