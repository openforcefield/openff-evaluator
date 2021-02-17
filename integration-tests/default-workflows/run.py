import os

from openff.toolkit.typing.engines.smirnoff import ForceField

from openff import evaluator
from openff.evaluator import unit
from openff.evaluator.backends import QueueWorkerResources
from openff.evaluator.backends.dask import DaskLSFBackend
from openff.evaluator.client import EvaluatorClient, RequestOptions
from openff.evaluator.datasets import PhysicalPropertyDataSet, PropertyPhase
from openff.evaluator.datasets.curation.components.filtering import (
    FilterBySubstancesSchema,
)
from openff.evaluator.datasets.curation.components.freesolv import ImportFreeSolvSchema
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)
from openff.evaluator.forcefield import ParameterGradientKey
from openff.evaluator.layers.simulation import SimulationSchema
from openff.evaluator.properties import (
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
    SolvationFreeEnergy,
)
from openff.evaluator.server import EvaluatorServer
from openff.evaluator.storage import LocalFileStorage
from openff.evaluator.substances import Component, ExactAmount, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import setup_timestamp_logging
from openff.evaluator.utils.utils import temporarily_change_directory


def define_data_set(reweighting: bool) -> PhysicalPropertyDataSet:

    # Define a common state to compute estimates at
    states = [
        ThermodynamicState(
            temperature=296.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
        ),
        ThermodynamicState(
            temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
        ),
        ThermodynamicState(
            temperature=300.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
        ),
    ]

    data_set = PhysicalPropertyDataSet()

    # Solvation free energies.
    if not reweighting:

        ethanol_substance = Substance.from_components("CCO")
        ethanol_substance.add_component(
            Component("CC=O", Component.Role.Solute), ExactAmount(1)
        )
        ethanal_substance = Substance.from_components("CC=O")
        ethanal_substance.add_component(
            Component("CCO", Component.Role.Solute), ExactAmount(1)
        )

        data_set.add_properties(
            SolvationFreeEnergy(
                thermodynamic_state=states[1],
                phase=PropertyPhase.Liquid,
                substance=ethanol_substance,
                value=0.0 * SolvationFreeEnergy.default_unit(),
            ),
            SolvationFreeEnergy(
                thermodynamic_state=states[1],
                phase=PropertyPhase.Liquid,
                substance=ethanal_substance,
                value=0.0 * SolvationFreeEnergy.default_unit(),
            ),
            *CurationWorkflow.apply(
                PhysicalPropertyDataSet(),
                CurationWorkflowSchema(
                    component_schemas=[
                        ImportFreeSolvSchema(),
                        FilterBySubstancesSchema(substances_to_include=[("O", "CO")]),
                    ]
                ),
            ),
        )

    for state in states:

        # Excess properties.
        data_set.add_properties(
            ExcessMolarVolume(
                thermodynamic_state=state,
                phase=PropertyPhase.Liquid,
                substance=Substance.from_components("CC=O", "CCO"),
                value=0.0 * ExcessMolarVolume.default_unit(),
            ),
            EnthalpyOfMixing(
                thermodynamic_state=state,
                phase=PropertyPhase.Liquid,
                substance=Substance.from_components("CC=O", "CCO"),
                value=0.0 * EnthalpyOfMixing.default_unit(),
            ),
        )
        # Pure properties
        data_set.add_properties(
            Density(
                thermodynamic_state=state,
                phase=PropertyPhase.Liquid,
                substance=Substance.from_components("CCO"),
                value=0.0 * Density.default_unit(),
            ),
            EnthalpyOfVaporization(
                thermodynamic_state=state,
                phase=PropertyPhase(PropertyPhase.Liquid | PropertyPhase.Gas),
                substance=Substance.from_components("CCO"),
                value=0.0 * EnthalpyOfVaporization.default_unit(),
            ),
            DielectricConstant(
                thermodynamic_state=state,
                phase=PropertyPhase.Liquid,
                substance=Substance.from_components("CCO"),
                value=0.0 * DielectricConstant.default_unit(),
            ),
        )

    return data_set


def solvation_free_energy_schema() -> SimulationSchema:
    """Override trailblazing to use the lambda values of used in the previous OFF study
    https://github.com/MobleyLab/SMIRNOFF_paper_code/tree/master/FreeSolv
    """

    default_schema = SolvationFreeEnergy.default_simulation_schema()
    workflow_schema = default_schema.workflow_schema

    conditional_group_schema = next(
        x for x in workflow_schema.protocol_schemas if x.id == "conditional_group"
    )
    conditional_group = conditional_group_schema.to_protocol()

    yank_protocol = conditional_group.protocols["run_solvation_yank"]

    yank_protocol.electrostatic_lambdas_1 = [
        1.00,
        0.75,
        0.50,
        0.25,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
    yank_protocol.steric_lambdas_1 = [
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        0.95,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
        0.05,
        0.00,
    ]

    yank_protocol.electrostatic_lambdas_2 = [1.00, 0.75, 0.50, 0.25, 0.00]
    yank_protocol.steric_lambdas_2 = [1.00, 1.00, 1.00, 1.00, 1.00]

    workflow_schema.protocol_schemas.remove(conditional_group_schema)
    workflow_schema.protocol_schemas.append(conditional_group.schema)

    return default_schema


def main():

    setup_timestamp_logging()

    # Retrieve the current version.
    version = evaluator.__version__.replace(".", "-").replace("v", "")

    if "+" in version:
        version = "latest"

    # Create a new directory to run the current versions results in.
    os.makedirs(os.path.join(version, "results"))

    with temporarily_change_directory(version):

        with DaskLSFBackend(
            minimum_number_of_workers=1,
            maximum_number_of_workers=12,
            resources_per_worker=QueueWorkerResources(
                number_of_gpus=1,
                preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                per_thread_memory_limit=5 * unit.gigabyte,
                wallclock_time_limit="05:59",
            ),
            setup_script_commands=[
                f"conda activate openff-evaluator-{version}",
                "module load cuda/10.0",
            ],
            queue_name="gpuqueue",
        ) as calculation_backend:

            with EvaluatorServer(
                calculation_backend,
                working_directory="outputs",
                storage_backend=LocalFileStorage("cached-data"),
            ):

                client = EvaluatorClient()

                for allowed_layer in ["SimulationLayer", "ReweightingLayer"]:

                    data_set = define_data_set(allowed_layer == "ReweightingLayer")

                    options = RequestOptions()
                    options.calculation_layers = [allowed_layer]
                    options.calculation_schemas = {
                        property_type: {} for property_type in data_set.property_types
                    }

                    if allowed_layer == "SimulationLayer":

                        options.add_schema(
                            "SimulationLayer",
                            "SolvationFreeEnergy",
                            solvation_free_energy_schema(),
                        )

                    request, _ = client.request_estimate(
                        data_set,
                        ForceField("openff-1.2.0.offxml"),
                        options,
                        parameter_gradient_keys=[
                            ParameterGradientKey("vdW", smirks, attribute)
                            for smirks in [
                                "[#1:1]-[#6X4]",
                                "[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]",
                                "[#1:1]-[#8]",
                                "[#6X4:1]",
                                "[#8X2H1+0:1]",
                                "[#1]-[#8X2H2+0:1]-[#1]",
                            ]
                            for attribute in ["epsilon", "rmin_half"]
                        ],
                    )

                    results, _ = request.results(synchronous=True, polling_interval=60)
                    results.json(os.path.join("results", f"{allowed_layer}.json"))


if __name__ == "__main__":
    main()
