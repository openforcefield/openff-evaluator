import os

from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit

from openff import evaluator
from openff.evaluator.backends.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import EvaluatorClient, RequestOptions
from openff.evaluator.datasets import PhysicalPropertyDataSet, PropertyPhase
from openff.evaluator.datasets.curation.components.filtering import (
    FilterBySubstancesSchema,
)
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)
from openff.evaluator.forcefield import ParameterGradientKey
from openff.evaluator.properties import (
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)
from openff.evaluator.server import EvaluatorServer
from openff.evaluator.storage import LocalFileStorage
from openff.evaluator.substances import Component, ExactAmount, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import setup_timestamp_logging
from openff.evaluator.utils.utils import temporarily_change_directory

os.setenv("CUDA_VISIBLE_DEVICES", "0")


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
            *CurationWorkflow.apply(
                PhysicalPropertyDataSet(),
                CurationWorkflowSchema(
                    component_schemas=[
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


def main():
    setup_timestamp_logging()

    # Retrieve the current version.
    version = evaluator.__version__.replace(".", "-").replace("v", "")

    if "+" in version:
        version = "latest"

    # Create a new directory to run the current versions results in.
    os.makedirs(os.path.join(version, "results"))

    with temporarily_change_directory(version):
        with DaskLocalCluster(
            number_of_workers=1,
            resources_per_worker=ComputeResources(
                number_of_threads=1,
                number_of_gpus=1,
                preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
            ),
        ) as local_backend:

            with EvaluatorServer(
                local_backend,
                working_directory="outputs",
                storage_backend=LocalFileStorage("cached-data"),
            ):
                client = EvaluatorClient()

                for allowed_layer in ["ReweightingLayer"]:
                    data_set = define_data_set(allowed_layer == "ReweightingLayer")

                    options = RequestOptions()
                    options.calculation_layers = [allowed_layer]
                    options.calculation_schemas = {
                        property_type: {} for property_type in data_set.property_types
                    }

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
