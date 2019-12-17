import json

from distributed import Adaptive

from propertyestimator import client, unit
from propertyestimator.backends import QueueWorkerResources, DaskLSFBackend
from propertyestimator.client import RequestOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.properties import PropertyPhase, MeasurementSource
from propertyestimator.properties.solvation import SolvationFreeEnergy
from propertyestimator.protocols.groups import ConditionalGroup
from propertyestimator.server.server import EvaluatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator.workflow import WorkflowOptions


class CustomAdaptive(Adaptive):
    """A temporary work-around to attempt to fix
    https://github.com/dask/distributed/issues/3154
    """

    async def recommendations(self, target: int) -> dict:
        """
        Make scale up/down recommendations based on current state and target
        """
        await self.cluster
        return await super(CustomAdaptive, self).recommendations(target)


def _get_fixed_lambda_schema(workflow_options):
    """Manually override trailblazing to set the values found in the previous OFF study
     https://github.com/MobleyLab/SMIRNOFF_paper_code/tree/master/FreeSolv

    Parameters
    ----------
    workflow_options: WorkflowOptions
        The options to use when building the workflow schema.

    Returns
    -------
    WorkflowSchema
        A workflow schema with the alchemical lambdas explicitly set.
    """

    default_schema = SolvationFreeEnergy.get_default_simulation_workflow_schema(workflow_options)

    conditional_group = ConditionalGroup('conditional_group')
    conditional_group.schema = default_schema.protocols['conditional_group']

    yank_protocol = conditional_group.protocols['run_solvation_yank']

    yank_protocol.electrostatic_lambdas_1 = [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                                             0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    yank_protocol.steric_lambdas_1 = [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50,
                                      0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]

    yank_protocol.electrostatic_lambdas_2 = [1.00, 0.75, 0.50, 0.25, 0.00]
    yank_protocol.steric_lambdas_2 = [1.00, 1.00, 1.00, 1.00, 1.00]

    default_schema.protocols['conditional_group'] = conditional_group.schema

    return default_schema


def _create_data_set():
    """Create a small data set of three properties taken from the
    FreeSolv data set: https://github.com/mobleylab/FreeSolv.

    Returns
    -------
    PhysicalPropertyDataSet
        The data set of three select FreeSolv properties.
    """

    butan_1_ol = Substance()
    butan_1_ol.add_component(Substance.Component('CCCCO', role=Substance.ComponentRole.Solute),
                             Substance.ExactAmount(1))
    butan_1_ol.add_component(Substance.Component('O', role=Substance.ComponentRole.Solvent),
                             Substance.MoleFraction(1.0))

    butan_1_ol_property = SolvationFreeEnergy(thermodynamic_state=ThermodynamicState(298.15*unit.kelvin,
                                                                                     1.0*unit.atmosphere),
                                              phase=PropertyPhase.Liquid,
                                              substance=butan_1_ol,
                                              value=-4.72*unit.kilocalorie / unit.mole,
                                              uncertainty=0.6*unit.kilocalorie / unit.mole,
                                              source=MeasurementSource(doi=' 10.1021/ct050097l'))

    methyl_propanoate = Substance()
    methyl_propanoate.add_component(Substance.Component('CCC(=O)OC', role=Substance.ComponentRole.Solute),
                                    Substance.ExactAmount(1))
    methyl_propanoate.add_component(Substance.Component('O', role=Substance.ComponentRole.Solvent),
                                    Substance.MoleFraction(1.0))

    methyl_propanoate_property = SolvationFreeEnergy(thermodynamic_state=ThermodynamicState(298.15*unit.kelvin,
                                                                                            1.0*unit.atmosphere),
                                                     phase=PropertyPhase.Liquid,
                                                     substance=methyl_propanoate,
                                                     value=-2.93*unit.kilocalorie / unit.mole,
                                                     uncertainty=0.6*unit.kilocalorie / unit.mole,
                                                     source=MeasurementSource(doi=' 10.1021/ct050097l'))

    benzamide = Substance()
    benzamide.add_component(Substance.Component('c1ccc(cc1)C(=O)N', role=Substance.ComponentRole.Solute),
                            Substance.ExactAmount(1))
    benzamide.add_component(Substance.Component('O', role=Substance.ComponentRole.Solvent),
                            Substance.MoleFraction(1.0))

    benzamide_property = SolvationFreeEnergy(thermodynamic_state=ThermodynamicState(298.15*unit.kelvin,
                                                                                    1.0*unit.atmosphere),
                                             phase=PropertyPhase.Liquid,
                                             substance=benzamide,
                                             value=-11.0*unit.kilocalorie / unit.mole,
                                             uncertainty=0.2*unit.kilocalorie / unit.mole,
                                             source=MeasurementSource(doi=' 10.1021/ct050097l'))

    data_set = PhysicalPropertyDataSet()
    data_set.properties[butan_1_ol.identifier] = [butan_1_ol_property]
    data_set.properties[methyl_propanoate.identifier] = [methyl_propanoate_property]
    data_set.properties[benzamide.identifier] = [benzamide_property]

    return data_set


def main():

    setup_timestamp_logging()

    # Load in the force field
    force_field_path = 'smirnoff99Frosst-1.1.0.offxml'
    force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

    # Create a data set containing three solvation free energies.
    data_set = _create_data_set()

    # Set up the compute backend which will run the calculations.
    working_directory = 'working_directory'
    storage_directory = 'storage_directory'

    queue_resources = QueueWorkerResources(number_of_threads=1,
                                           number_of_gpus=1,
                                           preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                           per_thread_memory_limit=5 * unit.gigabyte,
                                           wallclock_time_limit="05:59")

    worker_script_commands = [
        'conda activate propertyestimator',
        'module load cuda/10.1'
    ]

    calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                         maximum_number_of_workers=3,
                                         resources_per_worker=queue_resources,
                                         queue_name='gpuqueue',
                                         setup_script_commands=worker_script_commands,
                                         adaptive_interval='1000ms',
                                         adaptive_class=CustomAdaptive)

    # Set up a backend to cache simulation data in.
    storage_backend = LocalFileStorage(storage_directory)

    # Spin up the server object.
    EvaluatorServer(calculation_backend=calculation_backend,
                            storage_backend=storage_backend,
                            port=8005,
                            working_directory=working_directory)

    # Request the estimates.
    property_estimator = client.EvaluatorClient(client.ConnectionOptions(server_port=8005))

    options = RequestOptions()
    options.allowed_calculation_layers = ['SimulationLayer']

    workflow_options = WorkflowOptions(WorkflowOptions.ConvergenceMode.NoChecks)

    options.workflow_options = {'SolvationFreeEnergy': {'SimulationLayer': workflow_options}}
    options.workflow_schemas = {
        'SolvationFreeEnergy': {'SimulationLayer': _get_fixed_lambda_schema(workflow_options)}
    }

    request = property_estimator.request_estimate(property_set=data_set,
                                                  force_field_source=force_field_source,
                                                  options=options)

    # Wait for the results.
    results = request.results(True, 5)

    # Save the result to file.
    with open('solvation_free_energy_simulation.json', 'wb') as file:

        json_results = json.dumps(results, sort_keys=True, indent=2,
                                  separators=(',', ': '), cls=TypedJSONEncoder)

        file.write(json_results.encode('utf-8'))


if __name__ == "__main__":
    main()
