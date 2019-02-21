"""
The simulation reweighting estimation layer.
"""
import copy
import pickle
from os import path

import numpy as np
from simtk import unit

from propertyestimator.layers import register_calculation_layer, PropertyCalculationLayer
from propertyestimator.layers.layers import CalculationLayerResult
from propertyestimator.utils import create_molecule_from_smiles
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import serialize_force_field, deserialize_force_field
from propertyestimator.utils.statistics import ObservableType


@register_calculation_layer()
class ReweightingLayer(PropertyCalculationLayer):
    """A calculation layer which aims to calculate physical properties by
    reweighting the results of previous calculations.

    .. warning :: This class is still heavily under development and is subject to
                 rapid changes.
    """

    @staticmethod
    def schedule_calculation(calculation_backend, storage_backend, layer_directory,
                             data_model, callback, synchronous=False):

        force_field = storage_backend.retrieve_force_field(data_model.force_field_id)

        reweighting_futures = []

        for physical_property in data_model.queued_properties:

            existing_data = storage_backend.retrieve_simulation_data(physical_property.substance, True)

            if len(existing_data) == 0:
                continue

            # Take data from the storage backend and save it in the working directory.
            temporary_data_paths = {}
            temporary_force_field_paths = {}

            for substance_id in existing_data:

                temporary_data_paths[substance_id] = []

                for data in existing_data[substance_id]:

                    temporary_data_path = path.join(layer_directory, data.unique_id)

                    with open(temporary_data_path, 'wb') as file:
                        pickle.dump(data, file)

                    temporary_data_paths[substance_id].append(temporary_data_path)

                    temporary_force_field_path = path.join(layer_directory, data.force_field_id)
                    existing_force_field = storage_backend.retrieve_force_field(data.force_field_id)

                    with open(temporary_force_field_path, 'wb') as file:
                        pickle.dump(serialize_force_field(existing_force_field), file)

                    temporary_force_field_paths[data.force_field_id] = temporary_force_field_path

            temporary_force_field_path = path.join(layer_directory, data_model.force_field_id)

            with open(temporary_force_field_path, 'wb') as file:
                pickle.dump(serialize_force_field(force_field), file)

            temporary_force_field_paths[data_model.force_field_id] = temporary_force_field_path

            # Pass this data to the backend to attempt to reweight.
            reweighting_future = calculation_backend.submit_task(ReweightingLayer.perform_reweighting,
                                                                 physical_property,
                                                                 data_model.options,
                                                                 data_model.force_field_id,
                                                                 temporary_data_paths,
                                                                 temporary_force_field_paths)

            reweighting_futures.append(reweighting_future)

        PropertyCalculationLayer._await_results(calculation_backend,
                                                storage_backend,
                                                layer_directory,
                                                data_model,
                                                callback,
                                                reweighting_futures,
                                                synchronous)

    @staticmethod
    def perform_reweighting(physical_property, options, force_field_id, existing_data_paths,
                            existing_force_field_paths, available_resources, **kwargs):
        """A placeholder method that would be used to attempt
        to reweight previous calculations to yield the desired
        property.

        Parameters
        ----------
        physical_property: PhysicalProperty
            The physical property to attempt to estimate by reweighting.
        options: PropertyEstimatorOptions
            The options to use when executing the reweighting layer.
        force_field_id: str
            The id of the force field parameters which the property should be
            estimated with.
        existing_data_paths: dict of str and str
            A dictionary of file path to data which has been stored from previous calculations
            on systems of the same composition as the desired property.
        existing_force_field_paths: dict of str and str
            A dictionary of file paths to the force field parameters referenced by the
            `existing_data_paths`, which have been serialized with `serialize_force_field`
        available_resources: ComputeResources
            The compute resources available for this reweighting calculation.
        """
        property_class = physical_property.__class__

        if not hasattr(property_class, 'reweight'):
            return None

        existing_data = {}
        existing_force_fields = {}

        for substance_id in existing_data_paths:

            existing_data[substance_id] = []

            for data_path in existing_data_paths[substance_id]:

                with open(data_path, 'rb') as file:
                    existing_data[substance_id].append(pickle.load(file))

        for force_field_id in existing_force_field_paths:

            with open(existing_force_field_paths[force_field_id], 'rb') as file:
                existing_force_fields[force_field_id] = deserialize_force_field(pickle.load(file))

        reweighted_property = property_class.reweight(physical_property, options, force_field_id, existing_data,
                                                      existing_force_fields, available_resources)

        return_object = CalculationLayerResult()
        return_object.property_id = physical_property.id

        return_object.calculated_property = reweighted_property
        return_object.data_to_store = []

        return return_object

    @staticmethod
    def get_reduced_potential(substance, thermodynamic_state, target_force_field_id,
                              target_force_field, existing_data, available_resources):
        """Get the reduced potential from an existing data set. If the target force field
        id does not match the force field id used by the existing data, the reduced potential
        will be re-evaluated using the target force field.

        Parameters
        ----------
        substance: Mixture
            The substance for which to get reduced energies for.
        thermodynamic_state: ThermodynamicState
            The thermodynamic state to use when reducing the energies.
        target_force_field_id: str
            The id of the target force field parameters.
        target_force_field: openforcefield.typing.engines.smirnoff.ForceField
            The target set of force field parameters.
        existing_data: StoredSimulationData
            The reference data to get the energies from.
        available_resources: ComputeResources
            The compute resources available when generating the reduced potentials.

        Returns
        -------
        numpy.ndarray shape=(num_frames), dtype=float
            The reduced potential of the target state.
        """

        potential_energies = None

        if target_force_field_id != existing_data.force_field_id:

            potential_energies = ReweightingLayer.resample_data(substance, existing_data.trajectory_data,
                                                                target_force_field, available_resources)

        else:
            potential_energies = copy.deepcopy(existing_data.statistics_data.get_observable(
                ObservableType.PotentialEnergy))

        beta = 1.0 / (thermodynamic_state.temperature * unit.MOLAR_GAS_CONSTANT_R)

        reduced_potentials = potential_energies

        if thermodynamic_state.pressure is not None:

            volumes = existing_data.statistics_data.get_observable(ObservableType.Volume)
            reduced_potentials += volumes * thermodynamic_state.pressure * unit.AVOGADRO_CONSTANT_NA

        reduced_potentials *= beta

        return reduced_potentials

    @staticmethod
    def resample_data(substance, trajectory_to_resample,
                      force_field, available_resources):
        """Resample the frames of a trajectory using a different set
        of force field parameters than were used when generating the
        trajectory.

        Parameters
        ----------
        substance: Mixture
            The substance being resampled.
        trajectory_to_resample: mdtraj.Trajectory
            The trajectory to resample.
        force_field: openforcefield.typing.engines.smirnoff.ForceField
            The force field to use when calculating the energy of the frames of
            the trajectory.
        available_resources: ComputeResources
            The compute resources available when resampling the data.

        Returns
        -------
        numpy.ndarray
            The resampled energies.
        """

        molecules = [create_molecule_from_smiles(component.smiles) for component in substance.components]

        from openforcefield.typing.engines import smirnoff
        from simtk.openmm import Platform, Context, VerletIntegrator

        topology = trajectory_to_resample.topology.to_openmm

        system = force_field.createSystem(topology,
                                          molecules,
                                          nonbondedMethod=smirnoff.PME,
                                          chargeMethod='OECharges_AM1BCCSym')

        if system is None:

            return PropertyEstimatorException(directory='',
                                              message='Failed to create a system from the'
                                                      'provided topology and molecules')

        context = None
        integrator = VerletIntegrator(0.002 * unit.picoseconds)

        if available_resources.number_of_gpus > 0:

            # noinspection PyTypeChecker,PyCallByClass
            gpu_platform = Platform.getPlatformByName(str(available_resources.preferred_gpu_toolkit))
            properties = {'DeviceIndex': ','.join(range(available_resources.number_of_gpus))}

            context = Context(system, integrator, gpu_platform, properties)

        else:

            # noinspection PyTypeChecker,PyCallByClass
            cpu_platform = Platform.getPlatformByName('CPU')
            properties = {'Threads': str(available_resources.number_of_threads)}

            context = Context(system, integrator, cpu_platform, properties)

        resampled_energies = np.zeros((1, trajectory_to_resample.n_frames))

        for frame_index in range(trajectory_to_resample.n_frames):

            positions = trajectory_to_resample.openmm_positions(frame_index)
            box_vectors = trajectory_to_resample.openmm_boxes(frame_index)

            context.context.setPeriodicBoxVectors(*box_vectors)
            context.context.setPositions(positions)

            # set box vectors
            resampled_energies[frame_index] = context.getState(getEnergy=True).getPotentialEnergy()

        return resampled_energies
