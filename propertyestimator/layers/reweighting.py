"""
The simulation reweighting estimation layer.
"""

import numpy as np
from typing import List

import pymbar
from simtk import unit

from propertyestimator.layers import register_calculation_layer, PropertyCalculationLayer
from propertyestimator.properties import CalculationSource
from propertyestimator.storage import StoredSimulationData
from propertyestimator.utils.serialization import serialize_force_field, deserialize_force_field
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.properties.plugins import registered_properties


@register_calculation_layer()
class ReweightingLayer(PropertyCalculationLayer):
    """A calculation layer which aims to calculate physical properties by
    reweighting the results of previous calculations.

    .. warning :: This class has not yet been implemented.
    """

    @staticmethod
    def schedule_calculation(calculation_backend, storage_backend, layer_directory,
                             data_model, callback, synchronous=False):

        parameter_set = storage_backend.retrieve_force_field(data_model.parameter_set_id)

        reweighting_futures = []

        for physical_property in data_model.queued_properties:

            existing_data = storage_backend.retrieve_simulation_data(str(physical_property.substance))

            reweighting_future = calculation_backend.submit_task(ReweightingLayer.perform_reweighting,
                                                                 physical_property,
                                                                 data_model.parameter_set_id,
                                                                 serialize_force_field(parameter_set),
                                                                 existing_data)

            reweighting_futures.append(reweighting_future)

        PropertyCalculationLayer._await_results(calculation_backend,
                                                storage_backend,
                                                layer_directory,
                                                data_model,
                                                callback,
                                                reweighting_futures,
                                                synchronous)

    @staticmethod
    def perform_reweighting(physical_property, parameter_set_id, serialized_parameter_set, existing_data, **kwargs):
        """A placeholder method that would be used to attempt
        to reweight previous calculations to yield the desired
        property.

        .. warning :: This method has not yet been implemented.

        Parameters
        ----------
        physical_property: :obj:`propertyestimator.properties.PhysicalProperty`
            The physical property to attempt to estimate by reweighting.
        parameter_set_id: str
            The unique id given to this set of force field parameters.
        serialized_parameter_set: Dict[int, str]
            The force field parameters to use when estimating the property, which
            has been serialized with `serialize_force_field`
        existing_data: List[StoredSimulationData]
            Data which has been stored from previous calculations on systems
            of the same composition as the desired property.

        """

        parameter_set = deserialize_force_field(serialized_parameter_set)

        particle_counts = np.array([data.trajectory_data.n_chains for data in existing_data])
        maximum_molecule_count = particle_counts.max()

        # Only retain data which has the same number of molecules. For now
        # we choose the data which was calculated using the most molecules,
        # however perhaps we should instead choose data with the mode number
        # of molecules?

        useable_data = [data for data in existing_data if
                        data.trajectory_data.n_chains == maximum_molecule_count]

        # TODO: Add state of interest...

        for data in useable_data:

            # Calculate the number of uncorrelated samples per data object.
            frame_counts = np.array([data.trajectory_data.n_frames]*2)

            reduced_energies = np.zeros(2, frame_counts.max())
            observables = np.zeros(2, frame_counts.max())

            # Pull the reference state
            if data.statistics_data.has_observable(ObservableType.Enthalpy):
                energy_values = data.statistics_data.get_observable(ObservableType.Enthalpy)
            else:
                energy_values = data.statistics_data.get_observable(ObservableType.PotentialEnergy)

            beta = 1.0 / (data.thermodynamic_state.temperature * unit.MOLAR_GAS_CONSTANT_R)
            unitless_energy_values = beta * energy_values

            reduced_energies[0] = np.array(unitless_energy_values)

            # Set up the target state.

            # Resample the trajectories using the different parameter set.
            reduced_energies[1], frame_counts[1] = ReweightingLayer.resample_data(data.trajectory_data,
                                                                                  parameter_set)

            property_class = registered_properties[physical_property.type]

            # observables[0] = property_class.calculate_observable(data.trajectory_data, data.statistics_data,
            #                                                      data.parameter_set_id)
            # observables[0] = property_class.calculate_observable(data)

            mbar = pymbar.MBAR(reduced_energies, frame_counts, verbose=False, relative_tolerance=1e-12)
            results = mbar.computeExpectations(observables, state_dependent=True)

            all_values = results['mu']
            all_uncertainties = results['sigma']
    
        # physical_property.value = all_values[len(all_values) - 1]
        # physical_property.uncertainty = all_uncertainties[len(all_uncertainties) - 1]
        #
        # physical_property.source = CalculationSource()
        #
        # physical_property.source.fidelity = ReweightingLayer.__name__
        # physical_property.source.provenance = {
        #     'data_sources': []  # TODO: Add tags to data sources
        # }

    @staticmethod
    def resample_data(trajectory_to_resample, force_field):
        """Resample the frames of a trajectory using a different set
        of force field parameters than were used when generating the
        trajectory.

        Parameters
        ----------
        trajectory_to_resample: :obj:`mdtraj.Trajectory`
            The trajectory to resample.
        force_field: :obj:`openforcefield.typing.engines.smirnoff.ForceField`
            The force field to use when calculating the energy of the frames of
            the trajectory.

        Returns
        -------
        np.ndarray
            The resampled energies.
        int
            The number of resampled frames.
        """
        # TODO: Get resampled energies from OpenMM
        return np.zeros(1, 1), 1
