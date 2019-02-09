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
            existing_force_fields = {}

            for data in existing_data:

                existing_force_field = storage_backend.retrieve_force_field(data.parameter_set_id)
                existing_force_fields[data.parameter_set_id] = serialize_force_field(existing_force_field)

            existing_force_fields[data_model.parameter_set_id] = serialize_force_field(parameter_set)

            reweighting_future = calculation_backend.submit_task(ReweightingLayer.perform_reweighting,
                                                                 physical_property,
                                                                 data_model.parameter_set_id,
                                                                 existing_data,
                                                                 existing_force_fields)

            reweighting_futures.append(reweighting_future)

        PropertyCalculationLayer._await_results(calculation_backend,
                                                storage_backend,
                                                layer_directory,
                                                data_model,
                                                callback,
                                                reweighting_futures,
                                                synchronous)

    @staticmethod
    def _get_reduced_reference_energies(physical_property, data):
        """Get the reduced energies of a reference state

        Parameters
        ----------
        physical_property: PhysicalProperty
            The physical property being estimated.
        data: StoredSimulationData
            The reference data to get the energies from

        Returns
        -------
        :obj:`numpy.ndarray`
            None if the reference data was not collected in the same ensemble as
            the target property, otherwise the reduced energies of the reference
            state.
        """
        energy_values = None

        # Pull the reference state
        if (physical_property.thermodynamic_state.pressure is not None and
            data.statistics_data.has_observable(ObservableType.Enthalpy)):

            # Assume the NPT ensemble.
            energy_values = data.statistics_data.get_observable(ObservableType.Enthalpy) - \
                            data.statistics_data.get_observable(ObservableType.KineticEnergy)

        elif physical_property.thermodynamic_state.pressure is None:

            # Assume the NVT ensemble.
            energy_values = data.statistics_data.get_observable(ObservableType.PotentialEnergy)

        if energy_values is None:

            # The reference and target states are not compatible.
            return None

        beta = 1.0 / (data.thermodynamic_state.temperature * unit.MOLAR_GAS_CONSTANT_R)
        unitless_energy_values = beta * energy_values

        return np.array(unitless_energy_values)

    @staticmethod
    def _get_reduced_target_energies(physical_property, target_force_field_id, target_force_field, data):
        """Get the reduced energies of a target state

        Parameters
        ----------
        physical_property: PhysicalProperty
            The physical property being estimated.
        target_force_field_id: str
            The id of the target force field parameters.
        target_force_field: ForceField
            The target set of force field parameters to estimate the property
            with.
        data: StoredSimulationData
            The reference data to get the energies from

        Returns
        -------
        :obj:`numpy.ndarray`
            The reduced energies of the target state.
        """

        if target_force_field_id == data.parameter_set_id:

            # If the parameters haven't changed, we only need to swap out the T / P?
            energy_values = data.statistics_data.get_observable(ObservableType.PotentialEnergy)

            if data.thermodynamic_state.pressure is not None:

                volumes = data.statistics_data.get_observable(ObservableType.Volume)
                energy_values += volumes * physical_property.thermodynamic_state.pressure * unit.AVOGADRO_CONSTANT_NA

            beta = 1.0 / (physical_property.thermodynamic_state.temperature * unit.MOLAR_GAS_CONSTANT_R)
            unitless_energy_values = beta * energy_values

            return np.array(unitless_energy_values)

        return ReweightingLayer.resample_data(data.trajectory_data, physical_property.thermodynamic_state,
                                              target_force_field)

    @staticmethod
    def perform_reweighting(physical_property, parameter_set_id, existing_data,
                            existing_force_fields, **kwargs):

        """A placeholder method that would be used to attempt
        to reweight previous calculations to yield the desired
        property.

        Warnings
        --------
        This method has not yet been implemented.

        Parameters
        ----------
        physical_property: :obj:`propertyestimator.properties.PhysicalProperty`
            The physical property to attempt to estimate by reweighting.
        parameter_set_id: str
            The id of the force field parameters which the property should be
            estimated with.
        existing_data: List[propertyestimator.storage.StoredSimulationData]
            Data which has been stored from previous calculations on systems
            of the same composition as the desired property.
        existing_force_fields: Dict[str, Dict[int, str]]
            A dictionary of all of the force field parameters referenced by the
            `existing_data`, which have been serialized with `serialize_force_field`
        """

        target_force_field = deserialize_force_field(existing_force_fields[parameter_set_id])

        particle_counts = np.array([data.trajectory_data.n_chains for data in existing_data])
        maximum_molecule_count = particle_counts.max()

        # Only retain data which has the same number of molecules. For now
        # we choose the data which was calculated using the most molecules,
        # however perhaps we should instead choose data with the mode number
        # of molecules?
        useable_data = [data for data in existing_data if
                        data.trajectory_data.n_chains == maximum_molecule_count]

        for data in useable_data:

            # Calculate the number of uncorrelated samples per data object.
            frame_counts = np.array([data.trajectory_data.n_frames]*2)

            reduced_energies = np.zeros(2, frame_counts.max())
            observables = np.zeros(2, frame_counts.max())

            # The reference state energies.
            reduced_energies[0] = ReweightingLayer._get_reduced_reference_energies(physical_property, data)

            # The target state energies.
            reduced_energies[1] = ReweightingLayer._get_reduced_target_energies(physical_property, parameter_set_id,
                                                                                target_force_field, data)

            property_class = registered_properties[physical_property.type]

            # Calculate the
            observables[1] = property_class.calculate_observable(data)

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
    def resample_data(trajectory_to_resample, thermodynamic_state, force_field):
        """Resample the frames of a trajectory using a different set
        of force field parameters than were used when generating the
        trajectory.

        Parameters
        ----------
        trajectory_to_resample: :obj:`mdtraj.Trajectory`
            The trajectory to resample.
        thermodynamic_state: :obj:`propertyestimator.thermodynamics.ThermodynamicState`
            The thermodynamic state to resample the statistics at.
        force_field: :obj:`openforcefield.typing.engines.smirnoff.ForceField`
            The force field to use when calculating the energy of the frames of
            the trajectory.

        Returns
        -------
        :obj:`numpy.ndarray`
            The resampled energies.
        """
        # TODO: Get resampled energies from OpenMM
        return np.zeros(1, 1)
