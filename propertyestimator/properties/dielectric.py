"""
A collection of dielectric physical property definitions.
"""
import copy

import numpy as np
from simtk import openmm
from simtk.openmm import XmlSerializer

from propertyestimator import unit
from propertyestimator.attributes import UNDEFINED
from propertyestimator.datasets import PhysicalProperty, PropertyPhase
from propertyestimator.datasets.thermoml.plugins import thermoml_property
from propertyestimator.layers import register_calculation_schema
from propertyestimator.layers.reweighting import ReweightingLayer, ReweightingSchema
from propertyestimator.layers.simulation import SimulationLayer, SimulationSchema
from propertyestimator.protocols import analysis, reweighting
from propertyestimator.protocols.utils import (
    generate_base_reweighting_protocols,
    generate_base_simulation_protocols,
    generate_gradient_protocol_group,
)
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import timeseries
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import bootstrap
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.schemas import WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath


@workflow_protocol()
class ExtractAverageDielectric(analysis.AverageTrajectoryProperty):
    """Extracts the average dielectric constant from a simulation trajectory.
    """

    system_path = InputAttribute(
        docstring="The path to the XML system object which defines the forces present in the system.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic state at which the trajectory was generated.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    dipole_moments = OutputAttribute(
        docstring="The raw (possibly correlated) dipole moments which were used in "
        "the dielectric calculation.",
        type_hint=unit.Quantity,
    )
    volumes = OutputAttribute(
        docstring="The raw (possibly correlated) which were used in the dielectric calculation.",
        type_hint=unit.Quantity,
    )

    uncorrelated_volumes = OutputAttribute(
        docstring="The uncorrelated volumes which were used in the dielectric "
        "calculation.",
        type_hint=unit.Quantity,
    )

    def _bootstrap_function(self, **sample_kwargs):
        """Calculates the static dielectric constant from an
        array of dipoles and volumes.

        Notes
        -----
        The static dielectric constant is taken from for Equation 7 of [1]

        References
        ----------
        [1] A. Glattli, X. Daura and W. F. van Gunsteren. Derivation of an improved simple point charge
            model for liquid water: SPC/A and SPC/L. J. Chem. Phys. 116(22):9811-9828, 2002

        Parameters
        ----------
        sample_kwargs: dict of str and np.ndarray
            A key words dictionary of the bootstrap sample data, where the
            sample data is a numpy array of shape=(num_frames, num_dimensions)
            with dtype=float. The kwargs should include the dipole moment and
            the system volume

        Returns
        -------
        float
            The unitless static dielectric constant
        """

        dipole_moments = sample_kwargs["dipoles"]
        volumes = sample_kwargs["volumes"]

        temperature = self.thermodynamic_state.temperature

        dipole_mu = dipole_moments.mean(0)
        shifted_dipoles = dipole_moments - dipole_mu

        dipole_variance = (shifted_dipoles * shifted_dipoles).sum(-1).mean(0) * (
            unit.elementary_charge * unit.nanometers
        ) ** 2

        volume = volumes.mean() * unit.nanometer ** 3

        e0 = 8.854187817e-12 * unit.farad / unit.meter  # Taken from QCElemental

        dielectric_constant = 1.0 + dipole_variance / (
            3 * unit.boltzmann_constant * temperature * volume * e0
        )

        return dielectric_constant

    def _extract_charges(self):
        """Extracts all of the charges from a system object.

        Returns
        -------
        list of float
        """
        from simtk import unit as simtk_unit

        charge_list = []

        with open(self._system_path, "r") as file:
            system = XmlSerializer.deserialize(file.read())

        for force_index in range(system.getNumForces()):

            force = system.getForce(force_index)

            if not isinstance(force, openmm.NonbondedForce):
                continue

            for atom_index in range(force.getNumParticles()):
                charge = force.getParticleParameters(atom_index)[0]
                charge = charge.value_in_unit(simtk_unit.elementary_charge)

                charge_list.append(charge)

        return charge_list

    def _extract_dipoles_and_volumes(self):
        """Extract the systems dipole moments and volumes.

        Returns
        -------
        numpy.ndarray
            The dipole moments of the trajectory (shape=(n_frames, 3), dtype=float)
        numpy.ndarray
            The volumes of the trajectory (shape=(n_frames, 1), dtype=float)
        """
        import mdtraj

        dipole_moments = []
        volumes = []
        charge_list = self._extract_charges()

        for chunk in mdtraj.iterload(
            self.trajectory_path, top=self.input_coordinate_file, chunk=50
        ):

            dipole_moments.extend(mdtraj.geometry.dipole_moments(chunk, charge_list))
            volumes.extend(chunk.unitcell_volumes)

        dipole_moments = np.array(dipole_moments)
        volumes = np.array(volumes)

        return dipole_moments, volumes

    def _execute(self, directory, available_resources):

        super(ExtractAverageDielectric, self)._execute(directory, available_resources)

        # Extract the dipoles
        dipole_moments, volumes = self._extract_dipoles_and_volumes()
        self.dipole_moments = dipole_moments * unit.dimensionless

        (
            dipole_moments,
            self.equilibration_index,
            self.statistical_inefficiency,
        ) = timeseries.decorrelate_time_series(dipole_moments)

        uncorrelated_length = len(volumes) - self.equilibration_index

        sample_indices = timeseries.get_uncorrelated_indices(
            uncorrelated_length, self.statistical_inefficiency
        )
        sample_indices = [index + self.equilibration_index for index in sample_indices]

        self.volumes = volumes * unit.nanometer ** 3
        uncorrelated_volumes = volumes[sample_indices]

        self.uncorrelated_values = dipole_moments * unit.dimensionless
        self.uncorrelated_volumes = uncorrelated_volumes * unit.nanometer ** 3

        value, uncertainty = bootstrap(
            self._bootstrap_function,
            self.bootstrap_iterations,
            self.bootstrap_sample_size,
            dipoles=dipole_moments,
            volumes=uncorrelated_volumes,
        )

        self.value = EstimatedQuantity(
            value * unit.dimensionless, uncertainty * unit.dimensionless, self.id
        )


@workflow_protocol()
class ReweightDielectricConstant(reweighting.BaseMBARProtocol):
    """Reweights a set of dipole moments (`reference_observables`) and volumes
    (`reference_volumes`) using MBAR, and then combines these to yeild the reweighted
    dielectric constant. Uncertainties in the dielectric constant are determined
    by bootstrapping.
    """

    reference_dipole_moments = InputAttribute(
        docstring="A Quantity wrapped np.ndarray of the dipole moments of each "
        "of the reference states.",
        type_hint=list,
        default_value=UNDEFINED,
    )
    reference_volumes = InputAttribute(
        docstring="A Quantity wrapped np.ndarray of the volumes of each of the "
        "reference states.",
        type_hint=list,
        default_value=UNDEFINED,
    )

    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic state at which the trajectory was generated.",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    def __init__(self, protocol_id):
        super().__init__(protocol_id)
        self.bootstrap_uncertainties = True

    def _bootstrap_function(
        self,
        reference_reduced_potentials,
        target_reduced_potentials,
        **reference_observables,
    ):

        assert len(reference_observables) == 3

        transposed_observables = {}

        for key in reference_observables:
            transposed_observables[key] = np.transpose(reference_observables[key])

        values, _, _ = self._reweight_observables(
            np.transpose(reference_reduced_potentials),
            np.transpose(target_reduced_potentials),
            **transposed_observables,
        )

        average_squared_dipole = values["dipoles_sqr"]
        average_dipole_squared = np.linalg.norm(values["dipoles"])

        dipole_variance = (average_squared_dipole - average_dipole_squared) * (
            unit.elementary_charge * unit.nanometers
        ) ** 2

        volume = values["volumes"] * unit.nanometer ** 3

        e0 = 8.854187817e-12 * unit.farad / unit.meter  # Taken from QCElemental

        dielectric_constant = 1.0 + dipole_variance / (
            3
            * unit.boltzmann_constant
            * self.thermodynamic_state.temperature
            * volume
            * e0
        )

        return dielectric_constant

    def _execute(self, directory, available_resources):

        if len(self.reference_dipole_moments) == 0:
            raise ValueError("There were no dipole moments to reweight.")

        if len(self.reference_volumes) == 0:
            raise ValueError("There were no volumes to reweight.")

        if not isinstance(
            self.reference_dipole_moments[0], unit.Quantity
        ) or not isinstance(self.reference_volumes[0], unit.Quantity):

            raise ValueError(
                "The reference observables should be a list of "
                "unit.Quantity wrapped ndarray's.",
            )

        if len(self.reference_dipole_moments) != len(self.reference_volumes):

            raise ValueError(
                "The number of reference dipoles does not match the "
                "number of reference volumes.",
            )

        for reference_dipoles, reference_volumes in zip(
            self.reference_dipole_moments, self.reference_volumes
        ):

            if len(reference_dipoles) == len(reference_volumes):
                continue

            raise ValueError(
                "The number of reference dipoles does not match the "
                "number of reference volumes.",
            )

        self._reference_observables = self.reference_dipole_moments

        dipole_moments = self._prepare_observables_array(self.reference_dipole_moments)
        dipole_moments_sqr = np.array(
            [[np.dot(dipole, dipole) for dipole in np.transpose(dipole_moments)]]
        )

        volumes = self._prepare_observables_array(self.reference_volumes)

        if self.bootstrap_uncertainties:

            self._execute_with_bootstrapping(
                unit.dimensionless,
                dipoles=dipole_moments,
                dipoles_sqr=dipole_moments_sqr,
                volumes=volumes,
            )
        else:

            raise ValueError(
                "Dielectric constant can only be reweighted in conjunction "
                "with bootstrapped uncertainties.",
            )


@thermoml_property(
    "Relative permittivity at zero frequency", supported_phases=PropertyPhase.Liquid,
)
class DielectricConstant(PhysicalProperty):
    """A class representation of a dielectric property"""

    @staticmethod
    def default_simulation_schema(
        absolute_tolerance=UNDEFINED, relative_tolerance=UNDEFINED, n_molecules=1000
    ):
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations.

        Parameters
        ----------
        absolute_tolerance: unit.Quantity, optional
            The absolute tolerance to estimate the property to within.
        relative_tolerance: float
            The tolerance (as a fraction of the properties reported
            uncertainty) to estimate the property to within.
        n_molecules: int
            The number of molecules to use in the simulation.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """
        assert absolute_tolerance == UNDEFINED or relative_tolerance == UNDEFINED

        calculation_schema = SimulationSchema()
        calculation_schema.absolute_tolerance = absolute_tolerance
        calculation_schema.relative_tolerance = relative_tolerance

        # Define the protocol which will extract the average dielectric constant
        # from the results of a simulation.
        extract_dielectric = ExtractAverageDielectric("extract_dielectric")
        extract_dielectric.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        # Define the protocols which will run the simulation itself.
        use_target_uncertainty = (
            absolute_tolerance != UNDEFINED or relative_tolerance != UNDEFINED
        )

        protocols, value_source, output_to_store = generate_base_simulation_protocols(
            extract_dielectric, use_target_uncertainty, n_molecules=n_molecules,
        )

        # Make sure the input of the analysis protcol is properly hooked up.
        extract_dielectric.system_path = ProtocolPath(
            "system_path", protocols.assign_parameters.id
        )

        # Dielectric constants typically take longer to converge, so we need to
        # reflect this in the maximum number of convergence iterations.
        protocols.converge_uncertainty.max_iterations = 400

        # Set up the gradient calculations. For dielectric constants, we need to use
        # a slightly specialised reweighting protocol which we set up here.
        coordinate_source = ProtocolPath(
            "output_coordinate_file", protocols.equilibration_simulation.id
        )
        trajectory_source = ProtocolPath(
            "trajectory_file_path",
            protocols.converge_uncertainty.id,
            protocols.production_simulation.id,
        )
        statistics_source = ProtocolPath(
            "statistics_file_path",
            protocols.converge_uncertainty.id,
            protocols.production_simulation.id,
        )

        gradient_mbar_protocol = ReweightDielectricConstant("gradient_mbar")
        gradient_mbar_protocol.reference_dipole_moments = [
            ProtocolPath(
                "dipole_moments",
                protocols.converge_uncertainty.id,
                extract_dielectric.id,
            )
        ]
        gradient_mbar_protocol.reference_volumes = [
            ProtocolPath(
                "volumes", protocols.converge_uncertainty.id, extract_dielectric.id
            )
        ]
        gradient_mbar_protocol.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        gradient_mbar_protocol.reference_reduced_potentials = statistics_source

        (
            gradient_group,
            gradient_replicator,
            gradient_source,
        ) = generate_gradient_protocol_group(
            gradient_mbar_protocol,
            ProtocolPath("force_field_path", "global"),
            coordinate_source,
            trajectory_source,
            statistics_source,
        )

        # Build the workflow schema.
        schema = WorkflowSchema()

        schema.protocol_schemas = [
            protocols.build_coordinates.schema,
            protocols.assign_parameters.schema,
            protocols.energy_minimisation.schema,
            protocols.equilibration_simulation.schema,
            protocols.converge_uncertainty.schema,
            protocols.extract_uncorrelated_trajectory.schema,
            protocols.extract_uncorrelated_statistics.schema,
            gradient_group.schema,
        ]

        schema.protocol_replicators = [gradient_replicator]

        schema.outputs_to_store = {"full_system": output_to_store}

        schema.gradients_sources = [gradient_source]
        schema.final_value_source = value_source

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @staticmethod
    def default_reweighting_schema(
        absolute_tolerance=UNDEFINED,
        relative_tolerance=UNDEFINED,
        n_effective_samples=50,
    ):
        """Returns the default calculation schema to use when estimating
        this property by reweighting existing data.

        Parameters
        ----------
        absolute_tolerance: unit.Quantity, optional
            The absolute tolerance to estimate the property to within.
        relative_tolerance: float
            The tolerance (as a fraction of the properties reported
            uncertainty) to estimate the property to within.
        n_effective_samples: int
            The minimum number of effective samples to require when
            reweighting the cached simulation data.

        Returns
        -------
        ReweightingSchema
            The schema to follow when estimating this property.
        """
        assert absolute_tolerance == UNDEFINED or relative_tolerance == UNDEFINED

        calculation_schema = ReweightingSchema()
        calculation_schema.absolute_tolerance = absolute_tolerance
        calculation_schema.relative_tolerance = relative_tolerance

        data_replicator_id = "data_replicator"

        # Set up a protocol to extract the dielectric constant from the stored data.
        extract_dielectric = ExtractAverageDielectric(
            f"calc_dielectric_$({data_replicator_id})"
        )

        # For the dielectric constant, we employ a slightly more advanced reweighting
        # protocol set up for calculating fluctuation properties.
        reweight_dielectric = ReweightDielectricConstant("reweight_dielectric")
        reweight_dielectric.reference_dipole_moments = ProtocolPath(
            "uncorrelated_values", extract_dielectric.id
        )
        reweight_dielectric.reference_volumes = ProtocolPath(
            "uncorrelated_volumes", extract_dielectric.id
        )
        reweight_dielectric.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        reweight_dielectric.bootstrap_uncertainties = True
        reweight_dielectric.bootstrap_iterations = 200
        reweight_dielectric.required_effective_samples = n_effective_samples

        protocols, data_replicator = generate_base_reweighting_protocols(
            extract_dielectric, reweight_dielectric, data_replicator_id
        )

        # Make sure input is taken from the correct protocol outputs.
        extract_dielectric.system_path = ProtocolPath(
            "system_path", protocols.build_reference_system.id
        )
        extract_dielectric.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", protocols.unpack_stored_data.id
        )

        # Set up the gradient calculations
        coordinate_path = ProtocolPath(
            "output_coordinate_path", protocols.concatenate_trajectories.id
        )
        trajectory_path = ProtocolPath(
            "output_trajectory_path", protocols.concatenate_trajectories.id
        )
        statistics_path = ProtocolPath(
            "statistics_file_path", protocols.reduced_target_potential.id
        )

        reweight_dielectric_template = copy.deepcopy(reweight_dielectric)

        (
            gradient_group,
            gradient_replicator,
            gradient_source,
        ) = generate_gradient_protocol_group(
            reweight_dielectric_template,
            ProtocolPath("force_field_path", "global"),
            coordinate_path,
            trajectory_path,
            statistics_path,
            replicator_id="grad",
            effective_sample_indices=ProtocolPath(
                "effective_sample_indices", reweight_dielectric.id
            ),
        )

        schema = WorkflowSchema()
        schema.protocol_schemas = [
            *(x.schema for x in protocols),
            gradient_group.schema,
        ]
        schema.protocol_replicators = [data_replicator, gradient_replicator]
        schema.gradients_sources = [gradient_source]
        schema.final_value_source = ProtocolPath("value", protocols.mbar_protocol.id)

        calculation_schema.workflow_schema = schema
        return calculation_schema


# Register the properties via the plugin system.
register_calculation_schema(
    DielectricConstant, SimulationLayer, DielectricConstant.default_simulation_schema
)
register_calculation_schema(
    DielectricConstant, ReweightingLayer, DielectricConstant.default_reweighting_schema
)
