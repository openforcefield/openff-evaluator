"""
An API for importing a data set from the `taproom
<https://github.com/slochower/host-guest-benchmarks>`_ package.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pkg_resources
import yaml

from openff.evaluator import unit
from openff.evaluator.datasets import PhysicalPropertyDataSet, PropertyPhase, Source
from openff.evaluator.properties import HostGuestBindingAffinity
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.exceptions import MissingOptionalDependency

logger = logging.getLogger(__name__)


class TaproomSource(Source):
    """Contains metadata about the source of a host-guest binding affinity
    measurement which was pulled from the ``taproom`` package.
    """

    def __init__(
        self, doi="", comment="", technique="", host_identifier="", guest_identifier=""
    ):
        """Constructs a new MeasurementSource object.

        Parameters
        ----------
        doi : str
            The DOI for the source
        comment : str
            A description of where the value came from in the source.
        technique : str
            The technique used to measure this value.
        host_identifier : str
            The unique three letter host identifier
        guest_identifier : str
            The unique three letter guest identifier
        """

        self.doi = doi
        self.comment = comment
        self.technique = technique

        self.host_identifier = host_identifier
        self.guest_identifier = guest_identifier

    def __getstate__(self):

        return {
            "doi": self.doi,
            "comment": self.comment,
            "technique": self.technique,
            "host_identifier": self.host_identifier,
            "guest_identifier": self.guest_identifier,
        }

    def __setstate__(self, state):

        self.doi = state["doi"]
        self.comment = state["comment"]
        self.technique = state["technique"]

        self.host_identifier = state["host_identifier"]
        self.guest_identifier = state["guest_identifier"]

    def __str__(self):
        return (
            f"host={self.host_identifier} "
            f"guest={self.guest_identifier} "
            f"doi={self.doi}"
        )

    def __repr__(self):
        return f"<TaproomSource {self.__str__()}>"


class TaproomDataSet(PhysicalPropertyDataSet):
    """A dataset of host-guest binding affinity measurements which sources its data
    from the `taproom <https://github.com/slochower/host-guest-benchmarks>`_ package.

    The loaded ``HostGuestBindingAffinity`` properties will also be optionally (enabled
    by default) initialized with the metadata required by the APR estimation workflow.
    """

    def __init__(
        self,
        host_codes: List[str] = None,
        guest_codes: List[str] = None,
        host_guest_codes: Dict[str, List[str]] = None,
        exclude_systems: Dict[str, List[str]] = None,
        default_ionic_strength: Optional[unit.Quantity] = 150 * unit.millimolar,
        negative_buffer_ion: str = "[Cl-]",
        positive_buffer_ion: str = "[Na+]",
        in_vacuum: bool = False,
        attach_apr_meta_data: bool = True,
    ):
        """

        Parameters
        ----------
        host_codes
            The three letter codes of the host molecules to load from ``taproom``
            If no list is provided, all hosts will be loaded.
        guest_codes
            The three letter codes of the guest molecules to load from ``taproom``.
            If no list is provided, all guests will be loaded.
        host_guest_codes
            A dictionary containing the three letter codes of guest molecules for
            each host molecule to load from ``taproom``. This option provides greater
            control for choosing the host-guest systems. If specified, host_codes and
            guest_codes will be ignored.
        exclude_systems
            A dictionary for host-guest systems to exclude from the list generated
            from the lists of host_codes and guest_codes.
        default_ionic_strength
            The default ionic strength to use for measurements. The value
            specified in ``taproom`` will be ignored and this value used
            instead. If no value is provided, no buffer will be included.
        negative_buffer_ion
            The SMILES pattern of the negative buffer ion to use. The value
            specified in ``taproom`` will be ignored and this value used
            instead.
        positive_buffer_ion
            The SMILES pattern of the positive buffer ion to use. The value
            specified in ``taproom`` will be ignored and this value used
            instead.
        in_vacuum
            Whether to configure Taproom without solvents.
        attach_apr_meta_data
            Whether to add the metadata required for an APR based calculation
            using the ``paprika`` based workflow.
        """
        super().__init__()

        try:
            from openeye import oechem
        except ImportError:
            raise MissingOptionalDependency("openeye.oechem", False)

        unlicensed_library = "openeye.oechem" if not oechem.OEChemIsLicensed() else None

        if unlicensed_library is not None:
            raise MissingOptionalDependency(unlicensed_library, True)

        # TODO: Don't overwrite the taproom ionic strength and buffer ions.
        self._initialize(
            host_codes,
            guest_codes,
            host_guest_codes,
            exclude_systems,
            default_ionic_strength,
            negative_buffer_ion,
            positive_buffer_ion,
            in_vacuum,
            attach_apr_meta_data,
        )

    @staticmethod
    def _mol2_to_smiles(file_path: str) -> str:
        """Converts a mol2 file into a smiles string.

        Parameters
        ----------
        file_path: str
            The file path to the mol2 file.

        Returns
        -------
        str
            The smiles descriptor of the loaded molecule
        """
        from openff.toolkit.topology import Molecule

        receptor_molecule = Molecule.from_file(file_path, "MOL2")

        return receptor_molecule.to_smiles()

    @staticmethod
    def _build_substance(
        guest_smiles: Optional[str],
        host_smiles: str,
        ionic_strength: Optional[unit.Quantity],
        negative_buffer_ion: str = "[Cl-]",
        positive_buffer_ion: str = "[Na+]",
        in_vacuum: bool = False,
        toolkit="openeye",
    ):
        """Builds a substance containing a ligand and receptor solvated in an aqueous
        solution with a given ionic strength

        Parameters
        ----------
        guest_smiles
            The SMILES descriptor of the guest.
        host_smiles
            The SMILES descriptor of the host.
        ionic_strength
            The ionic strength of the aqueous solvent.

        Returns
        -------
            The built substance.
        """
        from openff.toolkit.topology import Molecule
        from simtk import unit as simtk_unit

        substance = Substance()

        if guest_smiles is not None:

            guest = Component(
                smiles=guest_smiles, role=Component.Role.Ligand, toolkit=toolkit
            )
            substance.add_component(component=guest, amount=ExactAmount(1))

        host = Component(
            smiles=host_smiles, role=Component.Role.Receptor, toolkit=toolkit
        )
        substance.add_component(component=host, amount=ExactAmount(1))

        if in_vacuum is False:
            water = Component(smiles="O", role=Component.Role.Solvent, toolkit=toolkit)
            sodium = Component(
                smiles=positive_buffer_ion,
                role=Component.Role.Solvent,
                toolkit=toolkit,
            )
            chlorine = Component(
                smiles=negative_buffer_ion,
                role=Component.Role.Solvent,
                toolkit=toolkit,
            )

            water_mole_fraction = 1.0

            if ionic_strength is not None:

                salt_mole_fraction = Substance.calculate_aqueous_ionic_mole_fraction(
                    ionic_strength
                )

                if isinstance(salt_mole_fraction, unit.Quantity):
                    # noinspection PyUnresolvedReferences
                    salt_mole_fraction = salt_mole_fraction.magnitude

                water_mole_fraction = 1.0 - salt_mole_fraction * 2

                substance.add_component(
                    component=sodium,
                    amount=MoleFraction(salt_mole_fraction),
                )
                substance.add_component(
                    component=chlorine,
                    amount=MoleFraction(salt_mole_fraction),
                )

            substance.add_component(
                component=water, amount=MoleFraction(water_mole_fraction)
            )

            host_molecule_charge = Molecule.from_smiles(host_smiles).total_charge
            guest_molecule_charge = (
                0.0 * simtk_unit.elementary_charge
                if guest_smiles is None
                else Molecule.from_smiles(guest_smiles).total_charge
            )

            net_charge = (host_molecule_charge + guest_molecule_charge).value_in_unit(
                simtk_unit.elementary_charge
            )
            n_counter_ions = abs(int(net_charge))

            if net_charge <= -0.9999:
                substance.add_component(sodium, ExactAmount(n_counter_ions))
            elif net_charge >= 0.9999:
                substance.add_component(chlorine, ExactAmount(n_counter_ions))

        return substance

    @staticmethod
    def _unnest_restraint_specs(
        restraint_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """A helper method to un-nest restraint lists parsed from a taproom
        yaml file.

        Parameters
        ----------
        restraint_specs
            The restraint specs to un-nest.
        """
        return [
            value["restraint"]
            for value in restraint_specs
            if value["restraint"] is not None
        ]

    @staticmethod
    def _generate_lambda_scaling(
        attach_lambdas: List[float],
        n_pull_windows: int,
        release_lambdas: List[float],
    ) -> Dict[str, Any]:
        """A help method to generate lambda scaling factors for use in error/convergence
        estimate.

        Parameters
        ----------
        attach_lambdas
            A list of lambdas for attach phase calculation
        n_pull_windows
            The number of pull windows
        release_lambdas
            A list of lambdas for release phase calculation

        Returns
        -------
        lambda_scaling
            A dictionary containing the lambda scaling factor for each APR window.
        """

        lambda_scaling = {}

        # Attach phase scaling
        n_windows = len(attach_lambdas)
        scale = np.zeros(n_windows)

        for i, fraction in enumerate(attach_lambdas):
            if i == 0:
                scale[i] = attach_lambdas[i + 1] / 2.0
            elif i == n_windows - 1:
                scale[i] = (1 - attach_lambdas[i - 1]) / 2.0
            else:
                scale[i] = (attach_lambdas[i + 1] - attach_lambdas[i - 1]) / 2.0

        lambda_scaling.update({"attach": scale})

        # Pull phase scaling
        scale = np.ones(n_pull_windows)

        lambda_scaling.update({"pull": scale})

        # Release phase scaling
        n_windows = len(release_lambdas)
        scale = np.zeros(n_windows)

        for i, fraction in enumerate(release_lambdas):
            if i == 0:
                scale[i] = (1 - release_lambdas[i + 1]) / 2.0
            elif i == n_windows - 1:
                scale[i] = release_lambdas[i - 1] / 2.0
            else:
                scale[i] = (release_lambdas[i - 1] - release_lambdas[i + 1]) / 2.0

        lambda_scaling.update({"release": scale})

        return lambda_scaling

    @classmethod
    def _build_metadata(
        cls,
        host_yaml_paths: Dict[str, str],
        guest_yaml_path: str,
        host_substance: Substance,
    ) -> Dict[str, Any]:
        """Constructs the metadata dictionary for a given host-guest
        system.

        Parameters
        ----------
        host_yaml_paths
            The file path to the host YAML file for each guest orientation.
        guest_yaml_path
            The file path to the guest YAML file.
        host_substance
            A substance containing only the host molecule.

        Returns
        -------
            The constructed metadata dictionary.
        """

        from paprika.restraints.read_yaml import read_yaml

        guest_spec = read_yaml(guest_yaml_path)

        guest_aliases = {
            guest_alias: atom_mask
            for guest_alias_entry in guest_spec["aliases"]
            for guest_alias, atom_mask in guest_alias_entry.items()
        }

        metadata = {
            "host_substance": host_substance,
            "host_file_paths": {},
            "guest_file_paths": {},
            "guest_restraints": cls._unnest_restraint_specs(
                guest_spec["restraints"]["guest"]
            ),
            "guest_orientation_mask": " ".join(
                [guest_aliases["G1"].strip(), guest_aliases["G2"].strip()]
            ),
            "guest_orientations": [],
            "n_guest_microstates": guest_spec["symmetry_correction"]["microstates"],
            "wall_restraints": cls._unnest_restraint_specs(
                guest_spec["restraints"]["wall_restraints"]
            ),
            "symmetry_restraints": guest_spec["symmetry_correction"]["restraints"],
        }

        for restraint in metadata["symmetry_restraints"]:
            del restraint["restraint"]

        dummy_atom_offset = metadata["guest_restraints"][0]["attach"]["target"]
        pull_distance = (
            metadata["guest_restraints"][0]["pull"]["target"] - dummy_atom_offset
        )

        metadata["dummy_atom_offset"] = dummy_atom_offset * unit.angstrom
        metadata["pull_distance"] = pull_distance * unit.angstrom

        unique_attach_lambdas = set()
        unique_n_pull_windows = set()
        unique_release_lambdas = set()

        unique_host_structures = set()

        for orientation, host_yaml_path in host_yaml_paths.items():

            host_spec = read_yaml(host_yaml_path)

            root_host_path = os.path.dirname(host_yaml_path)
            host_path = os.path.join(
                root_host_path,
                host_spec["structure"].replace(".mol2", ".pdb"),
            )
            unique_host_structures.add(host_path)

            root_complex_path = os.path.dirname(guest_yaml_path)
            complex_path = os.path.join(
                root_complex_path,
                guest_spec["complex"].replace(".pdb", f"-{orientation}.pdb"),
            )

            metadata["guest_orientations"].append(
                {
                    "coordinate_path": complex_path,
                    "static_restraints": cls._unnest_restraint_specs(
                        host_spec["restraints"]["static"]
                    ),
                    "conformational_restraints": cls._unnest_restraint_specs(
                        host_spec["restraints"]["conformational"]
                    ),
                }
            )

            unique_attach_lambdas.add(
                tuple(host_spec["calculation"]["lambda"]["attach"])
            )
            unique_n_pull_windows.add(host_spec["calculation"]["windows"]["pull"])
            unique_release_lambdas.add(
                tuple(host_spec["calculation"]["lambda"]["release"])
            )

        if len(unique_host_structures) != 1:
            raise NotImplementedError("There must only be a single host structure.")

        if (
            len(unique_attach_lambdas) != 1
            or len(unique_n_pull_windows) != 1
            or len(unique_release_lambdas) != 1
        ):
            raise NotImplementedError(
                "Currently all host orientations must use the same lambda paths."
            )

        attach_lambdas = [*next(iter(unique_attach_lambdas))]
        n_pull_windows = next(iter(unique_n_pull_windows))
        release_lambdas = [*next(iter(unique_release_lambdas))]

        lambda_scaling = cls._generate_lambda_scaling(
            attach_lambdas, n_pull_windows, release_lambdas
        )

        metadata.update(
            {
                "host_coordinate_path": next(iter(unique_host_structures)),
                "attach_windows_indices": [*range(len(attach_lambdas))],
                "attach_lambdas": attach_lambdas,
                "pull_windows_indices": [*range(n_pull_windows)],
                "n_pull_windows": n_pull_windows,
                "release_windows_indices": [*range(len(attach_lambdas))],
                "release_lambdas": release_lambdas,
                "bound_window_index": [[*range(n_pull_windows)][0]],
                "unbound_window_index": [[*range(n_pull_windows)][-1]],
                "lambda_scaling": lambda_scaling,
            }
        )

        return metadata

    def _initialize(
        self,
        host_codes: List[str],
        guest_codes: List[str],
        host_guest_codes: Dict[str, List[str]],
        exclude_systems: Dict[str, List[str]],
        ionic_strength: Optional[unit.Quantity],
        negative_buffer_ion: str,
        positive_buffer_ion: str,
        in_vacuum: bool,
        attach_apr_meta_data: bool,
    ):
        """Initializes the data set from the data made available by taproom.

        Parameters
        ----------
        host_codes
            The three letter codes of the host molecules to load from ``taproom``
            If no list is provided, all hosts will be loaded.
        guest_codes
            The three letter codes of the guest molecules to load from ``taproom``.
            If no list is provided, all guests will be loaded.
        host_guest_codes
            A dictionary containing the three letter codes of guest molecules for
            each host molecule to load from ``taproom``. This option provides greater
            control for choosing the host-guest systems. If specified, host_codes and
            guest_codes will be ignored.
        exclude_systems
            A dictionary for host-guest systems to exclude from the list generated
            from the lists of host_codes and guest_codes.
        ionic_strength
            The default ionic strength to use for measurements. The value
            specified in ``taproom`` will be ignored and this value used
            instead.
        negative_buffer_ion
            The SMILES pattern of the negative buffer ion to use. The value
            specified in ``taproom`` will be ignored and this value used
            instead.
        positive_buffer_ion
            The SMILES pattern of the positive buffer ion to use. The value
            specified in ``taproom`` will be ignored and this value used
            instead.
        in_vacuum
            Whether to configure Taproom without solvents.
        attach_apr_meta_data
            Whether to add the metadata required for an APR based calculation
            using the ``paprika`` based workflow.
        """

        installed_benchmarks = {}

        for entry_point in pkg_resources.iter_entry_points(group="taproom.benchmarks"):
            installed_benchmarks[entry_point.name] = entry_point.load()

        if len(installed_benchmarks) == 0:

            raise ValueError(
                "No installed benchmarks could be found. Make sure the "
                "`host-guest-benchmarks` package is installed."
            )

        measurements = installed_benchmarks["host_guest_measurements"]
        systems = installed_benchmarks["host_guest_systems"]

        if host_guest_codes:
            host_codes = list(host_guest_codes.keys())

        all_properties = []

        for host_name in measurements:

            if host_codes and host_name not in host_codes:
                continue

            if host_guest_codes:
                guest_codes = host_guest_codes[host_name]

            for guest_name in measurements[host_name]:

                if guest_codes and guest_name not in guest_codes:
                    continue

                # Make sure this measurement has a corresponding system
                if host_name not in systems or guest_name not in systems[host_name]:
                    continue

                # Exclude systems in specified
                if (
                    exclude_systems
                    and host_name in exclude_systems
                    and guest_name in exclude_systems[host_name]
                ):
                    continue

                measurement_path = measurements[host_name][guest_name]["yaml"]

                with open(measurement_path, "r") as file:
                    measurement_yaml = yaml.safe_load(file)

                temperature = unit.Quantity(measurement_yaml["state"]["temperature"])
                pressure = unit.Quantity(measurement_yaml["state"]["pressure"])

                value = unit.Quantity(measurement_yaml["measurement"]["delta_G"])
                uncertainty = unit.Quantity(
                    measurement_yaml["measurement"]["delta_G_uncertainty"]
                )

                source = TaproomSource(
                    doi=measurement_yaml["provenance"]["doi"],
                    comment=measurement_yaml["provenance"]["comment"],
                    technique=measurement_yaml["measurement"]["technique"],
                    host_identifier=host_name,
                    guest_identifier=guest_name,
                )

                orientations = [
                    orientation for orientation in systems[host_name]["yaml"]
                ]
                host_yaml_path = systems[host_name]["yaml"][orientations[0]]

                with open(host_yaml_path, "r") as file:
                    host_yaml = yaml.safe_load(file)

                host_mol2_path = str(
                    host_yaml_path.parent.joinpath(host_yaml["structure"])
                )
                host_monomer_path = None
                if "monomer" in host_yaml:
                    host_monomer_path = str(
                        host_yaml_path.parent.joinpath(host_yaml["monomer"])
                    )
                host_smiles = self._mol2_to_smiles(host_mol2_path)

                guest_yaml_path = systems[host_name][guest_name]["yaml"]

                host_tleap_template = str(
                    systems[host_name]["path"].joinpath(f"build_{host_name}.in")
                )

                with open(guest_yaml_path, "r") as file:
                    guest_yaml = yaml.safe_load(file)

                guest_mol2_path = str(
                    host_yaml_path.parent.joinpath(guest_name).joinpath(
                        guest_yaml["structure"]
                    )
                )
                guest_smiles = self._mol2_to_smiles(guest_mol2_path)
                logger.info(guest_smiles)
                substance = self._build_substance(
                    guest_smiles,
                    host_smiles,
                    ionic_strength,
                    negative_buffer_ion,
                    positive_buffer_ion,
                    in_vacuum,
                )
                host_only_substance = self._build_substance(
                    None,
                    host_smiles,
                    ionic_strength,
                    negative_buffer_ion,
                    positive_buffer_ion,
                    in_vacuum,
                )

                measured_property = HostGuestBindingAffinity(
                    thermodynamic_state=ThermodynamicState(temperature, pressure),
                    phase=PropertyPhase.Liquid,
                    substance=substance,
                    value=value,
                    uncertainty=uncertainty,
                    source=source,
                )

                if attach_apr_meta_data:

                    measured_property.metadata = self._build_metadata(
                        systems[host_name]["yaml"],
                        systems[host_name][guest_name]["yaml"],
                        host_only_substance,
                    )
                    measured_property.metadata["host_file_paths"].update(
                        {
                            "host_mol2_path": host_mol2_path,
                            "host_monomer_path": host_monomer_path,
                            "host_tleap_template": host_tleap_template,
                        }
                    )
                    measured_property.metadata["guest_file_paths"].update(
                        {
                            "guest_mol2_path": guest_mol2_path,
                        }
                    )

                all_properties.append(measured_property)

        self.add_properties(*all_properties)
