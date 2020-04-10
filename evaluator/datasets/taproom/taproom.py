"""
An API for importing a data set from the host-guest-benchmarks repository:
https://github.com/slochower/host-guest-benchmarks
"""
import logging

import pkg_resources
import yaml

from evaluator import unit
from evaluator.datasets import PhysicalPropertyDataSet, PropertyPhase, Source
from evaluator.properties import HostGuestBindingAffinity
from evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from evaluator.thermodynamics import ThermodynamicState

logger = logging.getLogger(__name__)


class TaproomSource(Source):
    """Contains any metadata about how a host-guest binding affinity
    was measured by experiment.
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


class TaproomDataSet(PhysicalPropertyDataSet):
    """A dataset of host-guest binding affinity measurements which sources
    its data from the `host-guest-benchmarks <https://github.com/slochower/
    host-guest-benchmarks>`_ repository.
    """

    def __init__(self, default_ionic_strength=150 * unit.millimolar):
        """Constructs a new TaproomDataSet object.

        Parameters
        ----------
        default_ionic_strength: unit.Quantity
            The default ionic strength to use. This is only temporary until
            a better solution can be found.
        """
        super().__init__()
        self._initialize(default_ionic_strength)

    @staticmethod
    def _mol2_to_smiles(file_path):
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
        from openforcefield.topology import Molecule

        receptor_molecule = Molecule.from_file(file_path, "MOL2")
        return receptor_molecule.to_smiles()

    @staticmethod
    def _build_substance(guest_smiles, host_smiles, ionic_strength=None):
        """Builds a substance containing a ligand and receptor solvated
        in an aqueous solution with a given ionic strength

        Parameters
        ----------
        guest_smiles: str, optional
            The smiles descriptor of the guest.
        host_smiles: str
            The smiles descriptor of the host.
        ionic_strength: pint.Quantity, optional
            The ionic strength of the aqueous solvent.

        Returns
        -------
        Substance
            The built substance.
        """
        from openforcefield.topology import Molecule

        substance = Substance()

        if guest_smiles is not None:

            guest = Component(smiles=guest_smiles, role=Component.Role.Ligand)
            substance.add_component(component=guest, amount=ExactAmount(1))

        host = Component(smiles=host_smiles, role=Component.Role.Receptor)
        substance.add_component(component=host, amount=ExactAmount(1))

        water = Component(smiles="O", role=Component.Role.Solvent)
        sodium = Component(smiles="[Na+]", role=Component.Role.Solvent)
        chlorine = Component(smiles="[Cl-]", role=Component.Role.Solvent)

        water_mole_fraction = 1.0

        if ionic_strength is not None:

            salt_mole_fraction = Substance.calculate_aqueous_ionic_mole_fraction(
                ionic_strength
            )
            water_mole_fraction = 1.0 - salt_mole_fraction

            substance.add_component(
                component=sodium, amount=MoleFraction(salt_mole_fraction / 2.0),
            )
            substance.add_component(
                component=chlorine, amount=MoleFraction(salt_mole_fraction / 2.0),
            )

        substance.add_component(
            component=water, amount=MoleFraction(water_mole_fraction)
        )

        host_molecule_charge = Molecule.from_smiles(host_smiles).total_charge
        guest_molecule_charge = (
            0.0
            if guest_smiles is None
            else Molecule.from_smiles(guest_smiles).total_charge
        )

        net_charge = host_molecule_charge + guest_molecule_charge
        counterions_needed = abs(int(net_charge))

        if net_charge <= -0.9999:
            substance.add_component(sodium, ExactAmount(counterions_needed))
        elif net_charge >= 0.9999:
            substance.add_component(chlorine, ExactAmount(counterions_needed))

        return substance

    def _initialize(self, ionic_strength):
        """Initializes the data set from the data made available by taproom.

        Parameters
        ----------
        ionic_strength: unit.Quantity
            The ionic strength to use. This is only temporary until
            a better solution can be found.
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

        all_properties = []

        for host_name in measurements:

            for guest_name in measurements[host_name]:

                # Make sure this measurement has a corresponding system
                if host_name not in systems or guest_name not in systems[host_name]:
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

                if value.units == unit.dimensionless:
                    logger.info(
                        f"The measurement for {host_name}-{guest_name} has no units and "
                        f"so will be skipped."
                    )
                    continue

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
                host_smiles = self._mol2_to_smiles(host_mol2_path)

                guest_yaml_path = systems[host_name][guest_name]["yaml"]

                with open(guest_yaml_path, "r") as file:
                    guest_yaml = yaml.safe_load(file)

                guest_mol2_path = str(
                    host_yaml_path.parent.joinpath(guest_name).joinpath(
                        guest_yaml["structure"]
                    )
                )

                guest_smiles = self._mol2_to_smiles(guest_mol2_path)

                # TODO: Don't hard code the ionic strength. Is there a way to determine this
                #       from the specified buffer?
                substance = self._build_substance(
                    guest_smiles, host_smiles, ionic_strength=ionic_strength
                )

                measured_property = HostGuestBindingAffinity(
                    thermodynamic_state=ThermodynamicState(temperature, pressure),
                    phase=PropertyPhase.Liquid,
                    substance=substance,
                    value=value,
                    uncertainty=uncertainty,
                    source=source,
                )

                measured_property.metadata = {
                    "guest_orientations": orientations,
                    "host_identifier": host_name,
                    "guest_identifier": guest_name,
                }

                all_properties.append(measured_property)

        self.add_properties(*all_properties)

    def filter_by_host_identifiers(self, *host_identifiers):
        """Filters out those properties which were measured for
         a host not specified in `host_identifiers`

        Parameters
        ----------
        host_identifiers: str
            The three letter identifier of the host.
        """

        def filter_function(physical_property):
            return physical_property.metadata["host_identifier"] in host_identifiers

        self.filter_by_function(filter_function)

    def filter_by_guest_identifiers(self, *guest_identifiers):
        """Filters out those properties which were measured for
         a guest not specified in `guest_identifiers`

        Parameters
        ----------
        guest_identifiers: str
            The three letter identifier of the guest.
        """

        def filter_function(physical_property):
            return physical_property.metadata["guest_identifier"] in guest_identifiers

        self.filter_by_function(filter_function)
