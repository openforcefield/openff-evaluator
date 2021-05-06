"""
A collection of wrappers around commonly employed force fields.
"""
import abc
import json
import os
import re
import shutil
import tempfile
import typing
from enum import Enum

import numpy as np
import parmed as pmd
from openff.toolkit.topology import Molecule
from simtk import unit as simtk_unit
from simtk.openmm.app import AmberPrmtopFile
from simtk.openmm.app import element as E

from openff.evaluator import unit
from openff.evaluator.utils.serialization import TypedBaseModel
from openff.evaluator.utils.utils import is_number


class ForceFieldSource(TypedBaseModel):
    """A helper object to define the source of a force field
    and any associated meta data, such as version, file paths,
    or generation options.

    Notes
    -----
    It is likely that this class and classes based off of it will
    not be permanent fixtures of the framework, but rather will
    exist until the force fields can be stored in a uniform format /
    object model.
    """

    @abc.abstractmethod
    def __getstate__(self):
        pass

    @abc.abstractmethod
    def __setstate__(self, state):
        pass


class SmirnoffForceFieldSource(ForceFieldSource):
    """A wrapper around force fields based on the
    `SMIRks Native Open Force Field (SMIRNOFF) specification
    <https://open-forcefield-toolkit.readthedocs.io/en/latest/smirnoff.html>`_.
    """

    def __init__(self, inner_xml=None):
        """Constructs a new SmirnoffForceFieldSource object

        Parameters
        ----------
        inner_xml: str, optional
            A string containing the xml representation of the force field.
        """
        self._inner_xml = inner_xml

    def to_force_field(self):
        """Returns the SMIRNOFF force field created from this source.

        Returns
        -------
        openff.toolkit.typing.engines.smirnoff.ForceField
            The created force field.
        """
        from openff.toolkit.typing.engines import smirnoff

        return smirnoff.ForceField(self._inner_xml)

    @classmethod
    def from_object(cls, force_field):
        """Creates a new `SmirnoffForceFieldSource` from an existing
        `ForceField` object

        Notes
        -----
        All cosmetic attributes will be discarded.

        Parameters
        ----------
        force_field: openff.toolkit.typing.engines.smirnoff.ForceField
            The existing force field.

        Returns
        -------
        SmirnoffForceFieldSource
            The created object.
        """

        return cls(force_field.to_string("XML", True))

    @classmethod
    def from_path(cls, file_path):
        """Creates a new `SmirnoffForceFieldSource` from the file path to a
        `ForceField` object.

        Notes
        -----
        All cosmetic attributes will be discarded.

        Parameters
        ----------
        file_path: str
            The file path to the force field object. This may also be the
            name of a file which can be loaded via an entry point.

        Returns
        -------
        SmirnoffForceFieldSource
            The created object.
        """

        from openff.toolkit.typing.engines.smirnoff import ForceField

        force_field = ForceField(file_path, allow_cosmetic_attributes=True)
        return cls.from_object(force_field)

    def __getstate__(self):
        return {"inner_xml": self._inner_xml}

    def __setstate__(self, state):
        self._inner_xml = state["inner_xml"]


class TLeapForceFieldSource(ForceFieldSource):
    """A wrapper around Amber force fields which may be
    applied via the `tleap` software package.

    Notes
    -----
    Currently this only supports force fields which are installed
    alongside `tleap`.
    """

    @property
    def leap_source(self):
        """list of str: The parameter file which should be sourced by `leap`
        when applying the force field.
        """
        return self._leap_source

    @property
    def cutoff(self):
        """openff.evaluator.unit.Quantity: The non-bonded interaction cutoff."""
        return self._cutoff

    @property
    def custom_frcmod(self):
        """dict: A dictionary containing frcmod parameters."""
        return self._custom_frcmod

    @custom_frcmod.setter
    def custom_frcmod(self, value: typing.Union[str, dict]):
        if isinstance(value, str):
            self._custom_frcmod = GAFFForceField.frcmod_file_to_dict(value)

        elif isinstance(value, dict):
            self._custom_frcmod = value

        else:
            raise KeyError(
                "Input value must either be a string filename or dictionary."
            )

    @property
    def igb(self):
        """int: The Amber Generalized Born Implicit Solvent (GBIS) model."""
        return self._igb

    @property
    def sa_model(self):
        """str: The surface area (SA) model for running GB/SA simulation."""
        return self._sa_model

    def __init__(
        self,
        leap_source="leaprc.gaff2",
        cutoff=9.0 * unit.angstrom,
        frcmod_file=None,
        igb=None,
        sa_model="ACE",
    ):
        """Constructs a new TLeapForceFieldSource object

        Parameters
        ----------
        leap_source: str
            The parameter file which should be sourced by `leap`
            when applying the force field. Currently only
            `'leaprc.gaff'` and `'leaprc.gaff2'` are supported.
        cutoff: openff.evaluator.unit.Quantity
            The non-bonded interaction cutoff.
        frcmod_file: str
            frcmod file for custom parameters.
        igb: int
            Generalized Born implicit solvent model based on Amber
            numbering. Allowed values are 1, 2, and 5.
        sa_model: str
            The surface area (SA) model when running GB/SA simulation.
            Can either be None or 'ACE' (analytical continuum
            electrostatics) based on the implementation in OpenMM).

        Examples
        --------
        To create a source for the GAFF force field with tip3p water:

        >>> amber_gaff_source = TLeapForceFieldSource('leaprc.gaff')

        To create a source for the GAFF2 force field with tip3p water:

        >>> amber_gaff_2_source = TLeapForceFieldSource('leaprc.gaff2')

        To create a source for the GAFF force field with the HCT GBIS model:

        >>> amber_gaff_gbis_source = TLeapForceFieldSource('leaprc.gaff', igb=1)

        To create a source for GAFF with a modified parameter:

        >>> amber_gaff_modified = TLeapForceFieldSource('leaprc.gaff', frcmod_file='custom.frcmod')
        """

        if leap_source is not None:
            assert leap_source == "leaprc.gaff2" or leap_source == "leaprc.gaff"

        if igb is not None:
            assert igb in [1, 2, 5]

        if sa_model is not None:
            assert sa_model == "ACE"

        self._leap_source = leap_source
        self._cutoff = cutoff
        self._custom_frcmod = (
            None
            if frcmod_file is None
            else GAFFForceField.frcmod_file_to_dict(frcmod_file)
        )
        self._igb = igb
        self._sa_model = sa_model

    @classmethod
    def from_object(cls, force_field):
        """Instantiate class from a GAFFForceField object."""

        if not isinstance(force_field, GAFFForceField):
            raise TypeError(
                "Only `GAFFForceField` is compatible with `TLeapForceField`."
            )

        new_instance = cls(
            leap_source=f"leaprc.{force_field.gaff_version}",
            cutoff=force_field.cutoff,
            igb=int(force_field.igb),
            sa_model=force_field.sa_model,
        )
        if force_field.frcmod_parameters:
            new_instance.custom_frcmod = force_field.frcmod_parameters.copy()

        return new_instance

    def __getstate__(self):
        return {
            "leap_source": self._leap_source,
            "cutoff": self._cutoff,
            "custom_frcmod": self._custom_frcmod,
            "igb": self._igb,
            "sa_model": self._sa_model,
        }

    def __setstate__(self, state):
        self._leap_source = state["leap_source"]
        self._cutoff = state["cutoff"]
        self._custom_frcmod = state["custom_frcmod"]
        self._igb = state["igb"]
        self._sa_model = state["sa_model"]


class LigParGenForceFieldSource(ForceFieldSource):
    """A wrapper and the OPLSAAM force field which can be applied
    via the `LigParGen server <http://zarbi.chem.yale.edu/ligpargen/>`_.

    References
    ----------
    [1] Potential energy functions for atomic-level simulations of water and organic and
        biomolecular systems. Jorgensen, W. L.; Tirado-Rives, J. Proc. Nat. Acad. Sci.
        USA 2005, 102, 6665-6670
    [2] 1.14*CM1A-LBCC: Localized Bond-Charge Corrected CM1A Charges for Condensed-Phase
        Simulations. Dodda, L. S.; Vilseck, J. Z.; Tirado-Rives, J.; Jorgensen, W. L.
        J. Phys. Chem. B, 2017, 121 (15), pp 3864-3870
    [3] LigParGen web server: An automatic OPLS-AA parameter generator for organic ligands.
        Dodda, L. S.;Cabeza de Vaca, I.; Tirado-Rives, J.; Jorgensen, W. L.
        Nucleic Acids Research, Volume 45, Issue W1, 3 July 2017, Pages W331-W336
    """

    class ChargeModel(Enum):

        CM1A_1_14_LBCC = "1.14*CM1A-LBCC"
        CM1A_1_14 = "1.14*CM1A"

    @property
    def preferred_charge_model(self):
        """ChargeModel: The preferred charge model to apply. In some cases
        the preferred charge model may not be applicable (e.g. 1.14*CM1A-LBCC
        may only be applied to neutral molecules) and so another model may be
        applied in its place.
        """
        return self._preferred_charge_model

    @property
    def cutoff(self):
        """openff.evaluator.unit.Quantity: The non-bonded interaction cutoff."""
        return self._cutoff

    @property
    def request_url(self):
        """str: The URL of the LIGPARGEN server file to send the parametrization to request to."""
        return self._request_url

    @property
    def download_url(self):
        """str: The URL of the LIGPARGEN server file to download the results of a request from."""
        return self._download_url

    def __init__(
        self,
        preferred_charge_model=ChargeModel.CM1A_1_14_LBCC,
        cutoff=9.0 * unit.angstrom,
        request_url="",
        download_url="",
    ):
        """Constructs a new LigParGenForceFieldSource object

        Parameters
        ----------
        preferred_charge_model: ChargeModel
            The preferred charge model to apply. In some cases
            the preferred charge model may not be applicable
            (e.g. 1.14*CM1A-LBCC may only be applied to neutral
            molecules) and so another model may be applied in its
            place.
        cutoff: openff.evaluator.unit.Quantity
            The non-bonded interaction cutoff.
        request_url: str
            The URL of the LIGPARGEN server file to send the parametrization to request to.
        download_url: str
            The URL of the LIGPARGEN server file to download the results of a request from.
        """
        self._preferred_charge_model = preferred_charge_model
        self._cutoff = cutoff

        self._request_url = request_url
        self._download_url = download_url

    def __getstate__(self):
        return {
            "preferred_charge_model": self._preferred_charge_model,
            "cutoff": self._cutoff,
            "request_url": self._request_url,
            "download_url": self._download_url,
        }

    def __setstate__(self, state):
        self._preferred_charge_model = state["preferred_charge_model"]
        self._cutoff = state["cutoff"]
        self._request_url = state["request_url"]
        self._download_url = state["download_url"]


class GAFFForceField:
    # TODO: add support for parsing MOL2 file for charge/electrostatic.

    @property
    def smiles_list(self):
        """list: A list containing the smiles string of the system substances."""
        return self._smiles_list

    @property
    def gaff_version(self):
        """str: The version of GAFF to use (`gaff` or `gaff2`)."""
        return self._gaff_version

    @gaff_version.setter
    def gaff_version(self, value):
        self._gaff_version = value

    @property
    def cutoff(self):
        """openff.evaluator.unit.Quantity: The non-bonded interaction cutoff."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: unit.Quantity):
        self._cutoff = value

    @property
    def igb(self):
        """int: The Amber Generalized Born Implicit Solvent (GBIS) model."""
        return self._igb

    @igb.setter
    def igb(self, value):
        self._igb = value

    @property
    def sa_model(self):
        """str: The surface area (SA) model for running GB/SA simulation."""
        return self._sa_model

    @sa_model.setter
    def sa_model(self, value):
        self._sa_model = value

    @property
    def topology(self):
        """ParmEd.structure: The topology of the system as a ParmEd object."""
        return self._topology

    @property
    def frcmod_parameters(self):
        """dict: The `frcmod` parameters stored in a dictionary."""
        return self._frcmod_parameters

    @frcmod_parameters.setter
    def frcmod_parameters(self, value):
        self._frcmod_parameters = value

    def __init__(
        self,
        smiles_list=None,
        gaff_version="gaff",
        cutoff=9.0 * unit.angstrom,
        frcmod_parameters=None,
        igb=None,
        sa_model="ACE",
    ):
        self._gaff_version = gaff_version
        self._cutoff = cutoff
        self._igb = igb
        self._sa_model = sa_model
        self._smiles_list = smiles_list
        self._topology = None
        self._frcmod_parameters = (
            self.frcmod_file_to_dict(frcmod_parameters)
            if isinstance(frcmod_parameters, str)
            else frcmod_parameters
        )

        if smiles_list is not None:
            self._initialize()

    def _initialize(self):
        from paprika.build.system import TLeap
        from paprika.evaluator.amber import generate_gaff
        from simtk.openmm.app.internal.customgbforces import _get_bonded_atom_list

        # Extract GAFF parameters
        working_directory = tempfile.mkdtemp()
        molecule_list = []

        for i, smiles in enumerate(self._smiles_list):
            # Generate mol2 file
            molecule = Molecule.from_smiles(smiles)
            molecule.partial_charges = (
                np.zeros(molecule.n_atoms) * simtk_unit.elementary_charge
            )
            molecule.to_file(
                os.path.join(working_directory, f"MOL{i}.mol2"),
                file_format="MOL2",
            )

            generate_gaff(
                mol2_file=f"MOL{i}.mol2",
                residue_name=f"MOL{i}",
                output_name=f"MOL{i}",
                need_gaff_atom_types=True,
                generate_frcmod=False,
                gaff_version=self.gaff_version,
                directory_path=working_directory,
            )

            # Generate prmtop file
            system = TLeap()
            system.output_path = working_directory
            system.output_prefix = f"MOL{i}.{self.gaff_version}"
            system.pbc_type = None
            system.neutralize = False
            system.template_lines = [
                f"source leaprc.{self.gaff_version}",
                f"MOL{i} = loadmol2 MOL{i}.{self.gaff_version}.mol2",
                f"saveamberparm MOL{i} {system.output_prefix}.prmtop {system.output_prefix}.rst7",
                "quit",
            ]
            system.build(clean_files=False, ignore_warnings=True)

            molecule_list.append(
                os.path.join(working_directory, f"{system.output_prefix}.prmtop")
            )

        # Generate OpenMM topology
        topology = pmd.load_file(molecule_list[0], structure=True)
        for molecule in molecule_list[1:]:
            topology += pmd.load_file(molecule, structure=True)
        topology.save(os.path.join(working_directory, "full.prmtop"), overwrite=True)
        self._topology = AmberPrmtopFile(os.path.join(working_directory, "full.prmtop"))

        # Generate full frcmod file
        pmd.tools.writeFrcmod(
            topology,
            os.path.join(working_directory, "complex.frcmod"),
        ).execute()
        self._frcmod_parameters = GAFFForceField.frcmod_file_to_dict(
            os.path.join(working_directory, "complex.frcmod")
        )

        # Delete temp folder
        shutil.rmtree(working_directory)

        if self.igb:
            all_bonds = _get_bonded_atom_list(self._topology.topology)

            # Apply `mbondi` radii (igb=1)
            if self.igb == 1:
                default_radius = 1.5
                element_to_const_radius = {
                    E.nitrogen: 1.55,
                    E.oxygen: 1.5,
                    E.fluorine: 1.5,
                    E.silicon: 2.1,
                    E.phosphorus: 1.85,
                    E.sulfur: 1.8,
                    E.chlorine: 1.7,
                }

                for atom in self._topology.topology.atoms():
                    element = atom.element

                    # Radius of H atom depends on element it is bonded to
                    if element in (E.hydrogen, E.deuterium):
                        bondeds = all_bonds[atom]
                        if bondeds[0].element in (E.carbon, E.nitrogen):
                            radii = 1.3
                            mask = "H-C" if bondeds[0].element is E.carbon else "H-N"
                        elif bondeds[0].element in (E.oxygen, E.sulfur):
                            radii = 0.8
                            mask = "H-O" if bondeds[0].element is E.oxygen else "H-S"
                        else:
                            radii = 1.2
                            mask = "H"

                    # Radius of C atom depends on what type it is
                    elif element is E.carbon:
                        radii = 1.7
                        mask = "C"

                    # All other elements have fixed radii
                    else:
                        radii = element_to_const_radius.get(element, default_radius)
                        mask = element.symbol

                    # Store radii into dictionary
                    if mask not in self._frcmod_parameters["GBSA"]:
                        self._frcmod_parameters["GBSA"].update(
                            {
                                mask: {
                                    "radius": radii / 10,
                                    "cosmetic": None,
                                }
                            }
                        )

            # Apply `mbondi2` radii (igb=2,5)
            elif self.igb in [2, 5]:
                default_radius = 1.5
                element_to_const_radius = {
                    E.nitrogen: 1.55,
                    E.oxygen: 1.5,
                    E.fluorine: 1.5,
                    E.silicon: 2.1,
                    E.phosphorus: 1.85,
                    E.sulfur: 1.8,
                    E.chlorine: 1.7,
                }

                for atom in self._topology.topology.atoms():
                    element = atom.element

                    # Radius of H atom depends on element it is bonded to
                    if element in (E.hydrogen, E.deuterium):
                        bondeds = all_bonds[atom]
                        if bondeds[0].element is E.nitrogen:
                            radii = 1.3
                            mask = "H-N"
                        else:
                            radii = 1.2
                            mask = "H"

                    # Radius of C atom depeends on what type it is
                    elif element is E.carbon:
                        radii = 1.7
                        mask = "C"

                    # All other elements have fixed radii
                    else:
                        radii = element_to_const_radius.get(element, default_radius)
                        mask = element.symbol

                    # Store radii into dictionary
                    if mask not in self._frcmod_parameters["GBSA"]:
                        self._frcmod_parameters["GBSA"].update(
                            {
                                mask: {
                                    "radius": radii / 10,
                                    "cosmetic": None,
                                }
                            }
                        )

    def get_parameter_value(self, tag, atom_mask, *attributes):
        """Returns an FF parameter(s) as a dictionary. Multiple parameters
        can be returned for a specific tag.

        Parameters
        ----------
        tag: str
           FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        attributes: str
            The attribute for the parameter (e.g., "rmin_half", "epsilon" for vdW).

        Returns
        -------
        parameter: dict
            A dictionary with the FF parameter.
        """
        if tag not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tag}` does not exist in the parameter list.")
        if atom_mask not in self._frcmod_parameters[tag]:
            raise KeyError(f"The atom mask `{atom_mask}` is not listed under `{tag}`.")

        parameter = {tag: {atom_mask: {}}}
        for attribute in attributes:
            if attribute in self._frcmod_parameters[tag][atom_mask]:
                parameter[tag][atom_mask].update(
                    {attribute: self._frcmod_parameters[tag][atom_mask][attribute]}
                )
            else:
                raise KeyError(
                    f"`{attribute}` is not an attribute of `{tag}-{atom_mask}`."
                )

        return parameter

    def set_parameter_value(self, tag, atom_mask, attribute, value):
        """Set the value for a FF parameter.

        Parameters
        ----------
        tag: str
            FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        attribute: str
            The attribute for the parameter (e.g., "rmin_half", "epsilon" for vdW).
        value: float
            The value for the FF parameter.
        """
        if tag not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tag}` does not exist in the parameter list.")

        if atom_mask not in self._frcmod_parameters[tag]:
            raise KeyError(f"The atom mask `{atom_mask}` is listed under `{tag}`.")

        if attribute not in self._frcmod_parameters[tag][atom_mask]:
            raise KeyError(
                f"The attribute `{attribute}` is not an attribute of `{tag}-{atom_mask}`."
            )

        self._frcmod_parameters[tag][atom_mask][attribute] = value

    def tag_parameter_to_optimize(self, tag, atom_mask, *attributes):
        """Tag a FF parameter(s) for use in a ForceBalance run. When writing
        to file, the tagged FF parameter(s) will have a comment "# PRM ..."
        at then end of the line.

        Parameters
        ----------
        tag: str
            FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        attributes: str
            The attribute for the parameter (e.g., "rmin_half", "epsilon" for vdW).
        """
        if tag not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tag}` does not exist in the parameter list.")
        if atom_mask not in self._frcmod_parameters[tag]:
            raise KeyError(f"The atom mask `{atom_mask}` is listed under `{tag}`.")

        cosmetic = "# PRM"
        for attribute in attributes:
            if attribute in self._frcmod_parameters[tag][atom_mask]:

                if tag == "BOND":
                    if attribute == "k":
                        cosmetic += " 1"
                    elif attribute == "length":
                        cosmetic += " 2"

                elif tag == "ANGLE":
                    if attribute == "k":
                        cosmetic += " 1"
                    elif attribute == "angle":
                        cosmetic += " 2"

                elif tag == "DIHEDRAL":
                    if attribute == "scaling":
                        cosmetic += " 1"
                    elif attribute == "barrier":
                        cosmetic += " 2"
                    elif attribute == "phase":
                        cosmetic += " 3"
                    elif attribute == "periodicity":
                        cosmetic += " 4"

                elif tag == "IMPROPER":
                    if attribute == "barrier":
                        cosmetic += " 1"
                    elif attribute == "phase":
                        cosmetic += " 2"
                    elif attribute == "periodicity":
                        cosmetic += " 3"

                elif tag == "VDW":
                    if attribute == "rmin_half":
                        cosmetic += " 1"
                    elif attribute == "epsilon":
                        cosmetic += " 2"

                elif tag == "GBSA":
                    if attribute == "radius":
                        cosmetic += " 1"

            else:
                raise KeyError(
                    f"`{attribute}` is not an attribute of `{tag}-{atom_mask}`."
                )

        self._frcmod_parameters[tag][atom_mask]["cosmetic"] = cosmetic

    @staticmethod
    def _parameter_to_string(tag, atom_mask, parameters):
        """Convert a parameter to a string in AMBER frcmod file format.

        Parameters
        ----------
        tag: str
            FF parameter tag name (MASS, BOND, ANGLE, DIHEDRAL, IMPROPER, VDW, GBSA).
        atom_mask: str
            The GAFF atom type for the particular parameter (bonded params
            are separated with "-").
        parameters: dict
            A dictionary containing the FF attribute and parameters.

        Return
        ------
        parameter_line: str
            A string with the FF parameter in AMBER frcmod format (https://ambermd.org/FileFormats.php#frcmod).
        """
        parameter_line = None

        if tag == "MASS":
            parameter_line = f"{atom_mask:2s}"
            parameter_line += f"{parameters['mass']:10.3f}"

        if tag == "BOND":
            parameter_line = f"{atom_mask:5s}"
            parameter_line += f"{parameters['k']:10.3f}"
            parameter_line += f"{parameters['length']:10.3f}"

        if tag == "ANGLE":
            parameter_line = f"{atom_mask:8s}"
            parameter_line += f"{parameters['k']:10.3f}"
            parameter_line += f"{parameters['theta']:10.3f}"

        if tag == "DIHEDRAL":
            parameter_line = f"{atom_mask:11s}"
            parameter_line += f"{parameters['scaling']:4d}"
            parameter_line += f"{parameters['barrier']:15.8f}"
            parameter_line += f"{parameters['phase']:10.3f}"
            parameter_line += f"{parameters['periodicity']:10.2f}"
            if parameters["SCEE"]:
                parameter_line += f"  SCEE={parameters['SCEE']:.1f}"
            if parameters["SCNB"]:
                parameter_line += f"  SCNB={parameters['SCNB']:.1f}"

        if tag == "IMPROPER":
            parameter_line = f"{atom_mask:11s}"
            parameter_line += f"{parameters['barrier']:15.8f}"
            parameter_line += f"{parameters['phase']:10.3f}"
            parameter_line += f"{parameters['periodicity']:10.2f}"

        if tag == "VDW":
            parameter_line = f"{atom_mask:4s}"
            parameter_line += f"{parameters['rmin_half']:15.8f}"
            parameter_line += f"{parameters['epsilon']:15.8f}"

        if tag == "GBSA":
            parameter_line = f"{atom_mask:4s}"
            parameter_line += f"{parameters['radius']:15.8f}"

        if parameters["cosmetic"]:
            parameter_line += f"   {parameters['cosmetic']}"

        assert parameter_line is not None

        parameter_line += "\n"

        return parameter_line

    def to_file(self, file_path, write_header=False, skip_gbsa=True):
        """Write the FF parameters to an AMBER frcmod file.

        Parameters
        ----------
        file_path: str
            The name of the frcmod file.
        write_header: bool
            Whether to print header information (used in ForceBalance runs).
        skip_gbsa: bool
            Whether to skip printing FF parameters for GBSA (not read in TLeap but used in ForceBalance runs).
        """

        with open(file_path, "w") as f:

            for tag in self._frcmod_parameters.keys():
                if tag == "HEADER" and write_header:
                    f.writelines(
                        "#evaluator_io: "
                        f"gaff_version={self._gaff_version} "
                        f"cutoff={self._cutoff.magnitude} "
                        f"igb={self._igb} "
                        f"sa_model={self._sa_model} \n"
                    )
                    continue
                elif tag == "HEADER" and not write_header:
                    f.writelines("Remark line goes here\n")
                    continue

                if tag == "GBSA" and skip_gbsa:
                    continue

                if tag == "DIHEDRAL":
                    f.writelines("DIHE\n")
                elif tag == "VDW":
                    f.writelines("NONBON\n")
                else:
                    f.writelines(f"{tag}\n")

                for atom_mask in self._frcmod_parameters[tag]:
                    f.writelines(
                        self._parameter_to_string(
                            tag,
                            atom_mask,
                            self._frcmod_parameters[tag][atom_mask],
                        )
                    )
                f.writelines("\n")

    @classmethod
    def from_file(cls, file_path: str):
        """Create an instance of this class by reading in a frcmod file."""
        frcmod_pdict = cls.frcmod_file_to_dict(file_path)

        gaff_version = "gaff"
        cutoff = 9.0 * unit.angstrom
        igb = None
        sa_model = None

        if frcmod_pdict["HEADER"]:
            gaff_version = frcmod_pdict["HEADER"]["leap_source"]
            cutoff = frcmod_pdict["HEADER"]["cutoff"] * unit.angstrom
            igb = int(frcmod_pdict["HEADER"]["igb"])
            sa_model = (
                None
                if frcmod_pdict["HEADER"]["sa_model"] == "None"
                else frcmod_pdict["HEADER"]["sa_model"]
            )

        new_instance = cls(
            gaff_version=gaff_version,
            cutoff=cutoff,
            igb=igb,
            sa_model=sa_model,
        )
        new_instance.frcmod_parameters = frcmod_pdict

        return new_instance

    @staticmethod
    def frcmod_file_to_dict(file_path: str) -> dict:
        """Read in a frcmod file and stores the information in a dictionary.

        .. note ::
            Parameters with polarizabilities are not supported yet and will be ignored.

        Parameters
        ----------
        file_path: str
            The fcmod file to process.

        Returns
        -------
        frcmod_dict: dict
            A dictionary containing the parameters from the frcmod file.
        """

        frcmod_dict = {
            "HEADER": {},
            "MASS": {},
            "BOND": {},
            "ANGLE": {},
            "DIHEDRAL": {},
            "IMPROPER": {},
            "VDW": {},
            "GBSA": {},
        }

        with open(file_path, "r") as f:

            for i, line in enumerate(f.readlines()):

                if i == 0 and line.startswith("#evaluator_io:"):
                    header = line.split()
                    frcmod_dict["HEADER"] = {
                        "leap_source": header[1].split("=")[-1],
                        "cutoff": float(header[2].split("=")[-1]),
                        "igb": int(header[3].split("=")[-1]),
                        "sa_model": header[4].split("=")[-1],
                    }
                    continue

                if (
                    (i == 0 and not line.startswith("#evaluator:"))
                    or line.strip() == 0
                    or line.startswith("\n")
                ):
                    continue

                if re.match("MASS", line.strip().upper()):
                    keyword = "MASS"
                    continue
                elif re.match("BOND|BONDS", line.strip().upper()):
                    keyword = "BOND"
                    continue
                elif re.match("ANGLE|ANGLES", line.strip().upper()):
                    keyword = "ANGLE"
                    continue
                elif re.match("DIHE|DIHEDRAL|DIHEDRALS", line.strip().upper()):
                    keyword = "DIHEDRAL"
                    continue
                elif re.match("IMPROPER", line.strip().upper()):
                    keyword = "IMPROPER"
                    continue
                elif re.match("NONBON|NONB|NONBONDED", line.strip().upper()):
                    keyword = "VDW"
                    continue
                elif re.match("RADII|GBSA|GBRADII", line.strip().upper()):
                    keyword = "GBSA"
                    continue

                # Read parameter
                cosmetic = None
                parameter = line.split()
                if "#" in line:
                    parameter = line[: line.index("#")].split()
                    cosmetic = line[line.index("#") :]

                atom_columns = []
                for j in range(len(parameter)):
                    # Convert to float
                    if is_number(parameter[j]) and not parameter[j].isdigit():
                        parameter[j] = float(parameter[j])

                    # Convert to int
                    elif is_number(parameter[j]) and parameter[j].isdigit():
                        parameter[j] = int(parameter[j])

                    # Get list element that are strings
                    elif "SC" not in parameter[j]:
                        atom_columns.append(j)

                # Get proper formatting for atom masks
                mask = parameter[0]
                if len(atom_columns) > 1:
                    atom_mask = "".join(parameter[: len(atom_columns)])
                    for k, col in enumerate(atom_columns):
                        parameter.remove(parameter[col - k])
                    mask = "-".join(f"{atom:2s}" for atom in atom_mask.split("-"))
                else:
                    parameter.pop(0)

                # Build parameter dictionary
                if keyword == "MASS":
                    param_dict = {"mass": parameter[0], "cosmetic": cosmetic}

                elif keyword == "BOND":
                    param_dict = {
                        "k": parameter[0],
                        "length": parameter[1],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "ANGLE":
                    param_dict = {
                        "k": parameter[0],
                        "theta": parameter[1],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "DIHEDRAL":
                    param_dict = {
                        "scaling": parameter[0],
                        "barrier": parameter[1],
                        "phase": parameter[2],
                        "periodicity": parameter[3],
                        "SCEE": None,
                        "SCNB": None,
                        "cosmetic": cosmetic,
                    }
                    if len(parameter) > 4:
                        if "SCEE" in parameter[4]:
                            param_dict["SCEE"] = float(parameter[4].split("=")[1])
                        if "SCNB" in parameter[4]:
                            param_dict["SCNB"] = float(parameter[4].split("=")[1])

                    if len(parameter) > 5:
                        if "SCEE" in parameter[5]:
                            param_dict["SCEE"] = float(parameter[5].split("=")[1])
                        if "SCNB" in parameter[5]:
                            param_dict["SCNB"] = float(parameter[5].split("=")[1])

                elif keyword == "IMPROPER":
                    param_dict = {
                        "barrier": parameter[0],
                        "phase": parameter[1],
                        "periodicity": parameter[2],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "VDW":
                    param_dict = {
                        "rmin_half": parameter[0],
                        "epsilon": parameter[1],
                        "cosmetic": cosmetic,
                    }

                elif keyword == "GBSA":
                    param_dict = {"radius": parameter[0], "cosmetic": cosmetic}

                # Update dictionary
                frcmod_dict[keyword].update({mask: param_dict})

        return frcmod_dict

    def to_json(self, file_path: str):
        """Save current FF parameters to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self._frcmod_parameters, f)

    @classmethod
    def from_json(cls, file_path: str):
        """Create an instance of this class by reading in a JSON file."""
        with open(file_path, "r") as f:
            frcmod_pdict = json.load(f)

        gaff_version = "gaff"
        cutoff = 9.0 * unit.angstrom
        igb = None
        sa_model = None

        if frcmod_pdict["HEADER"]:
            gaff_version = frcmod_pdict["HEADER"]["leap_source"]
            cutoff = frcmod_pdict["HEADER"]["cutoff"] * unit.angstrom
            igb = int(frcmod_pdict["HEADER"]["igb"])
            sa_model = (
                None
                if frcmod_pdict["HEADER"]["sa_model"] == "None"
                else frcmod_pdict["HEADER"]["sa_model"]
            )

        new_instance = cls(
            gaff_version=gaff_version,
            cutoff=cutoff,
            igb=igb,
            sa_model=sa_model,
        )
        new_instance.frcmod_parameters = frcmod_pdict

        return new_instance
