ThermoML Data Sets
==================

The ``ThermoMLDataset`` object offers an API for extracting physical properties from the `NIST ThermoML Archive
<http://trc.nist.gov/ThermoML.html>`_, both directly from the archive itself or from files stored in the IUPAC-
standard `ThermoML <http://trc.nist.gov/ThermoMLRecommendations.pdf>`_ format.

The API only supports extracting those properties which have been registered with the frameworks plug-in system,
and does not currently load the full set of metadata available in the archive files.

*If the metadata you require is currently not exposed, please open an issue on the* `GitHub issue tracker <https://
github.com/openforcefield/propertyestimator/issues>`_ *to request it.*

Currently the framework has built-in support for extracting:

* *Mass density, kg/m3* (``Density``)
* *Excess molar volume, m3/mol* (``ExcessMolarVolume``)
* *Relative permittivity at zero frequency* (``DielectricConstant``)
* *Excess molar enthalpy (molar enthalpy of mixing), kJ/mol* (``EnthalpyOfMixing``)
* *Molar enthalpy of vaporization or sublimation, kJ/mol* (``EnthalpyOfVaporization``)

where here both the ThermoML property name (as defined by the `IUPAC XML schema <https://trc.nist.gov/ThermoML.xsd>`_)
and the built-in framework class are listed.

Registering Properties
----------------------

Properties to be extracted from ThermoML archives must have a corresponding class representation to be loading into.
This class representation must both:

* inherit from the frameworks ``PhysicalProperty`` class and
* be registered with the frameworks plug-in system using either the ``thermoml_property`` decorator or the
  ``register_thermoml_property`` method.

As an example, a class representation of the ThermoML *'Mass density, kg/m3'* property could be defined and registered
with the plug-in system using::

    @thermoml_property("Mass density, kg/m3", supported_phases=PropertyPhase.Liquid)
    class Density(PhysicalProperty):
        """A class representation of a mass density property"""

The ``thermoml_property`` decorator takes in the name of the ThermoML property (as defined by the `IUPAC schema <https:
//trc.nist.gov/ThermoML.xsd>`_) as well as the phases which the framework will be able to estimate this property in.

Multiple ThermoML properties can be mapped onto a single class using the flexible ``register_thermoml_property``
function. As an example, the *'Specific volume, m3/kg'* property (which is simply the reciprocal of mass density) may
be mapped onto the ``Density`` by providing a ``conversion_function``::

    def specific_volume_to_mass_density(specific_volume):
        """Converts a specific volume measurement into a mass
        density.

        Parameters
        ----------
        specific_volume: ThermoMLProperty
            The specific volume measurement to convert.
        """
        mass_density = Density()

        mass_density.value = 1.0 / specific_volume.value

        if mass_density.uncertainty is not None:
            mass_density.uncertainty = 1.0 / mass_density.uncertainty

        mass_density.phase = specific_volume.phase

        mass_density.thermodynamic_state = specific_volume.thermodynamic_state
        mass_density.substance = specific_volume.substance

        return mass_density

    # Register the ThermoML property using the conversion function.
    register_thermoml_property(
        thermoml_string="Specific volume, m3/kg",
        supported_phases=PropertyPhase.Liquid,
        property_class=Density,
        conversion_function=specific_volume_to_mass_density
    )

Converting the different density derivatives into a single density class removes the need to produce many very similar
class representations of density measurements, and allows a single calculation schema to be defined for all variants.

Loading Data Sets
-----------------

Data sets are most easily loaded using their digital object identifiers (DOI). For example, to retrieve the `ThermoML
data set <http://trc.boulder.nist.gov/ThermoML/10.1016/j.jct.2005.03.012>`_ that accompanies `this paper
<http://www.sciencedirect.com/science/article/pii/S0021961405000741>`_, we can simply use the DOI
``10.1016/j.jct.2005.03.012``::

    data_set = ThermoMLDataset.from_doi('10.1016/j.jct.2005.03.012')

Data can be pulled from multiple sources at once by specifying multiple identifiers::

    identifiers = ['10.1021/acs.jced.5b00365', '10.1021/acs.jced.5b00474']
    dataset = ThermoMLDataset.from_doi(*identifiers)

Entire archives of properties can be downloaded directly from the `ThermoML website <https://trc.nist.gov/RSS/>`_
and parsed by the framework. For example, to create a data set object containing all of the measurements recorded
from the International Journal of Thermophysics::

    # Download the archive of all properties from the IJT journal.
    import requests
    request = requests.get("https://trc.nist.gov/ThermoML/IJT.tgz", stream=True)

    # Make sure the request went ok.
    assert request

    # Unzip the files into a new 'ijt_files' directory.
    import io, tarfile
    tar_file = tarfile.open(fileobj=io.BytesIO(request.content))
    tar_file.extractall("ijt_files")

    # Get the names of the extracted files
    import glob
    file_names = glob.glob("ijt_files/*.xml")

    # Create the data set object
    from propertyestimator.datasets.thermoml import ThermoMLDataSet
    data_set = ThermoMLDataSet.from_file(*file_names)

    # Save the data set to a JSON object
    data_set.json(file_path="ijt.json", format=True)

