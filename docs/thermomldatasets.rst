ThermoML Data Sets
==================

The ``ThermoMLDataset`` object offers an API for extracting physical properties from the `NIST ThermoML Archive
<http://trc.nist.gov/ThermoML.html>`_, both directly from the archive itself or from files stored in the IUPAC-
standard `ThermoML <http://trc.nist.gov/ThermoMLRecommendations.pdf>`_ format stored locally.

NOTE + SECTION ABOUT THE CLASS DECORATOR FOR REGISTERING PROPERTIES

For example, to retrieve the `ThermoML data set <http://trc.boulder.nist.gov/ThermoML/10.1016/j.jct.2005.03.012>`_
that accompanies `this paper <http://www.sciencedirect.com/science/article/pii/S0021961405000741>`_, we can simply
use the digital object identifier (DOI) ``10.1016/j.jct.2005.03.012``::

    data_set = ThermoMLDataset.from_doi('10.1016/j.jct.2005.03.012')

Data can be pulled from multiple sources at once by specifying multiple identifiers::

    identifiers = ['10.1021/acs.jced.5b00365', '10.1021/acs.jced.5b00474']
    dataset = ThermoMLDataset.from_doi(*identifiers)

Entire archives of properties can be downloaded directly from the `ThermoML RSS Feeds <https://trc.nist.gov/RSS/>`_
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

