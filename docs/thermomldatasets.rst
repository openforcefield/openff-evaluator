ThermoML Data Sets
==================

The ``ThermoMLDataset`` object offers an API for extracting physical properties from the `NIST ThermoML Archive
<http://trc.nist.gov/ThermoML.html>`_, both directly from the archive itself or from files stored in the IUPAC-
standard `ThermoML <http://trc.nist.gov/ThermoMLRecommendations.pdf>`_) format locally.

For example, to retrieve `the ThermoML dataset <http://trc.boulder.nist.gov/ThermoML/10.1016/j.jct.2005.03.012>`_
that accompanies `this paper <http://www.sciencedirect.com/science/article/pii/S0021961405000741>`_, we can simply
use the digital object identifier (DOI) ``10.1016/j.jct.2005.03.012``::

    data_set = ThermoMLDataset.from_doi('10.1016/j.jct.2005.03.012')

You can also specify multiple identifiers to create a dataset from multiple ThermoML files::

    identifiers = ['10.1021/acs.jced.5b00365', '10.1021/acs.jced.5b00474']
    dataset = ThermoMLDataset.doi(*identifiers)

Entire archives of properties can be downloaded directly from the `TRC website <>`_ and parsed by the framework::

    # Download the zip archive of all properties from the JCT journal
    x = requests.get()
    # Untar the files.
    y = untar()
    files_paths = glob()
    # Load the files into a data set object
    z = ThermoMLDataSet.from_file(file_paths

