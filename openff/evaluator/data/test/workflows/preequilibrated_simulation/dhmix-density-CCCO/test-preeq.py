#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openff.units import unit

from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import EvaluatorClient, RequestOptions
from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.server.server import EvaluatorServer
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.observables import ObservableType

# ## Create a toy dataset
#
# Properties can be downloaded from the internet, loaded from a file, or created dynamically. Here we quickly create a small dataset.

# In[2]:


dataset = PhysicalPropertyDataSet()
thermodynamic_state = ThermodynamicState(
    temperature=298.15 * unit.kelvin,
    pressure=101.325 * unit.kilopascal,
)
dataset.add_properties(
    Density(
        thermodynamic_state=thermodynamic_state,
        phase=PropertyPhase.Liquid,
        value=1.0 * Density.default_unit(),
        uncertainty=1.0 * Density.default_unit(),
        source=MeasurementSource(doi=" "),
        substance=Substance.from_components("CCCO"),
    ),
    EnthalpyOfMixing(
        thermodynamic_state=thermodynamic_state,
        phase=PropertyPhase.Liquid,
        value=1.0 * EnthalpyOfMixing.default_unit(),
        uncertainty=1.0 * EnthalpyOfMixing.default_unit(),
        source=MeasurementSource(doi=" "),
        substance=Substance.from_components("CCCO", "O"),
    ),
)

for i, prop in enumerate(dataset.properties):
    prop.id = str(i)

# We now need to define options for both equilibration and pre-equilibrated simulation.

# In[3]:

potential_energy = EquilibrationProperty()
potential_energy.absolute_tolerance = 200 * unit.kilojoules_per_mole
potential_energy.observable_type = ObservableType.PotentialEnergy

density = EquilibrationProperty()
density.relative_tolerance = 0.2
density.observable_type = ObservableType.Density


options = RequestOptions()
options.calculation_layers = ["PreequilibratedSimulationLayer"]
density_schema = Density.default_preequilibrated_simulation_schema(
    n_molecules=256,
    equilibration_error_tolerances=[potential_energy, density],
   n_uncorrelated_samples=100
)

dhmix_schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
    n_molecules=256,
    equilibration_error_tolerances=[potential_energy, density],
    n_uncorrelated_samples=100,
)

options.add_schema("PreequilibratedSimulationLayer", "Density", density_schema)
options.add_schema("PreequilibratedSimulationLayer", "EnthalpyOfMixing", dhmix_schema)

# We load a force field from SMIRNOFF.

# In[5]:


force_field_path = "openff-2.1.0.offxml"
force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)


# In[ ]:


with DaskLocalCluster(
    number_of_workers=1,
    resources_per_worker=ComputeResources(
        number_of_threads=1,
        number_of_gpus=1,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
    ),
) as calculation_backend:
    server = EvaluatorServer(
        calculation_backend=calculation_backend,
        working_directory=".",
        delete_working_files=False,
    )
    with server:
        client = EvaluatorClient()

        # test equilibration
        request, error = client.request_estimate(
            dataset,
            force_field_source,
            options,
        )

        results, exception = request.results(synchronous=True, polling_interval=30)

print(f"Simulation complete")
print(f"# estimated: {len(results.estimated_properties)}")
print(f"# unsuccessful: {len(results.unsuccessful_properties)}")
print(f"# exceptions: {len(results.exceptions)}")
