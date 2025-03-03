#!/usr/bin/env python
# coding: utf-8

# In[1]:


from openff.units import unit

from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import ConnectionOptions, EvaluatorClient, RequestOptions
from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.server.server import EvaluatorServer
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState

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
        source=MeasurementSource(doi=" "),
        substance=Substance.from_components("CCCO", "O"),
    ),
)

for i, prop in enumerate(dataset.properties):
    prop.id = str(i)

# We now need to define options for both equilibration and pre-equilibrated simulation.

# In[3]:


options = RequestOptions()
options.calculation_layers = ["SimulationLayer"]
density_schema = Density.default_simulation_schema(
    n_molecules=256,
)

dhmix_schema = EnthalpyOfMixing.default_simulation_schema(
    n_molecules=256,
)
options.add_schema(
    "SimulationLayer",
    "Density",
    density_schema,
)
options.add_schema(
    "SimulationLayer",
    "EnthalpyOfMixing",
    dhmix_schema,
)


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
        port=8998,
    )
    with server:
        client = EvaluatorClient(connection_options=ConnectionOptions(server_port=8998))

        request, error = client.request_estimate(
            dataset,
            force_field_source,
            options,
        )
        results, exception = request.results(synchronous=True, polling_interval=30)

print(len(results.queued_properties))

print(len(results.estimated_properties))

print(len(results.unsuccessful_properties))
print(len(results.exceptions))
results.estimated_properties.json("estimated_data_set.json", format=True)
# In[ ]:
