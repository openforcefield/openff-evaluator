# Example Kubernetes run

This directory contains files for an example Evaluator run on Kubernetes.
Please see [the documentation](https://docs.openforcefield.org/projects/evaluator/en/stable/backends/daskkubernetesbackend.html) for more.

The `run.py` script runs the following steps:

1. Create a PersistentVolumeClaim (PVC) for storage.
2. Create a DaskKubernetesBackend, mounting the PVC. This backend runs pods using a pre-built image with Evaluator installed. The spec of the cluster is written out to ``cluster-spec.yaml``.
3. Copies `server-existing.py` to start an EvaluatorServer with the local filesystem storage mounted.
4. Create a Deployment to run the script in step 3.
5. Forward a port from the Deployment to the local machine.
6. Estimates the dataset in `dataset.json` using the EvaluatorClient.

Output from an example run is captured in `run.log`:

```
python run.py > run.log 2>&1
```

## Environments

An input environment file is provided in ``input-environment.yaml``.
The full environment specification used for the example run is provided in ``full-environment.yaml``.

## Usage

Make sure to keep an eye on GPU usage to make sure it's not too low: https://grafana.nrp-nautilus.io/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus?var-namespace=openforcefield&orgId=1&refresh=30s&from=now-1h&to=now

Note that the [KubeCluster](https://kubernetes.dask.org/en/latest/operator_kubecluster.html) scales adaptively, so even though a maximum of 10 GPU workers
are requested in the `DaskKubernetesBackend`, only 2 are launched as there are only 2 properties in the dataset.