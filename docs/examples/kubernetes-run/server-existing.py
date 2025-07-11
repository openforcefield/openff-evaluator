"""
This script is written to run on a remote Kubernetes deployment and connect to an existing DaskKubernetesBackend cluster.

The script will start an EvaluatorServer instance and listen for incoming requests.
"""

import argparse
import logging
import sys

import os

from openff.evaluator.backends.dask_kubernetes import DaskKubernetesExistingBackend, KubernetesPersistentVolumeClaim
from openff.evaluator.backends.backends import ComputeResources, PodResources
from openff.toolkit.utils import OPENEYE_AVAILABLE
from openff.evaluator.server import EvaluatorServer
import openff.evaluator
from openff.units import unit


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--cluster-name", type=str, default="evaluator-lw")
parser.add_argument("--namespace", type=str, default="openforcefield")
parser.add_argument("--storage-path", type=str, default="/evaluator-storage")
parser.add_argument("--memory", type=int, default=8, help="Memory limit in GB")
parser.add_argument("--ephemeral-storage", type=int, default=20, help="Ephemeral storage limit in GB")
parser.add_argument("--port", type=int, default=8998)




if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(f"OpenEye is available: {OPENEYE_AVAILABLE}")

    logger.info("Evaluator version: " + openff.evaluator.__version__)

    # change directory to storage path
    os.chdir(args.storage_path)

    working_directory = os.path.abspath(
        os.path.join(args.storage_path, "working-directory")
    )

    volume = KubernetesPersistentVolumeClaim(
        name="evaluator-storage-lw",
        mount_path=args.storage_path,
    )


    calculation_backend = DaskKubernetesExistingBackend(
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        cluster_port=8786,
        gpu_resources_per_worker=PodResources(
            number_of_threads=1,
            memory_limit=args.memory * unit.gigabytes,
            ephemeral_storage_limit=args.ephemeral_storage * unit.gigabytes,
            number_of_gpus=1,
            preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
            preferred_gpu_precision=ComputeResources.GPUPrecision.mixed,
        ),
        annotate_resources=True,
        volumes=[volume],
    )

    logger.info(f"Calculating with backend {calculation_backend}")
    with calculation_backend:
        evaluator_server = EvaluatorServer(
            calculation_backend,
            working_directory=working_directory,
            port=args.port,
            delete_working_files=True,
        )
        logger.info("Starting server")
        evaluator_server.start(asynchronous=False)
        
