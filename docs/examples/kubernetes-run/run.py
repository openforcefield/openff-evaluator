import contextlib
import logging
import copy
import os
import pathlib
import subprocess
import sys
import time

import click
from kubernetes import client, config
import yaml

from openff.evaluator.backends.dask_kubernetes import (
    KubernetesPersistentVolumeClaim, KubernetesSecret,
    DaskKubernetesBackend,
)
from openff.units import unit
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions
from openff.evaluator.client import EvaluatorClient, RequestOptions, ConnectionOptions
from openff.evaluator.forcefield import SmirnoffForceFieldSource

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def _save_script(contents: str, path: str):
    """Save a script to a path.
    
    Parameters
    ----------
    contents : str
        The contents of the script.
    path : str
        The path to save the script to.
    
    """
    with open(path, "w") as f:
        f.write(contents)
    return path


def copy_file_to_storage(
    evaluator_backend,
    input_file,
    output_file,
):
    """
    Copy a file to the storage of a Kubernetes cluster.

    Parameters
    ----------
    evaluator_backend : DaskKubernetesBackend
        The backend to copy the file to.
    input_file : str
        The path to the file to copy (locally).
    output_file : str
        The path to save the file to (remotely).
    """
    with open(input_file, "r") as f:
        data = f.read()
    future = evaluator_backend._client.submit(_save_script, data, output_file, resources={"notGPU": 1, "GPU": 0})
    future.result()
    logger.info(f"Copied {input_file} to {output_file}")


def wait_for_pod(
    pod_name: str,
    namespace: str,
    status: str = "Running",
    timeout: int = 1000,
    polling_interval: int = 10,
):
    """
    Wait for a pod to reach a certain status.

    Parameters
    ----------
    pod_name : str
        The name of the pod.
    namespace : str
        The namespace of the pod.
    status : str
        The status to wait for.
    timeout : int
        The maximum time to wait.
    polling_interval : int
        The interval to poll the pod status.

    
    Raises
    ------
    TimeoutError
        If the pod does not reach the desired status within the timeout.
    """
    core_v1 = client.CoreV1Api()

    start_time = time.time()
    while time.time() - start_time < timeout:
        pod = core_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        if pod.status.phase == status:
            return pod
        time.sleep(polling_interval)
    
    raise TimeoutError(f"Pod {pod_name} did not reach status {status} within {timeout} seconds.")
    


def get_pod_name(
    deployment_name: str,
    namespace: str = "openforcefield",
) -> str:
    """
    Get the pod name of a deployment


    Parameters
    ----------
    deployment_name : str
        The name of the deployment.
    namespace : str
        The namespace of the deployment.


    Returns
    -------
    str
        The name of a pod in the deployment.
    """
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    
    # Get the deployment's labels
    deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    deployment_labels = deployment.spec.selector.match_labels

    # List pods with the deployment's labels
    label_selector = ",".join([f"{key}={value}" for key, value in deployment_labels.items()])
    pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector).items
    pod_name = pods[0].metadata.name.split("_")[0]
    return pod_name

    
@contextlib.contextmanager
def forward_port(
    deployment_name,
    namespace: str = "openforcefield",
    port: int = 8998,
):
    """
    Forward a port from a Kubernetes deployment to the local machine.

    This assumes that the deployment has at least one pod.

    Parameters
    ----------
    deployment_name : str
        The name of the deployment.
    namespace : str
        The namespace of the deployment.
    port : int
        The port to forward.
    """

    pod_name = get_pod_name(deployment_name, namespace)
    print(f"Pod name: {pod_name}")

    # Wait for the pod to be running
    wait_for_pod(pod_name, namespace, status="Running")
    command = [
        "kubectl", "port-forward", f"pod/{pod_name}", f"{port}:{port}",
        "-n", namespace,
    ]
    logger.info(f"Forwarding port {port} to pod {pod_name}")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the port forward to be established
    time.sleep(5)
    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"Port forward failed: {stderr.decode()}")
    try:
        yield
    finally:
        proc.terminate()



def create_pvc(
    namespace: str = "openforcefield",
    job_name: str = "lw",
    storage_class_name: str = "rook-cephfs-central",
    storage_space: unit.Quantity = 2 * unit.terabytes,
    apply_pvc: bool = True,
    timeout: int = 1000,
) -> str:
    """
    Create a persistent volume claim and deploy it.

    Possibly could be turned into a method of `KubernetesPersistentVolumeClaim`.

    Parameters
    ----------
    namespace : str
        The namespace to deploy the PVC in.
    job_name : str
        The name of the job.
    storage_class_name : str
        The name of the storage class (on NRP) to request a PVC.
    storage_space : unit.Quantity
        The amount of storage to request.
    apply_pvc : bool
        Whether to launch the PVC.
    timeout : int
        The maximum time to wait for the PVC to be bound.

    Returns
    -------
    str
        The name of the PVC.
    """
    core_v1 = client.CoreV1Api()
    
    pvc_spec = client.V1PersistentVolumeClaimSpec(
        access_modes=["ReadWriteMany"],
        storage_class_name=storage_class_name,
        resources=client.V1ResourceRequirements(
            requests={
                "storage": f"{storage_space.to(unit.gigabytes).m}Gi",
            }
        ),
    )


    pvc_name = f"evaluator-storage-{job_name}"
    metadata = client.V1ObjectMeta(name=pvc_name)
    pvc = client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=metadata,
        spec=pvc_spec,
    )
    if apply_pvc:
        api_response = core_v1.create_namespaced_persistent_volume_claim(
            namespace=namespace,
            body=pvc
        )
        logger.info(
            f"Created PVC {pvc.metadata.name}. State={api_response.status.phase}"
        )
    
        # wait
        end_time = time.time() + timeout
        while time.time() < end_time:
            pvc = core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
            if pvc.status.phase == "Bound":
                logger.info(f"PVC '{pvc_name}' is Bound.")
                return pvc_name
            logger.info(f"Waiting for PVC '{pvc_name}' to become Bound. Current phase: {pvc.status.phase}")
            time.sleep(5)
    return pvc_name


def create_deployment(
    calculation_backend,
    remote_script_path: str,
    remote_storage_path: str,
    env: dict = None,
    volumes: list[KubernetesPersistentVolumeClaim] = None,
    secrets: list[KubernetesSecret] = None,
    namespace: str = "openforcefield",
    job_name: str = "lw",
    port: int = 8998,
    image: str = "ghcr.io/lilyminium/openff-images:tmp-evaluator-dask-v2",
):
    """
    Create Kubernetes deployment for Evaluator server.

    Parameters
    ----------
    calculation_backend : DaskKubernetesBackend
        The backend to use.
    remote_script_path : str
        The path to the script to coy over and run
    remote_storage_path : str
        The path to the filesystem storage to mount.
    env : dict
        Environment variables to set in the container.
    volumes : list[KubernetesPersistentVolumeClaim]
        Volumes to mount.
    secrets : list[KubernetesSecret]
        Secrets to mount.
    namespace : str
        The namespace to deploy the deployment in.
    job_name : str
        The name of the job.
    port : int
        The server port to expose.
    image : str
        The image to use for each container.
    """
    server_name = f"evaluator-server-{job_name}-deployment"
    apps_v1 = client.AppsV1Api()
    
    metadata = client.V1ObjectMeta(
        name=f"evaluator-server-{job_name}",
        labels={"k8s-app": server_name},
    )

    # generate volume mounts and volumes
    k8s_volume_mounts = []
    k8s_volumes = []
    
    if volumes is None:
        volumes = []
    if secrets is None:
        secrets = []
    for volume in volumes + secrets:
        k8s_volume_mounts.append(volume._to_volume_mount_k8s())
        k8s_volumes.append(volume._to_volume_k8s())

    k8s_env = {}
    if env is not None:
        assert isinstance(env, dict)
        k8s_env.update(env)

    k8s_env_objects = [
        client.V1EnvVar(name=key, value=value)
        for key, value in k8s_env.items()
    ]
    resources = calculation_backend._resources_per_worker

    command = [
        "python",
        remote_script_path,
        "--cluster-name",
        calculation_backend._cluster.name,
        "--namespace",
        calculation_backend._cluster.namespace,
        "--memory",
        str(resources._memory_limit.m_as(unit.gigabytes)),
        "--ephemeral-storage",
        str(resources._ephemeral_storage_limit.m_as(unit.gigabytes)),
        "--storage-path",
        remote_storage_path,
        "--port",
        str(port)
    ]
    logger.info(f"Command: {command}")
    
    container = client.V1Container(
        name=server_name,
        image=image,
        env=k8s_env_objects,
        command=command,
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1", "memory": "4Gi"},
            limits={"cpu": "1", "memory": "4Gi"},
        ),
        volume_mounts=k8s_volume_mounts,
    )

    deployment_spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(
            match_labels={"k8s-app": server_name}
        ),
        template=client.V1PodTemplateSpec(
            metadata=metadata,
            spec=client.V1PodSpec(
                containers=[container],
                volumes=k8s_volumes,
            )
        ),
    )

    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=metadata,
        spec=deployment_spec,
    )

    # submit
    api_response = apps_v1.create_namespaced_deployment(
        namespace=namespace,
        body=deployment,
    )
    logger.info(
        f"Created deployment {deployment.metadata.name}. State={api_response.status}"
    )
    return deployment.metadata.name
    

def simulate(
    dataset_path: str = "dataset.json",
    n_molecules: int = 256,
    force_field: str = "openff-2.1.0.offxml",
    port: int = 8000
):
    """
    Simulate and run a dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset.
    n_molecules : int
        The number of molecules to simulate in a liquid box.
    force_field : str
        The path to the force field.
    port : int
        The server port to connect to
    """
    # load dataset
    dataset = PhysicalPropertyDataSet.from_json(dataset_path)
    print(f"Loaded {len(dataset.properties)} properties from {dataset_path}")

    error = 50

    options = RequestOptions()
    options.calculation_layers = ["SimulationLayer"]
    density_schema = Density.default_simulation_schema(n_molecules=n_molecules)

    dhmix_schema = EnthalpyOfMixing.default_simulation_schema(n_molecules=n_molecules)

    options.add_schema("SimulationLayer", "Density", density_schema)
    options.add_schema("SimulationLayer", "EnthalpyOfMixing", dhmix_schema)
    
    force_field_source = SmirnoffForceFieldSource.from_path(
        force_field
    )

    client = EvaluatorClient(
        connection_options=ConnectionOptions(server_port=port)
    )

    # we first request the equilibration data
    # this can be copied between different runs to avoid re-running
    # the data is saved in a directory called "stored_data"

    request, error = client.request_estimate(
        dataset,
        force_field_source,
        options,
    )
    assert error is None, error

    # block until computation finished
    results, exception = request.results(synchronous=True, polling_interval=30)
    assert exception is None, exception

    print(f"Simulation complete")
    print(f"# estimated: {len(results.estimated_properties)}")
    print(f"# unsuccessful: {len(results.unsuccessful_properties)}")
    print(f"# exceptions: {len(results.exceptions)}")

    with open("results.json", "w") as f:
        f.write(results.json())



@click.command()
@click.option("--namespace", default="openforcefield", help="The namespace to operate in.")
@click.option("--job-name", default="lw", help="The name of the job.")
@click.option("--storage-class-name", default="rook-cephfs-central", help="The name of the storage class to use for the PVC.")
@click.option("--storage-path", default="/evaluator-storage", help="The path to local filesystem storage for Evaluator.")
@click.option("--script-file", default="server-existing.py", help="The path to the script to copy over and run to execute an EvaluatorServer.")
@click.option("--port", default=8998, help="The port to forward from the deployment.")
@click.option("--image", default="ghcr.io/lilyminium/openff-images:tmp-evaluator-dask-v2", help="The image to use for the deployment.")
def main(
    namespace: str = "openforcefield",
    job_name: str = "lw",
    storage_class_name: str = "rook-cephfs-central",
    storage_path: str = "/evaluator-storage",
    script_file: str = "server-existing.py",
    port: int = 8998,
    image: str = "ghcr.io/lilyminium/openff-images:tmp-evaluator-dask-v2",
    storage_space: unit.Quantity = 500 * unit.gigabytes,
    memory: unit.Quantity = 8 * unit.gigabytes,
    ephemeral_storage: unit.Quantity = 20 * unit.gigabytes,
):
    """
    Run Evaluator on a Kubernetes cluster.

    This script runs the following steps:

    1. Create a PersistentVolumeClaim (PVC) for storage.
    2. Create a DaskKubernetesBackend, mounting the PVC.
    3. Copy a script to start an EvaluatorServer to the storage.
    4. Create a Deployment to run the script in step 3.
    5. Forward a port from the Deployment to the local machine.
    6. Run a simulation using the EvaluatorClient.

    Parameters
    ----------
    namespace : str
        The namespace to operate in.
    job_name : str
        The name of the job.
    storage_class_name : str
        The name of the storage class to use for the PVC.
    storage_space : unit.Quantity
        The amount of storage to request (should be compatible with GB).
    memory : unit.Quantity
        The amount of memory to request (should be compatible with GB).
    ephemeral_storage : unit.Quantity
        The amount of ephemeral storage to request (should be compatible with GB).
    storage_path : str
        The path to local filesystem storage for Evaluator.
    script_file : str
        The path to the script to copy over and run to execute an EvaluatorServer.
    port : int
        The port to forward from the deployment.
    image : str
        The image to use for the deployment.
    
    """
    config.load_kube_config()
    core_v1 = client.CoreV1Api()

    from openff.evaluator.backends.backends import PodResources, ComputeResources


    results = None

    # run in a try/except to clean up on error
    try:
        # 1. set up filesystem storage with PVC
        pvc_name = create_pvc(
            namespace=namespace,
            job_name=job_name,
            storage_class_name=storage_class_name,
            storage_space=storage_space,
            apply_pvc=True,
        )

        # 2. create and submit KubeCluster
        volume = KubernetesPersistentVolumeClaim(
            name=pvc_name,
            mount_path=storage_path,
        )
        secret = KubernetesSecret(
            name="openeye-license",
            secret_name="oe-license-feb-2024",
            mount_path="/secrets/oe_license.txt",
            sub_path="oe_license.txt",
            read_only=True,
        )
        cluster_name = f"evaluator-{job_name}"
        calculation_backend = DaskKubernetesBackend(
            cluster_name=cluster_name,
            gpu_resources_per_worker=PodResources(
                minimum_number_of_workers=0,
                maximum_number_of_workers=10,
                number_of_threads=1,
                memory_limit=memory,
                ephemeral_storage_limit=ephemeral_storage,
                number_of_gpus=1,
                preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
            ),
            cpu_resources_per_worker=PodResources(
                minimum_number_of_workers=0,
                maximum_number_of_workers=40,
                number_of_threads=1,
                memory_limit=memory,
                ephemeral_storage_limit=ephemeral_storage,
                number_of_gpus=0,
            ),
            image=image,
            namespace=namespace,
            env={
                "OE_LICENSE": "/secrets/oe_license.txt",
                # daemonic processes are not allowed to have children
                "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
                "DASK_LOGGING__DISTRIBUTED": "debug",
                "DASK__TEMPORARY_DIRECTORY": "/evaluator-storage",
                "STORAGE_DIRECTORY": "/evaluator-storage",
                "EXTRA_PIP_PACKAGES": "jupyterlab"
            },
            volumes=[volume],
            secrets=[secret],
            annotate_resources=True,
            cluster_kwargs={"resource_timeout": 300}
        )

        spec = calculation_backend._generate_cluster_spec()
        with open("cluster-spec.yaml", "w") as f:
            yaml.safe_dump(spec, f)
        calculation_backend.start()

        logger.info(f"Calculating with backend {calculation_backend}")

        # 3. copy script to storage
        remote_script_file = os.path.join(storage_path, pathlib.Path(script_file).name)
        copy_file_to_storage(
            calculation_backend,
            script_file,
            remote_script_file
        )



        # 4. create and submit deployment
        deployment_name = create_deployment(
            calculation_backend,
            remote_script_file,
            storage_path,
            volumes=[volume],
            secrets=[secret],
            namespace=namespace,
            job_name=job_name,
            port=port,
            env={
                "OE_LICENSE": "/secrets/oe_license.txt",
            },
            image=image,
        )

        # 5. forward port
        with forward_port(
            deployment_name,
            namespace=namespace,
            port=port,
        ):
            # 6. run simulation
            simulate(
                dataset_path="dataset.json",
                n_molecules=256,
                force_field="openff-2.1.0.offxml",
                port=port
            )

    except Exception as e:
        print(e)
        raise e

    finally:


        print(f"Cleaning up")
        # clean up deployment
        apps_v1 = client.AppsV1Api()
        apps_v1.delete_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
        )

        # clean up pvc
        # note this may fail if you have another pod looking at the storage
        core_v1.delete_namespaced_persistent_volume_claim(
            name=pvc_name,
            namespace=namespace,
        )
        
    


if __name__ == "__main__":
    main()
