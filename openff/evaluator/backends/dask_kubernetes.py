import copy
import logging
from enum import Enum

from dask.base import tokenize
from dask.utils import funcname
from openff.units import unit
from openff.utilities.utilities import requires_package
from pydantic import BaseModel, Field

from openff.evaluator.backends.backends import PodResources
from openff.evaluator.backends.dask import BaseDaskBackend, BaseDaskJobQueueBackend
from openff.evaluator.workflow.schemas import ProtocolSchema

logger = logging.getLogger(__name__)


class AccessMode(Enum):
    """An enumeration of the different access modes for a Kubernetes PVC"""

    READ_WRITE_ONCE = "ReadWriteOnce"
    READ_WRITE_MANY = "ReadWriteMany"
    READ_ONLY_MANY = "ReadOnlyMany"


class BaseKubernetesVolume(BaseModel):
    """A helper base class for specifying Kubernetes volume-like objects."""

    name: str = Field(
        ..., description="The name assigned to the volume during this run."
    )
    mount_path: str = Field(..., description="The path to mount the volume to.")

    def _to_volume_mount_spec(self):
        mount_path = self.mount_path
        if not mount_path.startswith("/"):
            mount_path = f"/{mount_path}"
        return {
            "name": self.name,
            "mountPath": mount_path,
            "readOnly": self.read_only,
        }

    @requires_package("kubernetes")
    def _to_volume_mount_k8s(self):
        from kubernetes import client

        return client.V1VolumeMount(
            name=self.name,
            mount_path=self.mount_path,
            read_only=self.read_only,
        )


class KubernetesSecret(BaseKubernetesVolume):
    """A helper class for specifying Kubernetes secrets."""

    secret_name: str = Field(..., description="The name of the saved secret to use.")
    sub_path: str = Field(None, description="The sub path to mount the secret to.")
    read_only: bool = Field(True, description="Whether the volume should be read-only.")

    def _to_volume_spec(self):
        return {
            "name": self.name,
            "secret": {"secretName": self.secret_name},
        }

    def _to_volume_mount_spec(self):
        spec = super()._to_volume_mount_spec()
        spec["subPath"] = self.sub_path
        return spec

    @requires_package("kubernetes")
    def _to_volume_mount_k8s(self):
        volume_mount = super()._to_volume_mount_k8s()
        volume_mount.sub_path = self.sub_path
        return volume_mount

    @requires_package("kubernetes")
    def _to_volume_k8s(self):
        from kubernetes import client

        return client.V1Volume(
            name=self.name,
            secret=client.V1SecretVolumeSource(secret_name=self.secret_name),
        )


class KubernetesPersistentVolumeClaim(BaseKubernetesVolume):
    """A helper class for specifying Kubernetes volumes."""

    read_only: bool = Field(
        False, description="Whether the volume should be read-only."
    )

    def _generate_pvc_spec(
        self,
        storage_class_name: str = "rook-cephfs-central",
        access_mode: AccessMode = AccessMode.READ_WRITE_MANY,
        storage: unit.Quantity = 5 * unit.terabytes,
    ):
        storage_tb = storage.to(unit.terabytes).m
        spec = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": self.name},
            "spec": {
                "storageClassName": storage_class_name,
                "accessModes": [access_mode.value],
                "resources": {"requests": {"storage": f"{storage_tb}Ti"}},
            },
        }
        return spec

    def _to_volume_spec(self):
        return {
            "name": self.name,
            "persistentVolumeClaim": {"claimName": self.name},
        }

    @requires_package("kubernetes")
    def _to_volume_k8s(self):
        from kubernetes import client

        return client.V1Volume(
            name=self.name,
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=self.name
            ),
        )


class KubernetesEmptyDirVolume(BaseKubernetesVolume):
    """A helper class for specifying Kubernetes emptyDir volumes."""

    read_only: bool = Field(
        False, description="Whether the volume should be read-only."
    )

    def _to_volume_spec(self):
        return {
            "name": self.name,
            "emptyDir": {},
        }

    @requires_package("kubernetes")
    def _to_volume_k8s(self):
        from kubernetes import client

        return client.V1Volume(
            name=self.name,
            empty_dir=client.V1EmptyDirVolumeSource(),
        )


class BaseDaskKubernetesBackend(BaseDaskBackend):

    def __init__(
        self,
        gpu_resources_per_worker=None,
        cpu_resources_per_worker=None,
        cluster_name="openff-evaluator",
        cluster_port=8786,
        disable_nanny_process=False,
        image: str = "ghcr.io/lilyminium/openff-images:evaluator-0.4.10-kubernetes-dask-v0",
        namespace: str = "openforcefield",
        env: dict = None,
        secrets: list[KubernetesSecret] = None,
        volumes: list[KubernetesPersistentVolumeClaim] = None,
        cluster_kwargs: dict = None,
        annotate_resources: bool = False,
        include_jupyter: bool = False,
    ):
        default_resources = None
        other_resources = {}
        if gpu_resources_per_worker is not None:
            # preference gpu resources over cpu
            assert isinstance(gpu_resources_per_worker, PodResources)
            default_resources = gpu_resources_per_worker
            other_resources["cpu"] = cpu_resources_per_worker
        elif cpu_resources_per_worker is not None:
            assert isinstance(cpu_resources_per_worker, PodResources)
            default_resources = cpu_resources_per_worker
        else:
            raise ValueError(
                "Either gpu_resources_per_worker or cpu_resources_per_worker must be specified."
            )

        super().__init__(
            default_resources._minimum_number_of_workers,
            default_resources,
        )

        self._cluster_name = cluster_name
        self._cluster_port = cluster_port
        self._namespace = namespace
        self._annotate_resources = annotate_resources
        self._image = image
        self._other_resources = other_resources
        self._include_jupyter = include_jupyter
        self._disable_nanny_process = disable_nanny_process
        self._env = {}
        if env is not None:
            assert isinstance(env, dict)
            self._env.update(env)

        self._secrets = []
        if secrets is not None:
            assert isinstance(secrets, list)
            for secret in secrets:
                assert isinstance(secret, KubernetesSecret)
                self._secrets.append(secret)

        self._volumes = []
        if volumes is not None:
            assert isinstance(volumes, list)
            for volume in volumes:
                assert isinstance(volume, BaseKubernetesVolume)
                self._volumes.append(volume)

        # fail if there are no volumes -- we need volumes...
        # unless we swap to S3?
        if len(self._volumes) == 0:
            raise ValueError("No volumes specified. We need at least a filesystem")

        self._cluster_kwargs = {}
        if cluster_kwargs is not None:
            assert isinstance(cluster_kwargs, dict)
            self._cluster_kwargs.update(cluster_kwargs)

    def _get_function_key(self, function, args, kwargs) -> str:
        """Returns a more useful function key

        Currently all functions passed through Dask are called `wrapped_function`.
        This returns a key that is either the actual function name, or the
        Protocol if the function is to execute a protocol.
        """
        funckey = funcname(function)
        if len(args) >= 2:
            # might be a protocol
            try:
                # TODO: might be slow...
                schema = ProtocolSchema.parse_json(args[1])
            except Exception:
                pass
            else:
                funckey = f"{schema.type}-{schema.id}"

        tokenized = tokenize(function, kwargs, *args)
        return f"{funckey}-{tokenized}"

    def submit_task(self, function, *args, **kwargs):
        from openff.evaluator.workflow.plugins import registered_workflow_protocols

        key = kwargs.pop("key", None)
        if key is None:
            key = self._get_function_key(function, args, kwargs)

        protocols_to_import = [
            protocol_class.__module__ + "." + protocol_class.__qualname__
            for protocol_class in registered_workflow_protocols.values()
        ]

        # look for simulation protocols
        if self._annotate_resources:
            resources = kwargs.get("resources", {})
            if len(args) >= 2:
                # schema is the second argument
                # awful temporary terribad hack
                schema_json = args[1]
                if (
                    '".allow_gpu_platforms": true' in schema_json
                    or "energy_minimisation" in schema_json
                ):
                    resources["GPU"] = 0.5
                    resources["notGPU"] = 0
                else:
                    resources["GPU"] = 0
                    resources["notGPU"] = 1
            kwargs["resources"] = resources
            logger.debug(f"Annotating resources: {resources}")

        return self._client.submit(
            BaseDaskJobQueueBackend._wrapped_function,
            function,
            *args,
            **kwargs,
            available_resources=self._resources_per_worker,
            registered_workflow_protocols=protocols_to_import,
            gpu_assignments={},
            per_worker_logging=True,
            key=key,
        )


class DaskKubernetesBackend(BaseDaskKubernetesBackend):
    """
    A class which defines a Dask backend which runs on a Kubernetes cluster

    This class is a wrapper around the Dask Kubernetes cluster class that
    uses the Dask Kubernetes operator. It allows for the creation of a
    Dask cluster on a Kubernetes cluster with adaptive scaling.
    However, adaptive scaling currently *only applies to the "default" worker group*.
    This is preferentially the GPU worker group, but will fall back to the CPU
    worker group if no GPU worker resources are specified.

    Parameters
    ----------
    gpu_resources_per_worker: PodResources
        The resources to allocate to each GPU worker.
    cpu_resources_per_worker: PodResources
        The resources to allocate to each CPU worker.
    cluster_name: str
        The name of the Dask cluster.
    cluster_port: int
        The port to use for the Dask cluster.
    disable_nanny_process: bool
        Whether to disable the Dask nanny process.
    image: str
        The Docker image to use for the Dask cluster.
    namespace: str
        The Kubernetes namespace to use.
    env: dict
        The environment variables to use for the Dask cluster.
    secrets: list[KubernetesSecret]
        The Kubernetes secrets to use for the Dask cluster.
    volumes: list[KubernetesPersistentVolumeClaim]
        The Kubernetes volumes to use for the Dask cluster.
    cluster_kwargs: dict
        Additional keyword arguments to pass to the Dask KubeCluster
        constructor.
    annotate_resources: bool
        Whether to annotate resources for the Dask cluster.
    include_jupyter: bool
        Whether to include a Jupyter notebook in the Dask cluster.
    """

    @requires_package("dask_kubernetes")
    def _generate_cluster_spec(self) -> dict[str, dict]:
        """
        Generate a Dask Kubernetes cluster specification
        that can be used to create a Dask cluster on a Kubernetes cluster.
        """
        from dask_kubernetes.operator import make_cluster_spec

        resources = self._resources_per_worker._to_kubernetes_resource_limits()
        full_resources = {
            "requests": copy.deepcopy(resources),
            "limits": copy.deepcopy(resources),
        }
        spec = make_cluster_spec(
            name=self._cluster_name,
            image=self._image,
            n_workers=self._resources_per_worker._minimum_number_of_workers,
            resources=full_resources,
            jupyter=self._include_jupyter,
            env=self._env,
        )

        # remove any gpu specifications from scheduler
        scheduler_spec = spec["spec"]["scheduler"]["spec"]
        scheduler_container = scheduler_spec["containers"][0]
        # need longer than default
        scheduler_container["readinessProbe"]["timeoutSeconds"] = 3600
        scheduler_resources = copy.deepcopy(full_resources)
        scheduler_resources["requests"].pop("nvidia.com/gpu", None)
        scheduler_resources["limits"].pop("nvidia.com/gpu", None)

        # set up port
        scheduler_container["resources"] = scheduler_resources
        port_list = scheduler_container["ports"]
        for port_spec in port_list:
            if port_spec["name"] == "tcp-comm":
                port_spec["containerPort"] = self._cluster_port

        # update worker spec
        worker_spec = spec["spec"]["worker"]["spec"]
        if self._resources_per_worker._affinity_specification:
            worker_spec["affinity"] = copy.deepcopy(
                self._resources_per_worker._affinity_specification
            )

        # update worker command with resources
        if self._annotate_resources:
            self._resources_per_worker._update_worker_with_resources(worker_spec)
        worker_container = worker_spec["containers"][0]

        # add volume mounts
        volume_mounts, volumes = self._generate_volume_specifications()

        # deepcopy all the dicts in case we write out to yaml
        # having references to the same object makes things weird
        worker_container["volumeMounts"] = copy.deepcopy(volume_mounts)
        scheduler_container["volumeMounts"] = copy.deepcopy(volume_mounts)
        worker_spec["volumes"] = copy.deepcopy(volumes)
        scheduler_spec["volumes"] = copy.deepcopy(volumes)

        return spec

    def _generate_volume_specifications(self) -> tuple[list[dict], list[dict]]:
        """
        Generate the volume mount and volume specifications for the cluster

        Returns
        -------
        tuple[list[dict], list[dict]]
            A tuple of lists of dictionaries representing the volume mounts
            and volumes for the cluster, in that order.
        """
        volume_mounts = []
        volumes = []

        for volume in self._volumes + self._secrets:
            volume_spec = volume._to_volume_spec()
            volume_mount_spec = volume._to_volume_mount_spec()

            volume_mounts.append(dict(volume_mount_spec))
            volumes.append(dict(volume_spec))

        return volume_mounts, volumes

    @requires_package("dask_kubernetes")
    def _generate_worker_spec(self, pod_resources) -> dict[str, dict]:
        """
        Generate a Dask Kubernetes worker specification
        """
        from dask_kubernetes.operator import make_worker_spec

        resources = pod_resources._to_kubernetes_resource_limits()

        k8s_resources = {
            "limits": copy.deepcopy(resources),
            "requests": copy.deepcopy(resources),
        }

        worker_spec = make_worker_spec(
            resources=k8s_resources,
            n_workers=pod_resources._maximum_number_of_workers,
            image=self._image,
            env=self._env,
        )

        # add volume mounts
        worker_container = worker_spec["spec"]["containers"][0]
        volume_mounts, volumes = self._generate_volume_specifications()
        worker_container["volumeMounts"] = copy.deepcopy(volume_mounts)
        worker_spec["spec"]["volumes"] = copy.deepcopy(volumes)

        # update worker spec
        if self._resources_per_worker._affinity_specification:
            worker_spec["spec"]["affinity"] = copy.deepcopy(
                self._resources_per_worker._affinity_specification
            )

        # update worker command with resources
        if self._annotate_resources:
            pod_resources._update_worker_with_resources(worker_spec["spec"])
        return worker_spec

    @requires_package("dask_kubernetes")
    def start(self):
        from dask_kubernetes.operator import KubeCluster
        from kubernetes import config

        config.load_kube_config()

        spec = self._generate_cluster_spec()
        self._cluster = KubeCluster(
            namespace=self._namespace, custom_cluster_spec=spec, **self._cluster_kwargs
        )
        self._cluster.adapt(
            minimum=self._resources_per_worker._minimum_number_of_workers,
            maximum=self._resources_per_worker._maximum_number_of_workers,
        )
        # add other worker groups
        for name, resources in self._other_resources.items():
            worker_spec = self._generate_worker_spec(resources)
            self._cluster.add_worker_group(
                name=name,
                n_workers=resources._maximum_number_of_workers,
                custom_spec=worker_spec,
            )

        super().start()


class DaskKubernetesExistingBackend(BaseDaskKubernetesBackend):
    """
    A class which defines a Dask backend which runs on an existing Kubernetes cluster.

    This class simply connects to an existing Dask cluster.
    Note that it is still important to define default resources
    as some of these get passed onto the protocols themselves,
    e.g. GPU availability and the GPUToolkit.

    """

    def start(self):
        self._cluster = (
            f"tcp://{self._cluster_name}-scheduler"
            f".{self._namespace}.svc.cluster.local:"
            f"{self._cluster_port}"
        )
        super().start()

    def stop(self):
        logger.warning("Cannot stop an existing Kubernetes cluster.")
