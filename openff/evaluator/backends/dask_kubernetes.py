import contextlib
import copy
from enum import Enum
import logging
import os
import pathlib
import subprocess
import time


from openff.evaluator._pydantic import BaseModel, Field
from openff.evaluator.backends.dask import (
    BaseDaskBackend, BaseDaskJobQueueBackend
)
from openff.evaluator.backends.backends import PodResources

from openff.units import unit
from openff.utilities.utilities import requires_package


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
    mount_path: str = Field(
        ..., description="The path to mount the volume to."
    )
    read_only: bool = Field(
        False, description="Whether the volume should be read-only."
    )

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
    secret_name: str = Field(
        ..., description="The name of the saved secret to use."
    )
    sub_path: str = Field(
        None, description="The sub path to mount the secret to."
    )

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
        cluster_name="openff-evaluator",
        cluster_port=8786,
        namespace: str = "openforcefield",
        number_of_workers: int = -1,
        resources_per_worker: PodResources = PodResources(),
        annotate_resources: bool = True,
    ):
        
        super().__init__(number_of_workers, resources_per_worker)

        self._cluster_name = cluster_name
        self._cluster_port = cluster_port
        self._namespace = namespace
        self._annotate_resources = annotate_resources

    def submit_task(self, function, *args, **kwargs):
        from openff.evaluator.workflow.plugins import registered_workflow_protocols

        key = kwargs.pop("key", None)

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
        logger.info(f"Annotating resources {self._annotate_resources}: {resources}")

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
    """A class which defines a Dask backend which runs on a Kubernetes cluster"""

    def __init__(
        self,
        minimum_number_of_workers=1,
        maximum_number_of_workers=1,
        resources_per_worker=PodResources(),
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
    ):
        
        super().__init__(
            cluster_name,
            cluster_port,
            namespace,
            minimum_number_of_workers,
            resources_per_worker,
            annotate_resources=annotate_resources,
        )

        assert isinstance(resources_per_worker, PodResources)
        assert minimum_number_of_workers <= maximum_number_of_workers

        if resources_per_worker.number_of_gpus > 0:
            if resources_per_worker.number_of_gpus > 1:
                raise ValueError("Only one GPU per worker is currently supported.")
        
        self._minimum_number_of_workers = minimum_number_of_workers
        self._maximum_number_of_workers = maximum_number_of_workers
        self._image = image
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

        self._cluster_kwargs = {}
        if cluster_kwargs is not None:
            assert isinstance(cluster_kwargs, dict)
            self._cluster_kwargs.update(cluster_kwargs)
    

    @requires_package("dask_kubernetes")
    def _generate_cluster_spec(self):
        from dask_kubernetes.operator import make_cluster_spec

        resources = self._resources_per_worker._to_kubernetes_resource_limits()
        full_resources = {
            "requests": copy.deepcopy(resources),
            "limits": copy.deepcopy(resources),
        }
        spec = make_cluster_spec(
            name=self._cluster_name,
            image=self._image,
            n_workers=self._minimum_number_of_workers,
            resources=full_resources,
            jupyter=False,
            env=self._env
        )

        # remove any gpu specifications from scheduler
        scheduler_spec = spec["spec"]["scheduler"]["spec"]
        scheduler_container = scheduler_spec["containers"][0]
        scheduler_resources = copy.deepcopy(full_resources)
        scheduler_resources["requests"].pop("nvidia.com/gpu", None)
        scheduler_resources["limits"].pop("nvidia.com/gpu", None)

        scheduler_container["resources"] = scheduler_resources
        port_list = scheduler_container["ports"]
        for port_spec in port_list:
            if port_spec["name"] == "tcp-comm":
                port_spec["containerPort"] = self._cluster_port

        worker_spec = spec["spec"]["worker"]["spec"]
        worker_container = worker_spec["containers"][0]

        # add volume mounts
        worker_container["volumeMounts"] = []
        scheduler_container["volumeMounts"] = []
        worker_spec["volumes"] = []
        scheduler_spec["volumes"] = []
        for volume in self._volumes + self._secrets:
            volume_spec = volume._to_volume_spec()
            volume_mount_spec = volume._to_volume_mount_spec()

            worker_container["volumeMounts"].append(dict(volume_mount_spec))
            scheduler_container["volumeMounts"].append(dict(volume_mount_spec))

            worker_spec["volumes"].append(
                copy.deepcopy(volume_spec)
            )
            scheduler_spec["volumes"].append(
                copy.deepcopy(volume_spec)
            )

        return spec

    @requires_package("dask_kubernetes")
    def start(self):
        from dask_kubernetes.operator import KubeCluster
        from kubernetes import config

        config.load_kube_config()

        spec = self._generate_cluster_spec()
        self._cluster = KubeCluster(
            namespace=self._namespace,
            custom_cluster_spec=spec,
            **self._cluster_kwargs

        )
        self._cluster.adapt(
            minimum=self._minimum_number_of_workers,
            maximum=self._maximum_number_of_workers,
        )
        super().start()


class DaskKubernetesExistingBackend(BaseDaskKubernetesBackend):
    def start(self):
        self._cluster = (
            f"tcp://{self._cluster_name}-scheduler"
            f".{self._namespace}.svc.cluster.local:"
            f"{self._cluster_port}"
        )
        super().start()
    
    def stop(self):
        logger.warning("Cannot stop an existing Kubernetes cluster.")
