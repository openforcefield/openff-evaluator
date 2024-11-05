import pytest

from openff.evaluator.backends.backends import PodResources
from openff.evaluator.backends.dask_kubernetes import (
    DaskKubernetesBackend,
    KubernetesEmptyDirVolume,
    KubernetesSecret
)
from openff.units import unit

class TestDaskKubernetesBackend:

    @pytest.fixture
    def calculation_backend(self):
        volume = KubernetesEmptyDirVolume(
            name="evaluator-storage",
            mount_path="/evaluator-storage",
        )
        secret = KubernetesSecret(
            name="openeye-license",
            secret_name="oe-license-feb-2024",
            mount_path="/secrets/oe_license.txt",
            sub_path="oe_license.txt",
        )
        calculation_backend = DaskKubernetesBackend(
            cluster_name="evaluator",
            minimum_number_of_workers=0,
            maximum_number_of_workers=3,
            resources_per_worker=PodResources(
                number_of_threads=1,
                memory_limit=4 * unit.gigabytes,
                ephemeral_storage_limit=20 * unit.gigabytes,
                number_of_gpus=1,
            ),
            image="ghcr.io/lilyminium/openff-images:evaluator-0.4.10-kubernetes-dask-v0",
            namespace="openforcefield",
            env={
                "OE_LICENSE": "/secrets/oe_license.txt",
                # daemonic processes are not allowed to have children
                "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
            },
            volumes=[volume],
            secrets=[secret],
        )
        return calculation_backend

    def test_creation(self, calculation_backend):
        calculation_backend.start()
        assert calculation_backend._cluster.name == "evaluator"
        logs = calculation_backend._cluster.get_logs()
        print(logs)
        calculation_backend.stop()

    def test_generate_spec(self, calculation_backend):
        spec = calculation_backend._generate_cluster_spec()

        expected_spec = {
            "apiVersion": "kubernetes.dask.org/v1",
            "kind": "DaskCluster",
            "metadata": {"name": "evaluator"},
            "spec": {
                "idleTimeout": 0,
                "worker": {
                    "replicas": 0,
                    "spec": {
                        "containers": [
                            {
                                "name": "worker",
                                "image": "ghcr.io/lilyminium/openff-images:evaluator-0.4.10-kubernetes-dask-v0",
                                "args": [
                                    "dask-worker",
                                    "--name",
                                    "$(DASK_WORKER_NAME)",
                                    "--dashboard",
                                    "--dashboard-address",
                                    "8788",
                                ],
                                "env": [
                                    {"name": "OE_LICENSE", "value": "/secrets/oe_license.txt"},
                                    {
                                        "name": "DASK_DISTRIBUTED__WORKER__DAEMON",
                                        "value": "False",
                                    },
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "1",
                                        "memory": "4Gi",
                                        "ephemeral-storage": "20Gi",
                                        "nvidia.com/gpu": "1",
                                    },
                                    "limits": {
                                        "cpu": "1",
                                        "memory": "4Gi",
                                        "ephemeral-storage": "20Gi",
                                        "nvidia.com/gpu": "1",
                                    },
                                },
                                "ports": [
                                    {
                                        "name": "http-dashboard",
                                        "containerPort": 8788,
                                        "protocol": "TCP",
                                    }
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "evaluator-storage",
                                        "mountPath": "/evaluator-storage",
                                        "readOnly": False,
                                    },
                                    {
                                        "name": "openeye-license",
                                        "mountPath": "/secrets/oe_license.txt",
                                        "readOnly": False,
                                        "subPath": "oe_license.txt",
                                    },
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "evaluator-storage", "emptyDir": {}},
                            {
                                "name": "openeye-license",
                                "secret": {"secretName": "oe-license-feb-2024"},
                            },
                        ],
                    },
                },
                "scheduler": {
                    "spec": {
                        "containers": [
                            {
                                "name": "scheduler",
                                "image": "ghcr.io/lilyminium/openff-images:evaluator-0.4.10-kubernetes-dask-v0",
                                "args": ["dask-scheduler", "--host", "0.0.0.0"],
                                "env": [
                                    {"name": "OE_LICENSE", "value": "/secrets/oe_license.txt"},
                                    {
                                        "name": "DASK_DISTRIBUTED__WORKER__DAEMON",
                                        "value": "False",
                                    },
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "1",
                                        "memory": "4Gi",
                                        "ephemeral-storage": "20Gi",
                                    },
                                    "limits": {
                                        "cpu": "1",
                                        "memory": "4Gi",
                                        "ephemeral-storage": "20Gi",
                                    },
                                },
                                "ports": [
                                    {
                                        "name": "tcp-comm",
                                        "containerPort": 8786,
                                        "protocol": "TCP",
                                    },
                                    {
                                        "name": "http-dashboard",
                                        "containerPort": 8787,
                                        "protocol": "TCP",
                                    },
                                ],
                                "readinessProbe": {
                                    "httpGet": {"port": "http-dashboard", "path": "/health"},
                                    "initialDelaySeconds": 0,
                                    "periodSeconds": 1,
                                    "timeoutSeconds": 300,
                                },
                                "livenessProbe": {
                                    "httpGet": {"port": "http-dashboard", "path": "/health"},
                                    "initialDelaySeconds": 15,
                                    "periodSeconds": 20,
                                },
                                "volumeMounts": [
                                    {
                                        "name": "evaluator-storage",
                                        "mountPath": "/evaluator-storage",
                                        "readOnly": False,
                                    },
                                    {
                                        "name": "openeye-license",
                                        "mountPath": "/secrets/oe_license.txt",
                                        "readOnly": False,
                                        "subPath": "oe_license.txt",
                                    },
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "evaluator-storage", "emptyDir": {}},
                            {
                                "name": "openeye-license",
                                "secret": {"secretName": "oe-license-feb-2024"},
                            },
                        ],
                    },
                    "service": {
                        "type": "ClusterIP",
                        "selector": {
                            "dask.org/cluster-name": "evaluator",
                            "dask.org/component": "scheduler",
                        },
                        "ports": [
                            {
                                "name": "tcp-comm",
                                "protocol": "TCP",
                                "port": 8786,
                                "targetPort": "tcp-comm",
                            },
                            {
                                "name": "http-dashboard",
                                "protocol": "TCP",
                                "port": 8787,
                                "targetPort": "http-dashboard",
                            },
                        ],
                    },
                },
            },
        }


        assert spec == expected_spec


