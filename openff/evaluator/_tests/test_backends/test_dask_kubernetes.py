import pathlib

import pytest
import yaml
from openff.utilities.utilities import get_data_dir_path

from openff.evaluator.backends.backends import PodResources
from openff.evaluator.backends.dask_kubernetes import (
    DaskKubernetesBackend,
    KubernetesEmptyDirVolume,
    KubernetesSecret,
)


class TestDaskKubernetesBackend:
    @pytest.fixture
    def gpu_resources(self):
        node_affinity = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "nvidia.com/cuda.runtime.major",
                                    "operator": "In",
                                    "values": ["12"],
                                },
                                {
                                    "key": "nvidia.com/cuda.runtime.minor",
                                    "operator": "In",
                                    "values": ["4"],
                                },
                            ]
                        }
                    ]
                }
            }
        }
        return PodResources(
            number_of_threads=1,
            number_of_gpus=1,
            affinity_specification=node_affinity,
            minimum_number_of_workers=1,
            maximum_number_of_workers=10,
        )

    @pytest.fixture
    def cpu_resources(self):
        return PodResources(
            number_of_threads=1,
            number_of_gpus=0,
            affinity_specification=None,
            maximum_number_of_workers=20,
        )

    @pytest.fixture
    def calculation_backend(self, gpu_resources, cpu_resources):
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
            gpu_resources_per_worker=gpu_resources,
            cpu_resources_per_worker=cpu_resources,
            cluster_name="evaluator",
            image="ghcr.io/lilyminium/openff-images:evaluator-0.4.10-kubernetes-dask-v0",
            namespace="openforcefield",
            env={
                "OE_LICENSE": "/secrets/oe_license.txt",
                # daemonic processes are not allowed to have children
                "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
                "DASK_LOGGING__DISTRIBUTED": "debug",
                "DASK__TEMPORARY_DIRECTORY": "/evaluator-storage",
            },
            volumes=[volume],
            secrets=[secret],
        )
        return calculation_backend

    def test_no_initialization_without_volumes(self, gpu_resources):
        with pytest.raises(ValueError, match="No volumes specified"):
            DaskKubernetesBackend(
                gpu_resources_per_worker=gpu_resources,
                cluster_name="evaluator",
                image="ghcr.io/lilyminium/openff-images:evaluator-0.4.10-kubernetes-dask-v0",
                namespace="openforcefield",
                env={
                    "OE_LICENSE": "/secrets/oe_license.txt",
                    # daemonic processes are not allowed to have children
                    "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
                    "DASK_LOGGING__DISTRIBUTED": "debug",
                    "DASK__TEMPORARY_DIRECTORY": "/evaluator-storage",
                },
            )

    def test_no_initialization_without_resources(self):
        with pytest.raises(ValueError, match="must be specified"):
            DaskKubernetesBackend()

    def test_generate_volume_specifications(self, calculation_backend):
        volume_mounts, volumes = calculation_backend._generate_volume_specifications()
        assert volume_mounts == [
            {
                "name": "evaluator-storage",
                "mountPath": "/evaluator-storage",
                "readOnly": False,
            },
            {
                "name": "openeye-license",
                "mountPath": "/secrets/oe_license.txt",
                "subPath": "oe_license.txt",
                "readOnly": True,
            },
        ]

        assert volumes == [
            {
                "name": "evaluator-storage",
                "emptyDir": {},
            },
            {
                "name": "openeye-license",
                "secret": {
                    "secretName": "oe-license-feb-2024",
                },
            },
        ]

    def test_generate_worker_spec(self, calculation_backend):
        data_directory = pathlib.Path(
            get_data_dir_path("test/kubernetes", "openff.evaluator")
        )
        reference_file = data_directory / "dask_worker_spec.yaml"

        worker_spec = calculation_backend._generate_worker_spec(
            calculation_backend._other_resources["cpu"]
        )
        with open(reference_file, "r") as file:
            reference_spec = yaml.safe_load(file)

        assert worker_spec == reference_spec

    def test_generate_cluster_spec(self, calculation_backend):
        cluster_spec = calculation_backend._generate_cluster_spec()

        data_directory = pathlib.Path(
            get_data_dir_path("test/kubernetes", "openff.evaluator")
        )
        reference_file = data_directory / "dask_cluster_spec.yaml"
        with open(reference_file, "r") as file:
            reference_spec = yaml.safe_load(file)

        assert cluster_spec == reference_spec

    @pytest.mark.skip(reason="Currently only works with existing kubectl credentials.")
    def test_start(self, calculation_backend):
        calculation_backend.start()
