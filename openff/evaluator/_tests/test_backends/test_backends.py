import pytest
from openff.units import unit

from openff.evaluator.backends.backends import PodResources


class TestPodResources:

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
            maximum_number_of_workers=1,
        )

    @pytest.fixture
    def cpu_resources(self):
        return PodResources(
            number_of_threads=1,
            number_of_gpus=0,
            memory_limit=5 * unit.terabyte,
            ephemeral_storage_limit=20.0 * unit.megabyte,
            affinity_specification=None,
            minimum_number_of_workers=1,
            maximum_number_of_workers=1,
        )

    def test_podresources_initialization_gpu(self, gpu_resources):
        assert gpu_resources._number_of_threads == 1
        assert gpu_resources._number_of_gpus == 1
        assert gpu_resources._affinity_specification == {
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
        assert gpu_resources._minimum_number_of_workers == 1
        assert gpu_resources._maximum_number_of_workers == 1
        assert gpu_resources._resources == {"GPU": 1, "notGPU": 0}

    def test_to_kubernetes_resources_limits_gpu(self, gpu_resources):
        k8s_resources = gpu_resources._to_kubernetes_resource_limits()
        assert k8s_resources == {
            "cpu": "1",
            "memory": "4.000Gi",
            "ephemeral-storage": "20.000Gi",
            "nvidia.com/gpu": "1",
        }

    def _to_dask_worker_resources_gpu(self, gpu_resources):
        assert gpu_resources._to_dask_worker_resources() == [
            "--resources",
            "GPU=1,notGPU=0",
        ]

    def test_podresources_initialization_cpu(self, cpu_resources):
        assert cpu_resources._number_of_threads == 1
        assert cpu_resources._number_of_gpus == 0
        assert cpu_resources._affinity_specification == {}
        assert cpu_resources._minimum_number_of_workers == 1
        assert cpu_resources._maximum_number_of_workers == 1
        assert cpu_resources._resources == {"GPU": 0, "notGPU": 1}

    def test_to_kubernetes_resources_limits_cpu(self, cpu_resources):
        k8s_resources = cpu_resources._to_kubernetes_resource_limits()
        assert k8s_resources == {
            "cpu": "1",
            "memory": "5000.000Gi",
            "ephemeral-storage": "0.020Gi",
        }

    def _to_dask_worker_resources_cpu(self, cpu_resources):
        assert cpu_resources._to_dask_worker_resources() == [
            "--resources",
            "GPU=0,notGPU=1",
        ]
