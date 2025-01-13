.. |dask_kubernetes_backend|        replace:: :py:class:`~openff.evaluator.backends.dask_kubernetes.DaskKubernetesBackend`
.. |evaluator_server|        replace:: :py:class:`~openff.evaluator.backends.dask_kubernetes.DaskKubernetesBackend`
.. |evaluator_client|        replace:: :py:class:`~openff.evaluator.backends.dask_kubernetes.DaskKubernetesBackend`
.. |pod_resources|        replace:: :py:class:`~openff.evaluator.backends.dask_kubernetes.DaskKubernetesBackend`
.. |compute_resources|        replace:: :py:class:`~openff.evaluator.backends.dask_kubernetes.DaskKubernetesBackend`


Dask Kubernetes Backend
========================

The framework implements a special set of calculation backends which integrate with the ``dask`` `distributed <https://
distributed.dask.org/>`_ and `dask-kubernetes <https://kubernetes.dask.org/en/latest/>`_ libraries.
These backends are designed to run on the `National Research Platform <https://nationalresearchplatform.org/nautilus/>`_
(NRP) and have not been otherwise tested.


Several separate components are required for executing Evaluator on NRP due to the limited user permissions we have:

* a shared filesystem that is accessible by the |evaluator_server| and the |dask_kubernetes_backend|.
  Typically this is constructed with a `PersistentVolumeClaim <https://ucsd-prp.gitlab.io/userdocs/tutorial/storage/>`_.
* a |dask_kubernetes_backend| that can submit tasks to the Kubernetes cluster. This must be initiated locally with NRP.
  The backend must have the PVC mounted.
* an |evaluator_server|, running remotely on a deployment on NRP, that can receive tasks from the local |evaluator_client|.
  This needs to connect to the |dask_kubernetes_backend| to submit tasks to the Kubernetes cluster.
  If permissions are limited as they are on NRP, you may not be able to create the |dask_kubernetes_backend| remotely.
  In that case, you will need a |dask_kubernetes_existing_backend| to connect to an existing KubeCluster.
* the |evaluator_server| port forwarded so ForceBalance can communicate with the |evaluator_server|.


PersistentVolumeClaims in Python
--------------------------------

A PVC can be constructed with `this tutorial <https://ucsd-prp.gitlab.io/userdocs/tutorial/storage/>`_,
or dynamically through Python using the Kubernetes client::

    import time
    from kubernetes import client, config
    from openff.units import unit

    core_v1 = client.CoreV1Api()

    # from https://ucsd-prp.gitlab.io/userdocs/storage/ceph/#currently-available-storageclasses
    storage_class_name = "rook-cephfs-central"

    # required space to request
    storage_space = 1 * unit.gigabytes
    
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
    api_response = core_v1.create_namespaced_persistent_volume_claim(
        namespace=namespace,
        body=pvc
    )
    logger.info(
        f"Created PVC {pvc.metadata.name}. State={api_response.status.phase}"
    )

    # wait for PVC to bind
    timeout = 1000
    end_time = time.time() + timeout
    while time.time() < end_time:
        pvc = core_v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
        if pvc.status.phase == "Bound":
            logger.info(f"PVC '{pvc_name}' is Bound.")
            return pvc_name
        logger.info(f"Waiting for PVC '{pvc_name}' to become Bound. Current phase: {pvc.status.phase}")
        time.sleep(5)


Dask Kubernetes Cluster
-----------------------

The |dask_kubernetes_backend| backend wraps around the dask `Dask KubeCluster <https://kubernetes.dask.org/en/latest/operator_kubecluster.html>`_
class to distribute tasks on Kubernetes::

    # replace with own docker image
    docker_image = "ghcr.io/lilyminium/openff-images:tmp-evaluator-dask-v2"
    cluster_name = "evaluator-cluster"
    namespace = "openforcefield"  # namespace on NRP

    backend = DaskKubernetesBackend(
        cluster_name=cluster_name,
        gpu_resources_per_worker=gpu_resources_per_worker,  # see below
        cpu_resources_per_worker=cpu_resources_per_worker,  # see below
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
        volumes=[volume], # see below
        secrets=[secret], # see below
        annotate_resources=True, # see below
        cluster_kwargs={"resource_timeout": 300}
    )


Specifying pod resources
~~~~~~~~~~~~~~~~~~~~~~~~

Pod resources should be specified using |pod_resources|, which works analogously to |compute_resources|,
but encodes settings for Kubernetes pods. For example::

    from openff.units import unit

    ephemeral_storage = 20 * unit.gigabytes
    memory = 8 * unit.gigabytes

    gpu_resources_per_worker=PodResources(
        minimum_number_of_workers=0,
        maximum_number_of_workers=10,
        number_of_threads=1,
        memory_limit=memory,
        ephemeral_storage_limit=ephemeral_storage,
        number_of_gpus=1,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
    )
    cpu_resources_per_worker=PodResources(
        minimum_number_of_workers=0,
        maximum_number_of_workers=40,
        number_of_threads=1,
        memory_limit=memory,
        ephemeral_storage_limit=ephemeral_storage,
        number_of_gpus=0,
    )


Specifying volumes
~~~~~~~~~~~~~~~~~~

Volumes should be specified as a list of |kubernetes_persistent_volume_claim| objects. For example::

    volume = KubernetesPersistentVolumeClaim(
        name="evaluator-storage",  # `pvc_name`, the name of the PVC
        mount_path="/evaluator-storage",  # where to mount the PVC
    )


Specifying secrets
~~~~~~~~~~~~~~~~~~

Secrets should be specified as a list of |kubernetes_secret| objects. For example::

    secret = KubernetesSecret(
        name="openeye-license",
        secret_name="oe-license",
        mount_path="/secrets/oe_license.txt",
        sub_path="oe_license.txt",
        read_only=True,
    )


This example of mounting an OpenEye license mounts the ``secret_name`` secret
at the ``mount_path`` path in the pod, at the ``sub_path`` path.

.. note::
    
    A secret should first be created in Kubernetes as following
    `the documentation <https://kubernetes.io/docs/tasks/configmap-secret/managing-secret-using-kubectl/#create-a-secret>`_.


Annotating resources
~~~~~~~~~~~~~~~~~~~~

Dask allows you to specify whether tasks require particular
`resources <https://distributed.dask.org/en/latest/resources.html>`_ to be available on the worker used
to execute them. Setting ``annotate_resources=True`` will split tasks into those that can only be
executed on GPU workers, and those that can only be executed on CPU workers.
Simulation protocols such as |openmm_simulation| are executed on GPUs, whereas tasks such as packing boxes
are executed on CPUs. Splitting tasks this way will increase the GPU utilization of GPU workers.

Setting ``annotate_resources=False`` will allow tasks to be executed on any worker.



Dask Kubernetes Existing Backend
--------------------------------

If you are unable to create a |dask_kubernetes_backend| remotely, you can connect to an existing KubeCluster
with the |dask_kubernetes_existing_backend| with the same arguments::

    from openff.evaluator.backends.dask_kubernetes import DaskKubernetesExistingBackend

    backend = DaskKubernetesExistingBackend(
        cluster_name=cluster_name,
        gpu_resources_per_worker=gpu_resources_per_worker,
        cpu_resources_per_worker=cpu_resources_per_worker,
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

Not all of these are important to keep the same, as this cluster simply connects to an
already initialized |dask_kubernetes_backend|. However, the following are important to keep the same:

* ``cluster_name`` -- for connection
* ``namespace`` -- for connection
* ``gpu_resources_per_worker`` -- the `preferred_gpu_toolkit` is important here, although not the number of workers
* ``volumes`` -- the PVC must be mounted
* ``secrets`` -- an OpenEye license would ideally be mounted
* ``annotate_resources`` -- this controls whether or not to split tasks between GPU/CPU workers


Deployment
~~~~~~~~~~

The |evaluator_server| can be deployed remotely on NRP with the following command::

    with backend:
        evaluator_server = EvaluatorServer(
            backend=backend,
            port=port,
            debug=True,
        )
        evaluator_server.start(asynchronous=False)

