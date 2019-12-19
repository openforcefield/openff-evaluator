#!/usr/bin/env python3
import argparse
import logging

from propertyestimator.backends import DaskLocalCluster, ComputeResources
from propertyestimator.server import EvaluatorServer
from propertyestimator.utils import setup_timestamp_logging


def validate_inputs(n_workers, cpus_per_worker, gpus_per_worker):
    """Validate the command line inputs.

    Parameters
    ----------
    n_workers:
        The number of compute workers to spawn.
    cpus_per_worker: int
        The number CPUs each worker should have access to. The server will
        consume a total of `n_workers * cpus_per_worker` CPUs.
    gpus_per_worker: int
        The number GPUs each worker should have access to. The server will
        consume a total of `n_workers * cpus_per_worker` GPUs.
    """
    if n_workers <= 0:
        raise ValueError("The number of workers must be greater than 0")
    if cpus_per_worker <= 0:
        raise ValueError("The number of CPUs per worker must be greater than 0")
    if gpus_per_worker < 0:
        raise ValueError(
            "The number of GPUs per worker must be greater than or equal to 0"
        )

    if 0 < gpus_per_worker != cpus_per_worker:

        raise ValueError(
            "The number of GPUs per worker must match the number of "
            "CPUs per worker."
        )


def main(n_workers, cpus_per_worker, gpus_per_worker):
    """Launch an evaluator server.

    Parameters
    ----------
    n_workers:
        The number of compute workers to spawn.
    cpus_per_worker: int
        The number CPUs each worker should have access to. The server will
        consume a total of `n_workers * cpus_per_worker` CPUs.
    gpus_per_worker: int
        The number GPUs each worker should have access to. The server will
        consume a total of `n_workers * cpus_per_worker` GPUs.
    """

    validate_inputs(n_workers, cpus_per_worker, gpus_per_worker)

    # Set up logging.
    setup_timestamp_logging()
    logger = logging.getLogger()

    # Set up a backend to run the calculations on with the requested resources.
    worker_resources = ComputeResources(
        number_of_threads=cpus_per_worker,
        number_of_gpus=gpus_per_worker,
    )

    calculation_backend = DaskLocalCluster(
        number_of_workers=n_workers, resources_per_worker=worker_resources
    )

    with calculation_backend:

        # Create an estimation server which will run the calculations.
        logger.info(
            f"Starting the server with {n_workers} workers, each with "
            f"{cpus_per_worker} CPUs and {gpus_per_worker} GPUs."
        )

        server = EvaluatorServer(calculation_backend=calculation_backend)
        server.start(asynchronous=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Start a PropertyEstimatorServer with a "
        "specified number of workers, each with "
        "access to the specified compute resources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--workers",
        "-nwork",
        type=int,
        help="The number of compute workers to spawn.",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--cpus_per_worker",
        "-ncpus",
        type=int,
        help="The number CPUs each worker should have access to. "
        "The server will consume a total of `nwork * ncpus` CPU's.",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--gpus_per_worker",
        "-ngpus",
        type=int,
        help="The number CPUs each worker should have access to. "
        "The server will consume a total of `nwork * ngpus` GPU's.",
        required=False,
        default=1,
    )

    args = parser.parse_args()

    main(args.workers, args.cpus_per_worker, args.gpus_per_worker)
