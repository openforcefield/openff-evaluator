Architecture
============

The evaluator framework is designed to scale, from running calculations on a local laptop up to larger supercomputers.

Written as a client server architecture. The client APIs allow users to curate data sets, design workflows for
estimating the properties within those data sets, and performing analysis on any calculation results.

.. figure:: _static/img/architecture.svg
    :align: center
    :width: 85%

    Lorem Ipsum

These are designed to be used on even modest hardware. Once the data sets to estimate and the schemas to use to estimate
them have been constructed on the client, they are sent to the sever.

The server is designed to run on the available compute resources, this is scalable from a single machine, up to larger
supercomputers. The server interacts with the resource it is running on via calculation backends to parallelize tasks
across the available resources and nodes and storage backends for caching the results of calculations.

The server is responsible for automatically employing the available calculation approach to the data set.