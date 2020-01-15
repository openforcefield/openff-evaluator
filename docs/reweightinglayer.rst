The ``ReweightingLayer``
========================

The ``ReweightingLayer`` is a calculation which employs molecular simulation to estimate data sets of physical
properties. It inherits the ``WorkflowflowLayer``, and makes use of per property workflows defined in the
``ReweightingLayerSchema`` to reweight any appropriate cached data found on disk.
