.. |build_coordinates_packmol|    replace:: :py:class:`~openff.evaluator.protocols.coordinates.BuildCoordinatesPackmol`

.. |build_smirnoff_system|        replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildSmirnoffSystem`
.. |build_tleap_system|           replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildTLeapSystem`
.. |build_lig_par_gen_system|     replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildLigParGenSystem`

.. |openmm_energy_minimisation|   replace:: :py:class:`~openff.evaluator.protocols.openmm.OpenMMEnergyMinimisation`
.. |openmm_simulation|            replace:: :py:class:`~openff.evaluator.protocols.openmm.OpenMMSimulation`
.. |openmm_reduced_potentials|    replace:: :py:class:`~openff.evaluator.protocols.openmm.OpenMMReducedPotentials`

.. |extract_average_statistic|              replace:: :py:class:`~openff.evaluator.protocols.analysis.ExtractAverageStatistic`
.. |extract_uncorrelated_statistics_data|   replace:: :py:class:`~openff.evaluator.protocols.analysis.ExtractUncorrelatedStatisticsData`
.. |extract_uncorrelated_trajectory_data|   replace:: :py:class:`~openff.evaluator.protocols.analysis.ExtractUncorrelatedTrajectoryData`

.. |concatenate_trajectories|          replace:: :py:class:`~openff.evaluator.protocols.reweighting.ConcatenateTrajectories`
.. |concatenate_statistics|            replace:: :py:class:`~openff.evaluator.protocols.reweighting.ConcatenateStatistics`
.. |reweight_statistics|               replace:: :py:class:`~openff.evaluator.protocols.reweighting.ReweightStatistics`

.. |unpack_stored_simulation_data|          replace:: :py:class:`~openff.evaluator.protocols.storage.UnpackStoredSimulationData`

.. |generate_base_simulation_protocols|   replace:: :py:meth:`~openff.evaluator.protocols.utils.generate_base_simulation_protocols`
.. |generate_base_reweighting_protocols|  replace:: :py:meth:`~openff.evaluator.protocols.utils.generate_base_reweighting_protocols`

.. |simulation_layer|    replace:: :doc:`Direct Simulation <../layers/simulationlayer>`
.. |reweighting_layer|   replace:: :doc:`MBAR Reweighting <../layers/reweightinglayer>`

Common Workflows
================

As may be expected, most of the workflows used to estimate the physical properties within the framework make use
of very similar workflows. This page aims to document the built-in 'template' workflows from which the more complex
physical property estimation workflows are constructed.

|simulation_layer|
------------------

Properties being estimated using the :doc:`direct simulation <../layers/simulationlayer>` calculation layer typically
base their workflows off of the |generate_base_simulation_protocols| template.

.. note:: This template currently assumes that a liquid phase property is being computed.

The workflow produced by this template proceeds as follows:

.. rst-class:: spaced-list

    1) 1000 molecules are inserted into a simulation box with an approximate density of 0.95 g / mL. property substance
       using packmol (|build_coordinates_packmol|).

    2) the system is parameterized using either the `OpenFF toolkit <#>`_, `TLeap <#>`_ or `LigParGen <#>`_ depending on the force field being employed (|build_smirnoff_system|, |build_tleap_system| or |build_lig_par_gen_system|).

    3) an energy minimization is performed using the default OpenMM energy minimizer (|openmm_energy_minimisation|).

    4) the system is equilibrated by running a short NPT simulation for 100000 steps using a timestep of 2 fs and using the OpenMM simulation engine (|openmm_simulation|).

    5) while the uncertainty in the average observable is greater than the requested tolerance (if specified):

        5a) a longer NPT production simulation is run for 1000000 steps using a timestep of 2 fs and using the OpenMM simulation engine (|openmm_simulation|).

        5b) the correlated samples are removed from the simulation outputs and the average value of the observable of interest are computed by bootstrapping with replacement for 250 iterations (|extract_average_statistic|).  See :cite:`2016:chodera` for details of the decorrelation procedure.

        5c) 5a) and 5b) are repeated until the uncertainty condition is met.

The decorrelated simulation outputs are then made available ready to be cached by a
:doc:`storage backend <../storage/storagebackend>` (|extract_uncorrelated_statistics_data|, |extract_uncorrelated_trajectory_data|).

|reweighting_layer|
-------------------

Properties being estimated using the :doc:`MBAR reweighting <../layers/reweightinglayer>` calculation layer typically
base their workflows off of the |generate_base_reweighting_protocols| template.

The workflow produced by this template proceeds as follows:

.. rst-class:: spaced-list

    1) for each stored simulation data:

        1a) the cached data is retrieved from disk (|unpack_stored_simulation_data|)

        1b) the cached data is subsampled so that the data which will be reweighted is decorrelated (|extract_average_statistic|, |extract_uncorrelated_statistics_data|, |extract_uncorrelated_trajectory_data|). See :cite:`2016:chodera` for details of the decorrelation procedure.

    2) the cached data from is concatenated together to form a single trajectory of configurations and observables (|concatenate_trajectories|, |concatenate_statistics|).

    3) for each stored simulation data:

        3a) the system is parameterized using the force field parameters which were used when originally generating the cached data i.e. one of the reference states (|build_smirnoff_system|, |build_tleap_system| or |build_lig_par_gen_system|).

        3b) the reduced potential of each configuration in the concatenated trajectory is evaluated using the parameterized system (|openmm_reduced_potentials|).

    4) the system is parameterized using the force field parameters with which the property of interest should be calculated using i.e. of the target state (|build_smirnoff_system|, |build_tleap_system| or |build_lig_par_gen_system|) and the reduced potential of each configuration in the concatenated trajectory is evaluated using the parameterized system (|openmm_reduced_potentials|).

    5) the MBAR method is employed to compute the average value of the observable of interest at the target state, taking the reference state reduced potentials as input. See :cite:`2018:messerly-a` for the theory behind this approach. An exception is raised if there are not enough effective samples to reweight (|reweight_statistics|).

References
----------

.. bibliography:: commonworkflows.bib
    :cited:
    :style: unsrt
