.. |build_coordinates_packmol|     replace:: :py:class:`~openff.evaluator.protocols.coordinates.BuildCoordinatesPackmol`

.. |build_smirnoff_system|         replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildSmirnoffSystem`
.. |build_tleap_system|            replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildTLeapSystem`
.. |build_lig_par_gen_system|      replace:: :py:class:`~openff.evaluator.protocols.forcefield.BuildLigParGenSystem`

.. |openmm_energy_minimisation|    replace:: :py:class:`~openff.evaluator.protocols.openmm.OpenMMEnergyMinimisation`
.. |openmm_simulation|             replace:: :py:class:`~openff.evaluator.protocols.openmm.OpenMMSimulation`
.. |openmm_evaluate_energies|      replace:: :py:class:`~openff.evaluator.protocols.openmm.OpenMMEvaluateEnergies`

.. |average_observable|            replace:: :py:class:`~openff.evaluator.protocols.analysis.AverageObservable`
.. |decorrelate_trajectory|        replace:: :py:class:`~openff.evaluator.protocols.analysis.DecorrelateTrajectory`
.. |decorrelate_observables|       replace:: :py:class:`~openff.evaluator.protocols.analysis.DecorrelateObservables`

.. |concatenate_trajectories|      replace:: :py:class:`~openff.evaluator.protocols.reweighting.ConcatenateTrajectories`
.. |concatenate_statistics|        replace:: :py:class:`~openff.evaluator.protocols.reweighting.ConcatenateStatistics`
.. |reweight_observable|           replace:: :py:class:`~openff.evaluator.protocols.reweighting.ReweightObservable`

.. |unpack_stored_simulation_data|        replace:: :py:class:`~openff.evaluator.protocols.storage.UnpackStoredSimulationData`

.. |generate_simulation_protocols|        replace:: :py:meth:`~openff.evaluator.protocols.utils.generate_simulation_protocols`
.. |generate_reweighting_protocols|       replace:: :py:meth:`~openff.evaluator.protocols.utils.generate_reweighting_protocols`
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
base their workflows off of the |generate_simulation_protocols| template.

.. note:: This template currently assumes that a liquid phase property is being computed.

The workflow produced by this template proceeds as follows:

.. rst-class:: spaced-list

    1) 1000 molecules are inserted into a simulation box with an approximate density of 0.95 g / mL using `packmol <http://m3g.iqm.unicamp.br/packmol/home.shtml>`_ (|build_coordinates_packmol|).

    2) the system is parameterized using either the `OpenFF toolkit <#>`_, `TLeap <#>`_ or `LigParGen <#>`_ depending on the force field being employed (|build_smirnoff_system|, |build_tleap_system| or |build_lig_par_gen_system|).

    3) an energy minimization is performed using the default OpenMM energy minimizer (|openmm_energy_minimisation|).

    4) the system is equilibrated by running a short NPT simulation for 100000 steps using a timestep of 2 fs and using the OpenMM simulation engine (|openmm_simulation|).

    5) while the uncertainty in the average observable is greater than the requested tolerance (if specified):

        5a) a longer NPT production simulation is run for 1000000 steps with a timestep of 2 fs and using the OpenMM simulation protocol (|openmm_simulation|) with its default Langevin integrator and Monte Carlo barostat.

        5b) the correlated samples are removed from the simulation outputs and the average value of the observable of interest and its uncertainty are computed by bootstrapping with replacement for 250 iterations (|average_observable|).  See :cite:`2016:chodera` for details of the decorrelation procedure.

        5c) steps 5a) and 5b) are repeated until the uncertainty condition (if applicable) is met.

The decorrelated simulation outputs are then made available ready to be cached by a
:doc:`storage backend <../storage/storagebackend>` (|decorrelate_observables|, |decorrelate_trajectory|).

|reweighting_layer|
-------------------

Properties being estimated using the :doc:`MBAR reweighting <../layers/reweightinglayer>` calculation layer typically
base their workflows off of the |generate_reweighting_protocols| template.

The workflow produced by this template proceeds as follows:

.. rst-class:: spaced-list

    1) for each stored simulation data:

        1a) the cached data is retrieved from disk (|unpack_stored_simulation_data|)

    2) the cached data from is concatenated together to form a single trajectory of configurations and observables (|concatenate_trajectories|, |concatenate_statistics|).

    3) for each stored simulation data:

        3a) the system is parameterized using the force field parameters which were used when originally generating the cached data i.e. one of the reference states (|build_smirnoff_system|, |build_tleap_system| or |build_lig_par_gen_system|).

        3b) the reduced potential of each configuration in the concatenated trajectory is evaluated using the parameterized system (|openmm_evaluate_energies|).

    4) the system is parameterized using the force field parameters with which the property of interest should be calculated using i.e. of the target state (|build_smirnoff_system|, |build_tleap_system| or |build_lig_par_gen_system|) and the reduced potential of each configuration in the concatenated trajectory is evaluated using the parameterized system (|openmm_evaluate_energies|).

        4a) *(optional)* if the observable of interest is a function of the force field parameters it is recomputed using the target state parameters. These recomputed values then replace the original concatenated observables loaded from the cached data.

    5) the reference potentials, target potentials and the joined observables are sub-sampled to only retain equilibrated, uncorrelated samples (|average_observable|, |decorrelate_observables|, |decorrelate_trajectory|). See :cite:`2016:chodera` for details of the decorrelation procedure.

    6) the MBAR method is employed to compute the average value of the observable of interest and its uncertainty at the target state, taking the reference state reduced potentials as input. See :cite:`2018:messerly-a` for the theory behind this approach. An exception is raised if there are not enough effective samples to reweight (|reweight_observable|).

In more specialised cases the |generate_base_reweighting_protocols| template (which |generate_reweighting_protocols| is
built off of) is instead used due to its greater flexibility.

References
----------

.. bibliography:: commonworkflows.bib
    :cited:
    :style: unsrt
