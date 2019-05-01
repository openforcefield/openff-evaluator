General Simulation Practices
============================

Neat Liquid Simulations
-----------------------

The following procedures are followed for neat liquid simulations.

Simulation Box Setup
~~~~~~~~~~~~~~~~~~~~

Simulation boxes are generated using the `solvationtoolkit` module

Energy Minimization
~~~~~~~~~~~~~~~~~~~~
Energy minimizations are performed using the OpenMM `simulation.minimizeEnergy` function, which uses *insert details here* algorithm.


Molecular Dynamics Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Integration parameters: A Langevin Integrator with collision frequency 1 ps^-1 and step size 1 fs

- Barostat: Monte Carlo Barostatting (`simtk.openmm.MonteCarloBarostat`) with a move frequency of 25 steps

- Long-range cutoff: The PME method for calculating long-range electrostatic interactions is used, with a cutoff distance of 0.95 nm.  The same cutoff is used for van der Waals interactions, with a long-range isotropic dispersion correction employed to correct for the truncation of Lennard-Jones interactions outside of the 0.95 nm cutoff.


Equilibration and detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For production simulations, the `detectEquilibration` function from the `pymbar.timeseries` package is used to discard non-equilibrium configurations, and the `subsampleCorrelatedData` function is used to compute the statistical inefficiency **g** and choose uncorrelated samples for analysis

References
~~~~~~~~~~
Solvation Toolkit package: https://github.com/MobleyLab/SolvationToolkit/
