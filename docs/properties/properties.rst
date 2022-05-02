.. |average_observable|           replace:: :py:class:`~openff.evaluator.protocols.analysis.AverageObservable`
.. |average_dielectric_constant|  replace:: :py:class:`~openff.evaluator.protocols.analysis.AverageDielectricConstant`

.. |reweight_observable|          replace:: :py:class:`~openff.evaluator.protocols.reweighting.ReweightObservable`
.. |reweight_dielectric_constant| replace:: :py:class:`~openff.evaluator.protocols.reweighting.ReweightDielectricConstant`

.. |add_values|                   replace:: :py:class:`~openff.evaluator.protocols.miscellaneous.AddValues`
.. |subtract_values|              replace:: :py:class:`~openff.evaluator.protocols.miscellaneous.SubtractValues`
.. |divide_value|                 replace:: :py:class:`~openff.evaluator.protocols.miscellaneous.DivideValue`
.. |multiply_value|                 replace:: :py:class:`~openff.evaluator.protocols.miscellaneous.MultiplyValue`
.. |weight_by_mole_fraction|      replace:: :py:class:`~openff.evaluator.protocols.miscellaneous.WeightByMoleFraction`

.. |simulation_layer|    replace:: :doc:`Direct Simulation <../layers/simulationlayer>`
.. |reweighting_layer|   replace:: :doc:`MBAR Reweighting <../layers/reweightinglayer>`

Physical Properties
===================

A core philosophy of this framework is that users should be able to seamlessly curate data sets of physical properties
and then estimate that data set using computational methods without significant user intervention and using sensible,
well validated workflows.

This page aims to provide an overview of which physical properties are supported by the framework and how they
are computed using the different :doc:`calculation layers <../layers/calculationlayers>`.

In this document :math:`\left<X\right>` will be used to denote the ensemble average of an observable :math:`X`.

Density
"""""""

The density (:math:`\rho`) is computed according to

.. math::

    \rho = \left<\dfrac{M}{V}\right>

where :math:`M` and :math:`V` are the total molar mass and volume the system respectively.

|simulation_layer|
*******************

The density is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>` without
modification. The estimation of liquid densities is assumed.

|reweighting_layer|
*******************

The density is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>` without
modification. The estimation of liquid densities is assumed.

Dielectric Constant
"""""""""""""""""""

The dielectric constant (:math:`\varepsilon`) is computed from the fluctuations in a systems dipole moment (see Equation
7 of :cite:`2002:glattli`) according to:

.. math::

    \varepsilon = 1 + \dfrac{\left<\vec{\mu}^2\right> - \left<\vec{\mu}\right>^2} {3\varepsilon_0\left<V\right>k_bT}

where :math:`\vec{\mu}`, :math:`V` are the systems dipole moment and volume respectively, :math:`k_b` the Boltzmann
constant, :math:`T` the temperature, and :math:`\varepsilon_0` the permittivity of free space.

.. note:: In *v0.2.2* and earlier of the framework the variance was computed as
          :math:`\left<{\left(\vec{\mu} - \left<\vec{\mu}\right>\right)}^2\right>` in order to match the
          `mdtraj <http://mdtraj.org/>`_ implementation which has been used in previous studies by the OpenFF
          Consortium (see for example :cite:`2015:beauchamp`). The two approaches should be numerically
          indistinguishable however.

|simulation_layer|
******************

The dielectric is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>`
which has been modified to use the specialized |average_dielectric_constant| protocol in place of the default
|average_observable| protocol. The estimation of liquid dielectric constants is assumed.

|reweighting_layer|
*******************

The dielectric is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>`
which has been modified to use the specialized |reweight_dielectric_constant| protocol in place of the default
|reweight_observable| protocol. It should be noted that the |reweight_dielectric_constant| protocol employs
bootstrapping to compute the uncertainty in the average dielectric constant, rather than attempting to propagate
uncertainties in the average dipole moments and volumes. The estimation of liquid dielectric constants is assumed.

Enthalpy of Vaporization
""""""""""""""""""""""""

The enthalpy of vaporization :math:`\Delta H_{vap}` (see :cite:`2011:wang`) can be computed according to

.. math::

    \Delta H_{vap} = \left<H_{gas}\right> - \left<H_{liquid}\right> = \left<E_{gas}\right> - \left<E_{liquid}\right> + p\left(\left<V_{gas}\right>-\left<V_{liquid}\right>\right)

where :math:`H`, :math:`E`, and :math:`V` are the enthalpy, total energy and volume respectively.

Under the assumption that :math:`V_{gas} >> V_{liquid}` and that the gas is ideal the above expression can be simplified
to

.. math::

    \Delta H_{vap} = \left<U_{gas}\right> - \left<U_{liquid}\right> + RT

where :math:`U` is the potential energy, :math:`T` the temperature and :math:`R` the universal gas constant. This
simplified expression is computed by default by this framework.

|simulation_layer|
******************

.. rst-class:: spaced-list

    - **Liquid phase**: The potential energy of the liquid phase is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>`,
      and divided by the number of molecules in the simulation box using the ``divisor`` input of the
      |average_observable| protocol.

    - **Gas phase**: The potential energy of the gas phase is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>`,
      which has been modified so that

        - the simulation box only contains a single molecule.
        - all periodic boundary conditions have been disabled.
        - all simulations are performed in the NVT ensemble.
        - the production simulation is run for 15000000 steps at a time (rather than 1000000 steps).
        - all simulations are run using the OpenMM reference platform (CPU only) regardless of whether a GPU is
          available. This is fastest platform to use when simulating a single molecule in vacuum with OpenMM.

The final enthalpy is then computed by subtracting the gas potential energy from the liquid potential energy
(|subtract_values|) and adding the :math:`RT` term (|add_values|). Uncertainties are propagated through the subtraction
by the normal means using the `uncertainties <https://pythonhosted.org/uncertainties/>`_ package.

|reweighting_layer|
*******************

.. rst-class:: spaced-list

    - **Liquid phase**: The potential energy of the liquid phase is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>`,
      and divided by the number of molecules in the simulation box using an extra |divide_value| protocol.

    - **Gas phase**: The potential energy of the gas phase is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>`,
      which has been modified so that all periodic boundary conditions have been disabled.

The final enthalpy is then computed by subtracting the gas potential energy from the liquid potential energy
(|subtract_values|) and adding the :math:`RT` term (|add_values|). Uncertainties are propagated through the subtraction
by the normal means using the `uncertainties <https://pythonhosted.org/uncertainties/>`_ package.

Enthalpy of Mixing
""""""""""""""""""

The enthalpy of mixing :math:`\Delta H_{mix}\left(x_0, \cdots, x_{M-1}\right)` for a system of :math:`M` components
is computed according to

.. math::

    \Delta H_{mix}\left(x_0, \cdots, x_{M-1}\right) = \dfrac{\left<H_{mix}\right>}{N_{mix}} - \sum_i^M x_i \dfrac{\left<H_i\right>}{N_i}

where :math:`H_{mix}` is the enthalpy of the full mixture, and :math:`H_i`, :math:`x_i` are the enthalpy and the mole
fraction of component :math:`i` respectively. :math:`N_{mix}` and :math:`N_i` are the total number of molecules used in
the full mixture simulations and the simulations of each individual component respectively.

When re-weighting cached data to compute :math:`H_{mix}` we make the approximation that the kinetic energy contributions
cancel out between the mixture and each of the components, and hence can be computed by only re-weighting the NPT
reduced potential:

.. math::

    \Delta H_{mix}\left(x_0, \cdots, x_{M-1}\right) \approx \dfrac{1}{\beta} \left( \dfrac{\left<u_{mix}\right>}{N_{mix}} - \sum_i^M x_i \dfrac{\left<u_i\right>}{N_i} \right)

where :math:`u \equiv \beta \left(U + pV\right)` is the NPT reduced potential, :math:`U` the potential energy,
:math:`p` the pressure and :math:`V` the volume.

|simulation_layer|
******************

.. rst-class:: spaced-list

    - **Mixture**: The enthalpy of the full mixture is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>`
      and divided by the number of molecules in the simulation box using the ``divisor`` input of the
      |average_observable| protocol.

    - **Components**: The enthalpy of each of the components is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>`,
      divided by the number of molecules in the simulation box using the ``divisor`` input of the
      |average_observable| protocol, and weighted by their mole fraction *in the mixture simulation box* using
      the |weight_by_mole_fraction| protocol.

The final enthalpy is then computed by summing the component enthalpies (|add_values|) and subtracting these from
the mixture enthalpy (|subtract_values|). Uncertainties are propagated through the summation and subtraction by the
normal means using the `uncertainties <https://pythonhosted.org/uncertainties/>`_ package.

|reweighting_layer|
*******************

.. rst-class:: spaced-list

    - **Mixture**: The reduced potential of the full mixture is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>`
      and divided by the number of molecules in the reweighting box using an extra |divide_value| protocol.

    - **Components**: The reduced potential of each of the components is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>`,
      divided by the number of molecules in the reweighting box using an extra |divide_value| protocol, and weighted by
      their mole fraction using the |weight_by_mole_fraction| protocol.

The final enthalpy is then computed by summing the component enthalpies (|add_values|), subtracting these from
the mixture enthalpy (|subtract_values|), and multiplying by :math:`1 / \beta` (|multiply_value|). Uncertainties are propagated
by the normal means using the `uncertainties <https://pythonhosted.org/uncertainties/>`_ package.

Excess Molar Volume
"""""""""""""""""""

The excess molar volume :math:`\Delta V_{excess}\left(x_0, \cdots, x_{M-1}\right)` for a system of :math:`M` components
is computed according to

.. math::

    \Delta V_{excess}\left(x_0, \cdots, x_{M-1}\right) = N_A \left( \dfrac{\left<V_{mix}\right>}{N_{mix}} - \sum_i^M x_i \dfrac{\left<V_i\right>}{N_i} \right)

where :math:`V_{mix}` is the volume of the full mixture, and :math:`V_i`, :math:`x_i` are the volume and the mole
fraction of component :math:`i` respectively. :math:`N_{mix}` and :math:`N_i` are the total number of molecules used in
the full mixture simulations and the simulations of each individual component respectively, and :math:`N_A` is the
Avogadro constant.

|simulation_layer|
******************

.. rst-class:: spaced-list

    - **Mixture**: The molar volume of the full mixture is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>`
      and divided by the molar number of molecules in the simulation box using the ``divisor`` input of the
      |average_observable| protocol.

    - **Components**: The molar volume of each of the components is estimated using the default :ref:`simulation workflow <properties/commonworkflows:|simulation_layer|>`,
      divided by the molar number of molecules in the simulation box using the ``divisor`` input of the
      |average_observable| protocol, and weighted by their mole fraction *in the mixture simulation box* using
      the |weight_by_mole_fraction| protocol.

The final excess molar volume is then computed by summing the component molar volumes (|add_values|) and subtracting these from
the mixture molar volume (|subtract_values|). Uncertainties are propagated through the summation and subtraction by the
normal means using the `uncertainties <https://pythonhosted.org/uncertainties/>`_ package.

|reweighting_layer|
*******************

.. rst-class:: spaced-list

    - **Mixture**: The enthalpy of the full mixture is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>`
      and divided by the molar number of molecules in the reweighting box using an extra |divide_value| protocol.

    - **Components**: The enthalpy of each of the components is estimated using the default :ref:`reweighting workflow <properties/commonworkflows:|reweighting_layer|>`,
      divided by the molar number of molecules in the reweighting box using an extra |divide_value| protocol, and weighted by
      their mole fraction using the |weight_by_mole_fraction| protocol.

The final enthalpy is then computed by summing the component enthalpies (|add_values|) and subtracting these from
the mixture enthalpy (|subtract_values|). Uncertainties are propagated through the summation and subtraction by the
normal means using the `uncertainties <https://pythonhosted.org/uncertainties/>`_ package.

Solvation Free Energies
"""""""""""""""""""""""

Solvation free energies are currently computed using the `Yank <http://getyank.org/>`_ free energy package using direct
molecular simulations. By default the calculations attempt to use 2000 solvent molecules, and the alchemical lambda
spacings are selected using the built-in 'trailblazing' algorithm.

See the `Yank <http://getyank.org/latest/>`_ documentation for more details.

Host-Guest Binding Free Energy
""""""""""""""""""""""""""""""

.. warning:: The computation of this property is still in beta. Users are heavily recommended to validate any
             calculations involving this property.

Host-guest binding free energies are currently computed using the attach-pull-release (APR) method
:cite:`2019:slochower` through integration with the `pAPRika <https://github.com/slochower/pAPRika>`_ framework.

References
""""""""""

.. bibliography:: properties.bib
    :cited:
    :style: unsrt
