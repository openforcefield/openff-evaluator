.. |openmm_gradient_potentials|    replace:: :py:class:`~openff.evaluator.protocols.openmm.OpenMMGradientPotentials`

Gradients
=========

A most fundamental feature of this framework is its ability to rapidly compute the gradients of physical properties with
respect to the force field parameters used to estimate them.

Theory
------

The framework currently employs the central finite difference approach to computing gradients:

.. math::

    \dfrac {d \left<X \left( \theta \right) \right>}{d \theta_i} = \dfrac { \left<X \left( \theta_i + h \right) \right> - \left<X \left( \theta_i - h \right) \right> }{ 2 h}

where :math:`\left<X\right>` is used to denote the ensemble average of an observable :math:`X`, :math:`\theta_i` is
the force field parameter of interest, and by default :math:`h = 1 \times 10^{-4} \times \theta_i`. Although more
expensive than computing either the forward or backwards derivative, the central difference method should give a more
accurate estimate of the gradient at the minima, maxima and transition points.

Rather than running an entirely new simulation to compute the values of :math:`\left<X \left( \theta_i + h \right) \right>`
and :math:`\left<X \left( \theta_i - h \right) \right>`, these values are directly estimated using the MBAR reweighting
method :cite:`2008:shirts`, :cite:`2018:messerly-b`. This approach has several advantages:

.. rst-class:: spaced-list

- there is a convenient cancellation of errors when computing the finite difference as the average value of the
  observables at the perturbed parameters are computed from the same set of configurations and hence have errors which
  are highly correlated. This thus avoids the need to run prohibitively long simulations to compute the average
  observables to within a low enough error to produce meaningful differences between similar numbers.

- the reduced potentials of the configurations that the observable of interest was computed from can be rapidly
  re-evaluated by only re-computing the energy terms which have changed upon perturbing the parameters (see the
  |openmm_gradient_potentials| protocol).

References
----------

.. bibliography:: gradients.bib
    :cited:
    :style: unsrt
