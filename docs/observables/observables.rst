.. |observable|              replace:: :py:class:`~openff.evaluator.utils.observables.Observable`
.. |observable_array|        replace:: :py:class:`~openff.evaluator.utils.observables.ObservableArray`
.. |observable_frame|        replace:: :py:class:`~openff.evaluator.utils.observables.ObservableFrame`
.. |observable_type|         replace:: :py:class:`~openff.evaluator.utils.observables.ObservableType`

.. |observable_array_join|   replace:: :py:meth:`~openff.evaluator.utils.observables.ObservableArray.join`
.. |observable_array_subset| replace:: :py:meth:`~openff.evaluator.utils.observables.ObservableArray.subset`
.. |observable_frame_join|   replace:: :py:meth:`~openff.evaluator.utils.observables.ObservableFrame.join`
.. |observable_frame_subset| replace:: :py:meth:`~openff.evaluator.utils.observables.ObservableFrame.subset`

.. |parameter_gradient|      replace:: :py:class:`~openff.evaluator.forcefield.ParameterGradient`
.. |parameter_gradient_key|  replace:: :py:class:`~openff.evaluator.forcefield.ParameterGradientKey`

.. |quantity|                replace:: :py:class:`~openff.evaluator.utils.units.Quantity`
.. |measurement|             replace:: :py:class:`~openff.evaluator.utils.units.Measurement`

.. |float|                   replace:: :py:class:`~float`
.. |int|                     replace:: :py:class:`~int`

Observables
===========

A key feature of this framework is its ability to compute the gradients of physical properties with respect to the
force field parameters used to estimate them. This requires the framework be able to, internally, be able to not only
track the gradients of all quantities which combine to yield the final observable of interest, but to also be able to
propagate the gradients of those composite quantities through to the final value.

The framework offers three such objects to this end (|observable|, |observable_array| and |observable_frame| objects)
which will be covered in this document.

.. note:: In future versions of the framework the objects described here will likely be at least in part deprecated
          in favour of using full automatic differentiation libraries such as `jax <https://github.com/google/jax>`_.
          Supporting these libraries will take a large re-write of the framework however, as well as full support
          between differentiable simulation engines like `timemachine <https://github.com/proteneer/timemachine>`_ and
          the OpenFF toolkit. As such, these objects are implemented as stepping stones which can be gently phased out
          while working towards that larger, more modern goal.

Observable Objects
------------------

The base object used to track observables is the |observable| object. It stores the average value, the standard error
in the value and the gradient of the value with respect to force field parameters of interest.

Currently the value and error are internally stored in a composite |measurement| object, which themselves wrap around
the `uncertainties <https://pythonhosted.org/uncertainties/>`_ package. This allows uncertainties to be automatically
propagated through operations without the need for user intervention.

.. note:: Although uncertainties are automatically propagated, it is still up to property estimation workflow authors to
          ensure that such propagation (assuming a Gaussian error model) is appropriate. An alternative, which is
          employed throughout the framework is to make use of the bootstrapping technique.

Gradients are stored in a list as |parameter_gradient| gradient objects, which store both the floating value of the
gradient alongside an identifying |parameter_gradient_key|.

Supported Operations
""""""""""""""""""""

.. rst-class:: spaced-list

    - **+** and **-**: |observable| objects can be summed with and subtracted from other |observable| objects,
      |quantity| objects, floats or integers. When two |observable| objects are summed / subtracted, their gradients are
      combined by summing / subtracting also. When an |observable| is summed / subtracted with a |quantity|,
      |float| or |int| object it is assumed that these objects do not depend on any force field parameters.

    - **\***: |observable| objects may be multiplied by other |observable| objects, |quantity| objects, and |float|
      or |int| objects. When two |observable| objects are multiplied their gradients are propagated using the product
      rule. When an |observable| is multiplied by a |quantity|, |float| or |int| object it is assumed that these
      objects do not depend on any force field parameters.

    - **/**: |observable| objects may be divided by other |observable| objects, |quantity| objects, and |float| or
      |int| objects. Gradients are propagated through the division using the quotient rule. When an |observable| is
      divided by a |quantity|, |float| or |int| object (or when these objects are divided by an |observable| object)
      it is assumed that these objects do not depend on any force field parameters.

In all cases two |observable| objects can only be operated on provided the contain gradient information with respect
to the same set of force field parameters.

Observable Arrays
-----------------

An extension of the |observable| object is the |observable_array| object. Unlike an |observable|, an |observable_array| 
object does not contain error information, but rather the value it stores and the gradients of that value should be a 
numpy array with ``shape=(n_data_points, n_dimensions)``. It is designed to store information such as the potential 
energy evaluated at each configuration sampled during a simulation, as well as the gradient of the potential, which can
then be ensemble averaged using a fluctuation formula to propagate the gradients through to the average.

Like with |observable| objects, gradients are stored in a list as |parameter_gradient| gradient objects. The length
of the gradients is required to match the length of the value array.

|observable_array| objects may be concatenated together using their |observable_array_join| method or sub-sampled using
their |observable_array_subset| method.

Supported Operations
""""""""""""""""""""

The |observable_array| object supports the same operations as the |observable| object, whereby all operations are 
applied elementwise to the stored arrays. 

Observable Frames
-----------------

An |observable_frame| is a wrapper around a collection of |observable_array| which contain the types of observable
specified by the |observable_type| enum. It behaves as a dictionary which can take either an |observable_type| or
a string value of an |observable_type| as an index.

Like an |observable_array|, observable frames may be concatenated together using their |observable_frame_join| method 
or sub-sampled using their |observable_frame_subset| method.

Supported Operations
""""""""""""""""""""

No operations are supported between observable frames.
