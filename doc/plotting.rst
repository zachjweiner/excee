.. _distribution-plots:

Distribution plots
==================

The main user-facing methods documented in :ref:`joint-distributions` and
:ref:`marginal-distributions` dispatch individual one- and two-dimensional distribution
plotting to the methods documented in :ref:`low-level`.
These in turn make use of the density estimation methods in :ref:`density-estimation`.

The methods in :ref:`joint-distributions` and :ref:`marginal-distributions`,
which generate plots for multiple distributions at once, allow some arguments to be
specified for individual datasets and variables or for all at once.
The semantics of this "broadcasting" for each function argument are indicated by its type
annotation, which are described in :ref:`broadcasting-semantics`.


.. _joint-distributions:

Joint distribution plots
------------------------

.. currentmodule:: excee
.. autofunction:: compare_2d_dists
.. autofunction:: plot_joint_dist
.. autofunction:: test_smoothing


.. _marginal-distributions:

One-dimensional distributions
-----------------------------

.. autofunction:: compare_1d_dists
.. autofunction:: compare_violin
.. autofunction:: excee.plot.plot_violin


.. _helper-functions:

Helper functions
----------------

.. currentmodule:: excee.plot
.. autofunction:: get_1d_level
.. autofunction:: get_2d_level
.. autofunction:: get_inclusive_limits
.. autofunction:: get_inclusive_limits_from_2d_levels


.. _low-level:

Low-level plotting functions
----------------------------

.. currentmodule:: excee.plot.dist
.. autofunction:: plot_1d_dist
.. autofunction:: plot_2d_dist
.. autofunction:: plot_2d_density
