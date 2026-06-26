Analysis
========

.. _io:

Loaders
-------

.. currentmodule:: excee
.. autofunction:: load_result_tree
.. currentmodule:: excee.io
.. autofunction:: to_dataarray
.. autofunction:: to_dataset
.. autofunction:: restore_dsets
.. autofunction:: compress
.. autofunction:: decompress
.. autofunction:: extract_posterior
.. autofunction:: construct_dt
.. autofunction:: construct_dt_for_storage
.. autofunction:: deconstruct_dt


.. _autocorrelation:

Autocorrelation analysis
------------------------

.. currentmodule:: excee
.. autofunction:: autocorr_time
.. autofunction:: autocorr_time_over_time
.. autofunction:: discard_and_thin


.. _other_analysis_utils:

Other utilities
---------------

.. autofunction:: get_random_sample
.. autofunction:: project_sample
.. currentmodule:: excee.analysis
.. autofunction:: split_vector_vars


.. _diagnostic-plots:

Diagnostic plots
----------------

.. currentmodule:: excee
.. autofunction:: plot_trace_2d
.. autofunction:: plot_autocorr_evolution


.. _stats:

Statistics
----------

.. currentmodule:: excee
.. autofunction:: eff_gaussian_tension
