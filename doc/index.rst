Welcome to excee's documentation!
=================================

:mod:`excee` is (yet another) library for analyzing and plotting sampled distributions, in particular from Markov chain Monte Carlo simulations.

This documentation is a work in progress.


Yet another "why yet another MCMC plotting library" explainer
-------------------------------------------------------------

:mod:`excee` is another in a long line of libraries for visualizing high-dimensional, sampled distributions.
Its defining design feature is its usage of :mod:`xarray` data structures, which provide sufficient metadata to remove numerous pain points of using plain :mod:`numpy` arrays (selecting variables by name rather than index, specifying labels via :attr:`~xarray.DataArray.attrs` rather than ordered lists, and enabling per-variable configuration via dictionary input) without introducing custom classes.
:mod:`xarray`'s data structures, aside from having their own slate of features (including built-in serialization to and from disk and semantic indexing; see `Why xarray? <https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html>`_), remain compatible with almost the entire Scientific Python ecosystem.
For the uninitiated, :class:`xarray.Dataset`, the main workhorse for :mod:`excee`, could be thought of as a dictionary of :mod:`numpy` arrays that still acts like an array.

Several additional considerations motivate its design and features:

* For :ref:`posterior plots <distribution-plots>`, it originally wrapped `corner <https://corner.readthedocs.io/>`_. Eventually, new features required a reimplementation that remains heavily inspired by corner. These include:

  * Panel layouts other than the classic isosceles triangle. :func:`~excee.compare_2d_dists` and :func:`~excee.plot_joint_dist` allow specifying row and column variables separately and even support completely arbitrary arrangements (irregular plot grids).
  * Smoothing via :ref:`kernel density estimation <density-estimation>` that avoids the most common sources of bias, in particular from parameter boundaries (which naive smoothing bleeds across) and correlations between variables in 2D (which naive smoothing artificially isotropizes).
  * Automatic boundary detection from samples with a simple but effective `coefficient of L-variation <https://en.wikipedia.org/wiki/L-moment>`_ threshold, used both for boundary handling in kernel density estimation and for (optionally) automated credible interval summaries (choosing between one- and two-sided summaries).
  * Built-in support for comparing multiple distributions on the same axes with :func:`~excee.compare_2d_dists`.
  * One-dimensional distribution plots, in standard style (:func:`~excee.compare_1d_dists`) and as "violin" plots (:func:`~excee.compare_violin`).

* Some libraries with similar feature sets tend to oversmooth by default, which can deceive the user into thinking that their samples have converged sufficiently. In the spirit of "crappy contours" being an extremely useful diagnostic tool, joint distribution plots display unsmoothed distributions by default. To leave little excuse for not checking that smoothed density estimates faithfully represent the underlying sample, :func:`~excee.test_smoothing`, a thin wrapper of :func:`~excee.compare_2d_dists`, makes the check trivial.
* By centralizing on :mod:`xarray` for its data format and avoiding custom containers for :mod:`matplotlib` figures, :mod:`excee` introduces no library-specific classes or structures for users to learn and interface with---just a simple, functional API.
* Through simple design and effective use of functionality already implemented in :mod:`numpy` and :mod:`scipy`, the library itself is dramatically simpler and a fraction of the size of some others.
* Beyond posterior plots, :mod:`excee` provides diagnostic utilities to assess convergence, including (performance-optimized) :ref:`autocorrelation analysis <autocorrelation>` and :ref:`diagnostic plots <diagnostic-plots>` as well as other :ref:`statistics <stats>`.
* The very fact that this explainer is a cliché might suggest that (some of) the workflow these libraries address is too user-specific for a general purpose library. Nevertheless, :mod:`excee` attempts to impose a minimum of stylistic choices, leaving as much plot styling accessible (via propagated keyword argument dictionaries) as possible.


Table of contents
-----------------

.. toctree::
    :maxdepth: 2
    :caption: API Reference:

    plotting
    density
    analysis
    sampling
    types
    changes
