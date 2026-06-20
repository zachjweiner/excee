__copyright__ = "Copyright (C) 2025 Zachary J Weiner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from itertools import cycle
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy.stats import Normal
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D, blended_transform_factory
from excee.util import ordered_union, label_from_attrs, _init_kwargs_dict
from excee.density import compute_1d_density, detect_boundaries
from excee.plot.titles import make_ci_str, add_stacked_title
from excee.stats import compute_ci
from excee.plot.diagnostic import plot_autocorr_evolution, plot_trace_2d
from excee.plot.dist import (
    get_1d_level, get_2d_level, sigma_from_2d_level,
    plot_1d_dist, plot_2d_dist, plot_joint_dist,
    _bcast_to_dict
)

from typing import Any, Literal
from collections.abc import Sequence, Mapping
from numpy.typing import ArrayLike
from excee._typing import (
    DataSpec, BroadcastableToDatasets, BroadcastableToVarsAndDatasets,
    BoundsTuple, LimitsSpecifiers, AxesScale, CIKind, ColorType
)


def as_dataset(node):
    return node.dataset if isinstance(node, xr.DataTree) else node


def _get_n_colors(n, colors=None, ax=None):
    if colors is not None:
        cycler = cycle(colors)
    elif ax is not None:
        def _ax_cycle():
            while True:
                yield ax._get_lines.get_next_color()
        cycler = _ax_cycle()
    else:
        cycler = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    return [next(cycler) for _ in range(n)]


def process_dict_options_to_tuple(options, keys, default=None):
    if options is None:
        return tuple(default for _ in keys)
    return tuple(options.get(key, default) for key in keys)


def get_inclusive_limits(dsets, quantiles=None, sigma=2.5):
    if quantiles is None:
        quantiles = Normal().cdf([-sigma, sigma])

    bounded = xr.concat(
        [detect_boundaries(ds) for ds in dsets],
        "ds", fill_value=False,
    )
    # vars missing from some dsets would force bounded = True without fill_value
    bounded = bounded.any("ds").rename(tail="quantile")
    lims = xr.concat(
        [
            xr.concat([ds.min() for ds in dsets], "ds").min(),
            xr.concat([ds.max() for ds in dsets], "ds").max(),
        ],
        "quantile",
    )
    ds_qs = xr.concat(
        [ds.quantile(quantiles, method="inverted_cdf") for ds in dsets], "ds",
    )
    qs = xr.concat(
        [
            ds_qs.isel(quantile=0).min("ds", skipna=True),
            ds_qs.isel(quantile=1).max("ds", skipna=True),
        ],
        "quantile",
    )
    return xr.where(bounded, lims, qs)


def get_inclusive_limits_from_2d_levels(dsets, levels, pad=1):
    sigma = sigma_from_2d_level(np.max(levels))
    return get_inclusive_limits(dsets, sigma=sigma+pad)


def _parse_limits(dsets, all_keys, limits, limit_sigma):
    if isinstance(limits, Mapping):
        # ensure xr.Datasets are converted to true dicts
        limits = {key: limits[key] for key in limits}
    limits = _bcast_to_dict(limits, all_keys, "auto")
    _autos = [
        key for key in all_keys
        if isinstance(limits[key], str) and limits[key] == "auto"
    ]
    if _autos:
        _dsets = [ds[[k for k in _autos if k in ds]] for ds in dsets]
        _dsets = [ds for ds in _dsets if ds.data_vars]  # drop empty dsets
        _alims = get_inclusive_limits(_dsets, sigma=limit_sigma)
        limits |= {key: np.asarray(_alims[key]) for key in _autos}

    return limits


def _bcast_to_list(inpt, n, default):
    inpt = inpt if inpt is not None else default
    if isinstance(inpt, list):
        if len(inpt) != n:
            raise ValueError(
                f"Supply listed per-dataset arguments for all {n} datasets.")
        return inpt
    if isinstance(inpt, dict):
        return [inpt.copy() for _ in range(n)]
    return [inpt] * n


def bcast_zip(*args):
    lens = {len(a) for a in args if isinstance(a, list)}
    if len(lens) > 1:
        raise ValueError("All passed list arguments must have the same length")

    for i in range(lens.pop() if lens else 1):
        yield tuple(arg[i] if isinstance(arg, list) else arg for arg in args)


def _bcast_to_list_of_dict(inpt, keys, default, n):
    return [_bcast_to_dict(a, keys, default) for a in _bcast_to_list(inpt, n, None)]


def compare_2d_dists(
    dsets: Sequence[DataSpec],
    cols: Sequence[str] | None = None,
    *,
    rows: Sequence[str] | None = None,
    var_names: Sequence[str] | None = None,
    rowcols: ArrayLike[tuple[str | None, str | None]] | None = None,
    colors: Sequence[ColorType] | None = None,
    bins: BroadcastableToVarsAndDatasets[int] | None = None,
    smooth: BroadcastableToVarsAndDatasets[float] | None = None,
    bounds: BroadcastableToVarsAndDatasets[BoundsTuple] | None = None,
    density_kwargs: BroadcastableToDatasets[Mapping[str, Any]] | None = None,
    levels: Sequence[float] | None = None,
    axes_scale: BroadcastableToVarsAndDatasets[AxesScale] = "linear",
    limits: Mapping[str, LimitsSpecifiers] | Literal["auto"] | None = "auto",
    limit_pad: float = 1,
    kwargs_1d: BroadcastableToDatasets[Mapping[str, Any]] | None = None,
    kwargs_2d: BroadcastableToDatasets[Mapping[str, Any]] | None = None,
    exclude_1d_idx: Sequence[int] | None = None,
    exclude_2d_idx: Sequence[int] | None = None,
    ci_kind: BroadcastableToVarsAndDatasets[CIKind | None] = "auto",
    default_ci_kind: Literal["hdi", "eti"] = "eti",
    ci_prob: BroadcastableToVarsAndDatasets[float] | None = None,
    show_titles: bool = True,
    title_kwargs: Mapping[str, Any] | None = None,
    fig: plt.Figure | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Compare one- and two-dimensional marginal distributions among datasets.

    This method plots multiple distributions at once, calling
    :func:`~excee.plot_joint_dist` for each, but it may be equivalently used for
    a single distribution by simply passing a list containing a single dataset.
    There is therefore little reason to use :func:`~excee.plot_joint_dist` itself,
    but its documentation provides more details for some parameters.

    Parameters
    ----------
    dsets
        Sequence of datasets to compare.
    cols
        Names of variables to plot along the columns.
        Defaults to the union of all keys present in the elements of ``dsets``.
    rows
        Names of variables to plot along the rows.
        Defaults to ``cols``.
    var_names
        Alternative specification of variable names for both ``rows`` and ``cols``
        if ``cols`` is not passed.
    rowcols
        Explicit 2D array of variable pairs.
        Overrides ``rows`` and ``cols``.
        See :func:`plot_joint_dist`.
    colors
        Sequence of colors used for each dataset.
    bins
        Bin count/grid size for the 2D histogram/kernel density estimate.
        Defaults to ``None``, in which case it is determined as described in
        :func:`plot_2d_dist`.
    smooth
        Smoothing factor for kernel density estimates that multiplies the
        estimated optimal bandwidth.
        Defaults to ``0`` (plotting raw histograms without KDE smoothing).
    bounds
        Interval of variables' prior support.
        Automatically inferred if not passed.
    density_kwargs
        Additional keyword arguments passed to
        :func:`~excee.density.compute_1d_density` and
        :func:`~excee.density.compute_2d_density`.
    levels
        Mass levels to display for two-dimensional distributions.
        Defaults to the :math:`1` and :math:`2 \\sigma` levels of a 2D normal
        (:math:`39.3\\%` and :math:`86.5\\%`).
    axes_scale
        Axes scales.
        Defaults to ``"linear"`` for all variables.
    limits
        Axes limits. Can be specified as

        * ``"auto"`` to automatically determine limits with
          :func:`get_inclusive_limits_from_2d_levels`
        * ``None`` to leave unmodified
        * or on a per-variable basis with a :class:`~collections.abc.Mapping` of
          (all or a subset of) ``var_names`` to :py:type:`LimitsSpecifiers`\\ s.

        Defaults to ``"auto"`` for any of ``var_names`` not explicitly specified.
    limit_pad
        Padding applied to automatically determined limits, in standard deviations.
        Defaults to ``1``.
    kwargs_1d
        Additional keyword arguments passed to :func:`plot_1d_dist`.
    kwargs_2d
        Additional keyword arguments passed to :func:`plot_2d_dist`.
    exclude_1d_idx
        Indices of datasets to exclude from one-dimensional distribution plots.
    exclude_2d_idx
        Indices of datasets to exclude from two-dimensional distribution plots.
    ci_kind
        Type of credible interval to display.
        Defaults to ``"auto"``, in which case it is automatically decided by
        :func:`~excee.plot.titles.decide_ci_kind`.
    default_ci_kind
        Fallback credible interval type if ``ci_kind`` is ``"auto"`` and
        :func:`~excee.plot.titles.decide_ci_kind` does not detect a one-sided
        distribution.
    ci_prob
        Probability mass enclosed by depicted credible intervals.
    show_titles
        Whether to display titles with credible interval summaries on
        one-dimensional distribution plots.
    title_kwargs
        Formatting keyword arguments for axes titles, which are
        passed as expected by :func:`~excee.plot.titles.make_ci_str`
        and :func:`~excee.plot.titles.add_stacked_title`.
    fig
        Existing :class:`~matplotlib.figure.Figure` figure to draw on.
        If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to :func:`plot_joint_dist`.
    """

    dsets = [as_dataset(ds) for ds in dsets]
    n = len(dsets)
    exclude_1d_idx = exclude_1d_idx or []
    exclude_2d_idx = exclude_2d_idx or []

    colors = _get_n_colors(n, colors)

    if rowcols is not None:
        _rowcols = rowcols
    else:
        # though duplicated from plot_joint_dist, don't pass so that
        # it doesn't think we have a nonstandard layout (custom_layout)
        cols = cols if cols is not None else var_names
        if cols is None:
            cols = ordered_union([list(data.keys()) for data in dsets])

        rows = rows if rows is not None else cols

        from excee.plot.dist import assemble_rowcols
        _rowcols = assemble_rowcols(
            rows, cols,
            reverse=kwargs.get("reverse", False),
            ensure_1d_dists=kwargs.get("ensure_1d_dists", True),
        )

    all_keys = np.unique(np.lib.recfunctions.structured_to_unstructured(_rowcols))
    all_keys = [key for key in all_keys if key]

    if levels is None:
        levels = get_2d_level(np.arange(1, 3))

    sigma = sigma_from_2d_level(np.max(levels)) + limit_pad
    limits = _parse_limits(dsets, all_keys, limits, sigma)

    kw_sets = {
        name: _bcast_to_list_of_dict(inpt, all_keys, default, n)
        for name, inpt, default in (
            ("bins", bins, None),
            ("smooth", smooth, None),
            ("bounds", bounds, None),
            ("axes_scale", axes_scale, "linear"),
            ("ci_kind", ci_kind, "auto"),
            ("ci_prob", ci_prob, None),
        )
    }
    kw_sets |= {
        name: _bcast_to_list(inpt, n, default)
        for name, inpt, default in (
            ("kwargs_1d", kwargs_1d, {}),
            ("density_kwargs", density_kwargs, {}),
        )
    }
    kwargs_2d = _bcast_to_list(kwargs_2d, n, {})
    ds_kws = [
        {
            **{k: v[i] for k, v in kw_sets.items()},
            "color": colors[i],
            "skip_1d": i in exclude_1d_idx,
            "skip_2d": i in exclude_2d_idx,
        }
        for i in range(n)
    ]

    for data, ds_kw, kw_2d in zip(dsets, ds_kws, kwargs_2d):
        fig, axes = plot_joint_dist(
            data, rows=rows, cols=cols, rowcols=rowcols,
            levels=levels, limits=limits,
            fig=fig, show_titles=False, default_ci_kind=default_ci_kind,
            **kwargs, **ds_kw, **kw_2d,
        )

    title_kwargs = _init_kwargs_dict(title_kwargs)
    format_kwargs = {
        key: title_kwargs.pop(key)
        for key in ("err_prec", "rescale_thresh", "style", "include_long_names")
        if key in title_kwargs
    }
    if show_titles:
        axes_var_names = [
            [axes[idx], rc[0]]
            for idx, rc in np.ndenumerate(_rowcols)
            if rc[0] == rc[1] and rc[0] != ""
        ]
        for ax, key in axes_var_names:
            arys = [
                ds.get(key) if i not in exclude_1d_idx else None
                for i, ds in enumerate(dsets)
            ]
            titles = [
                make_ci_str(
                    ary, ci_kind=_ci_kind[key], ci_prob=_ci_prob[key],
                    default_ci_kind=default_ci_kind, **format_kwargs
                )
                if ary is not None else None
                for ary, _ci_kind, _ci_prob in zip(
                    arys, kw_sets["ci_kind"], kw_sets["ci_prob"]
                )
            ]
            add_stacked_title(ax, titles, colors=colors, **title_kwargs)

    return fig, axes


def compare_1d_dists(
    dsets: Sequence[DataSpec],
    *,
    var_names: Sequence[str] | None = None,
    smooth: BroadcastableToVarsAndDatasets[float] = 1,
    limits: Mapping[str, LimitsSpecifiers] | Literal["auto"] | None = "auto",
    limit_sigma: float = 3,
    ncol: int = 4,
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Compare one-dimensional marginal distributions among datasets for multiple
    variables by panel via a thin wrapper of :func:`compare_2d_dists`.

    Parameters
    ----------
    dsets
        Sequence of datasets to compare.
    var_names
        Names of variables to plot, one per panel.
        Defaults to the union of all keys present in the elements of ``dsets``.
    smooth
        Smoothing factor for kernel density estimates that multiplies the
        estimated optimal bandwidth.
        Unlike :func:`compare_2d_dists`, defaults to ``1``.
    limits
        Axes limits, as described in :func:`compare_2d_dists` with the exception
        that automatically determined axes limits use
        :func:`get_inclusive_limits`.
    limit_sigma
        Extent of automatically determined axes limits as a number of standard
        deviations from the median.
    ncol
        Maximum number of columns in the plot grid.
    **kwargs
        Additional keyword arguments passed to :func:`compare_2d_dists`.
    """
    dsets = [as_dataset(ds) for ds in dsets]
    if var_names is None:
        var_names = ordered_union([list(data.keys()) for data in dsets])

    smooth = _bcast_to_list_of_dict(smooth, var_names, 1, len(dsets))
    limits = _parse_limits(dsets, var_names, limits, limit_sigma)

    n = len(var_names)
    ncol = min(n, ncol)
    nrow = (n - 1) // ncol + 1

    rowcols = [(key, key) for key in var_names]
    rowcols += [("", "") for _ in range(nrow * ncol - n)]
    from excee.plot.dist import rowcol_dt
    rowcols = np.asarray(rowcols, dtype=rowcol_dt).reshape(nrow, ncol)
    fig, axes = compare_2d_dists(
        dsets, rowcols=rowcols, smooth=smooth, limits=limits, **kwargs,
    )

    return fig, axes


def plot_1d_dists(data, **kwargs):
    if "color" in kwargs:
        kwargs["colors"] = [kwargs.pop("color")]
    return compare_1d_dists([data], **kwargs)


def compare_violin(
    dsets: Sequence[DataSpec],
    *,
    var_names: Sequence[str] | None = None,
    side_labels: Sequence[str | None] | None = None,
    colors: Sequence[ColorType] | None = None,
    alphas: BroadcastableToDatasets[float] | None = None,
    limits: Mapping[str, LimitsSpecifiers] | Literal["auto"] | None = "auto",
    limit_sigma: float = 3,
    bins: BroadcastableToVarsAndDatasets[int] = 1024,
    smooth: BroadcastableToVarsAndDatasets[float] = 1,
    bounds: BroadcastableToVarsAndDatasets[BoundsTuple] | None = None,
    axes_scale: BroadcastableToVarsAndDatasets[AxesScale] = "linear",
    ci_kind: BroadcastableToVarsAndDatasets[CIKind | None] = "eti",
    default_ci_kind: Literal["eti", "hdi"] = "eti",
    ncol: int = 4,
    fig: plt.Figure | None = None,
    density_kwargs: BroadcastableToDatasets[Mapping[str, Any]] | None = None,
    fill_kwargs: BroadcastableToDatasets[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Compare one-dimensional marginal distributions among datasets for multiple
    variables with vertically stacked violin plots.

    Parameters
    ----------
    dsets
        Sequence of datasets to compare.
    var_names
        Names of variables to plot, one per panel.
        Defaults to the union of all keys present in the elements of ``dsets``.
    side_labels
        Sequence of dataset labels to display alongside the violin plots.
        Appear outside of the leftmost axes in each row.
    colors
        Sequence of colors used for each dataset.
    alphas : BroadcastabletoDatasets[float]
        Transparency of each violin plot.
        Defaults to ``1``.
    density_kwargs : BroadcastableToDatasets[Mapping[str, Any]]
        Additional keyword arguments passed to
        :func:`~excee.density.compute_1d_density`.
    fill_kwargs
        Additional keyword arguments passed to
        :meth:`matplotlib.axes.Axes.fill_between`.
    limits
        Axes limits, as described in :func:`compare_2d_dists` with the exception
        that automatically determined axes limits use
        :func:`get_inclusive_limits`.
    limit_sigma
        Extent of automatically determined axes limits as a number of standard
        deviations from the median.
    bins
        Grid size for kernel density estimates.
        Defaults to ``1024``.
    smooth
        Smoothing factor for kernel density estimates that multiplies the
        estimated optimal bandwidth.
        Unlike :func:`compare_2d_dists`, defaults to ``1``.
        Must be greater than zero.
    bounds
        Interval of variables' prior support.
        Automatically inferred if not passed.
    axes_scale
        Axes scales.
        Defaults to ``"linear"`` for all variables.
    ci_kind
        Type of credible interval to display.
    default_ci_kind
        Fallback credible interval type if ``ci_kind`` is ``"auto"`` and
        :func:`~excee.plot.titles.decide_ci_kind` does not detect a one-sided
        distribution.
    ncol
        Maximum number of columns in the plot grid.
    fig
        Existing :class:`~matplotlib.figure.Figure` figure to draw on.
        If ``None``, a new figure is created.
    **kwargs
        Additional keyword arguments passed to :func:`plot_violin`.
    """
    dsets = [as_dataset(ds) for ds in dsets]
    if var_names is None:
        var_names = ordered_union([list(ds.keys()) for ds in dsets])

    n = len(var_names)
    ncol = min(n, ncol)
    nrow = (n - 1) // ncol + 1

    if fig is not None:
        axes = np.array(fig.axes)
    else:
        fig, axes = plt.subplots(nrow, ncol, squeeze=False)

    if side_labels is None:
        side_labels = [None] * len(dsets)

    limits = _parse_limits(dsets, var_names, limits, limit_sigma)

    for col, (ax, key) in enumerate(zip(axes.flat, var_names)):
        if (lims := limits.get(key, None)) is not None:
            ax.set_xlim(lims)

        def _get_kwarg(arg, default):
            if isinstance(arg, list):
                return [_get_kwarg(_arg, key) for _arg in arg]  # noqa: B023
            elif isinstance(arg, dict):
                return arg.get(key, default)  # noqa: B023
            else:
                return arg

        _ = plot_violin(
            ax, [ds.get(key) for ds in dsets],
            bins=_get_kwarg(bins, 1024),
            smooth=_get_kwarg(smooth, 1),
            bounds=_get_kwarg(bounds, None),
            axes_scale=_get_kwarg(axes_scale, "linear"),
            ci_kind=_get_kwarg(ci_kind, default_ci_kind),
            default_ci_kind=default_ci_kind,
            side_labels=side_labels if col % ncol == 0 else None,
            colors=colors, alphas=alphas,
            density_kwargs=density_kwargs, fill_kwargs=fill_kwargs,
            **kwargs,
        )
    for ax in axes.flat[n:]:
        ax.axis("off")

    return fig, axes


def plot_violin(
    ax: plt.Axes,
    arys: Sequence[ArrayLike | xr.DataArray | None],
    *,
    input_kind: BroadcastableToDatasets[Literal["sample", "density"]] = "sample",
    colors: Sequence[ColorType] | None = None,
    alphas: BroadcastableToDatasets[float] = 1,
    bins: BroadcastableToDatasets[int] = 1024,
    smooth: BroadcastableToDatasets[float] = 1,
    bounds: BroadcastableToDatasets[BoundsTuple | None] = None,
    axes_scale: AxesScale = "linear",
    density_kwargs: BroadcastableToDatasets[Mapping[str, Any]] | None = None,
    fill_kwargs: BroadcastableToDatasets[Mapping[str, Any]] | None = None,
    relative_height: float = 1,
    title_pad: float = 0.3,
    interviolin_pad: float = 0.5,
    gap_fraction: float = 0.0025,
    show_titles: bool = True,
    title_kwargs: Mapping[str, Any] | None = None,
    plot_ci: bool = True,
    ci_kind: BroadcastableToDatasets[CIKind | None] = "eti",
    default_ci_kind: Literal["eti", "hdi"] = "eti",
    include_long_names: bool = False,
    label: str | None = None,
    limit_xpad_fraction: float = 0.01,
    min_x_upper_label: float = -np.inf,
    max_x_lower_label: float = np.inf,
    side_labels: Sequence[str | None] | None = None,
    side_label_kwargs: Mapping[str, Any] | None = None,
    side_label_pad: float = 0.0075,
) -> plt.Axes:
    """
    Plot vertically stacked violin plots.

    Parameters
    ----------
    ax
        Axis on which to draw the distributions.
    arys
        Sequence of sets of samples.
    input_kind
        Whether the input data represents raw ``"sample"``\\ s or an evaluated
        ``"density"``.
    colors
        Sequence of colors used for each distribution.
    alphas
        Transparency of the violin plots.
    bins
        Grid size for kernel density estimates.
        Defaults to ``1024``.
    smooth
        Smoothing factor for kernel density estimates that multiplies the
        estimated optimal bandwidth.
        Defaults to ``1``.
        Must be greater than zero.
    bounds
        Interval of variables' prior support.
        Automatically inferred if not passed.
    axes_scale
        Axes scales.
        Defaults to ``"linear"``.
    density_kwargs
        Additional keyword arguments passed to
        :func:`~excee.density.compute_1d_density`.
    fill_kwargs
        Additional keyword arguments passed to
        :meth:`~matplotlib.axes.Axes.fill_between`.
    relative_height
        Height of the violin relative to the font size.
    title_pad
        Padding between the density and its title, in units of the font size.
    interviolin_pad
        Padding between adjacent violin plots (including titles, if present),
        in units of the font size.
    gap_fraction
        Fraction of the total x-axis span used for gaps around the median.
    show_titles
        Whether to display titles with credible interval summaries.
    title_kwargs
        Formatting keyword arguments for titles.
    plot_ci
        Whether to plot credible intervals as highlighted regions.
    ci_kind
        Type of credible interval to display.
    default_ci_kind
        Fallback credible interval type if ``ci_kind`` is ``"auto"``.
    include_long_names
        Whether to include parameter names in the titles.
    label
        Label to use for the title if ``include_long_names=True``.
    limit_xpad_fraction
        Padding between violin limits and their labels, as a fraction
        of the x-axis span.
    min_x_upper_label
        Minimum x coordinate for upper limit labels.
    max_x_lower_label
        Maximum x coordinate for lower limit labels.
    side_labels
        Labels to display to the left of each violin plot.
    side_label_kwargs
        Additional keyword arguments passed to
        :meth:`~matplotlib.axes.Axes.text` for side labels.
    side_label_pad
        Padding between side labels and the axis edge.
    """

    def _get_density(ary, input_kind, bins, smooth, bounds, density_kwargs):
        density_kwargs = _init_kwargs_dict(density_kwargs)
        if ary is None:
            return None
        if input_kind == "density":
            coord, pdf = ary.coords[ary.dims[0]], np.asarray(ary)
        else:
            _ary = np.log(ary) if axes_scale == "log" else ary
            coord, pdf = compute_1d_density(
                np.asarray(_ary), bins, smooth, bounds=bounds, **density_kwargs)
            if axes_scale == "log":
                coord = np.exp(coord)
        da = xr.DataArray(pdf, dims="x", coords={"x": coord})
        try:
            da = da.assign_attrs(ary.attrs)
        except AttributeError:
            pass
        return da

    densities = [
        _get_density(*args)
        for args in bcast_zip(arys, input_kind, bins, smooth, bounds, density_kwargs)
    ]

    def _get_splits(da, ci_kind):
        if da is None:
            return None
        xmin, xmax = da.x[0].values, da.x[-1].values
        if not plot_ci:
            return np.array([xmin, xmax])

        _da = da.assign_coord(x=np.log(da.x)) if axes_scale == "log" else da

        ci1 = compute_ci(
            _da, "density",
            ci_kind=default_ci_kind if "limit" in ci_kind else ci_kind,
            ci_prob=get_1d_level(1),
        )
        if axes_scale == "log":
            ci1 = np.exp(ci1)
        if ci_kind in ("eti", "hdi"):
            ci2 = compute_ci(
                _da, "density", ci_kind=ci_kind, ci_prob=get_1d_level(2),
            )
            if axes_scale == "log":
                ci2 = np.exp(ci2)
            splits = [*ci1, *ci2, xmin, xmax]
        elif "limit" in ci_kind:
            lim = compute_ci(
                _da, "density", ci_kind=ci_kind, ci_prob=get_1d_level(2),
            )
            if axes_scale == "log":
                lim = np.exp(lim)
            splits = [*ci1, lim, xmin if "upper" in ci_kind else xmax]
        elif ci_kind is None:
            splits = [xmin, xmax]
        else:
            raise NotImplementedError(f"{ci_kind=}")

        return np.sort(np.unique(splits))

    splits = [_get_splits(*args) for args in bcast_zip(densities, ci_kind)]

    title_kwargs = _init_kwargs_dict(title_kwargs)
    ci_fmt_kw = {"include_long_names": include_long_names} | {
        key: title_kwargs.pop(key)
        for key in ("err_prec", "rescale_thresh", "style")
        if key in title_kwargs
    }
    title_kwargs.setdefault("clip_on", True)

    def _get_title(da, ci_kind):
        if show_titles and ci_kind is not None and da is not None:
            return make_ci_str(
                da, input_kind="density", ci_kind=ci_kind,
                ci_prob=get_1d_level(2 if "limit" in ci_kind else 1),
                default_ci_kind=default_ci_kind,
                label=label or label_from_attrs(da), **ci_fmt_kw,
            )
        else:
            return None

    titles = [_get_title(*args) for args in bcast_zip(densities, ci_kind)]

    ax.set_xscale(axes_scale)

    if ax.get_autoscalex_on():
        _splits = [s for s in splits if s is not None]
        if axes_scale == "log":
            _splits = np.log(_splits)
        xspan = np.max(_splits) - np.min(_splits)
    else:
        # axes limits have (presumably) already been set manually
        _xlim = ax.get_xlim()
        if axes_scale == "log":
            _xlim = np.log(_xlim)
        xspan = np.diff(_xlim)[0]

    if axes_scale == "log":
        # implemented up until this point
        # need to translate quantile_gap, etc., to log separation
        raise NotImplementedError("violin with axes_scale = 'log'")

    quantile_gap = gap_fraction * xspan
    text_xpad = limit_xpad_fraction * xspan

    side_label_kwargs = _init_kwargs_dict(side_label_kwargs)
    if side_labels is None:
        side_labels = [None] * len(arys)

    has_top_labels = ci_kind in ("eti", "hdi") and any(t is not None for t in titles)

    # determine layout

    fontsize_spec = title_kwargs.get("fontsize", plt.rcParams["font.size"])
    fs_pts = FontProperties(size=fontsize_spec).get_size_in_points()

    pad_label_pts = title_pad * fs_pts
    pad_next_pts = interviolin_pad * fs_pts
    density_height = relative_height * fs_pts
    text_height = 1.25 * fs_pts

    step_pts = density_height + pad_next_pts
    bottom_buffer = density_height / 2
    top_buffer = density_height / 2
    if has_top_labels:
        step_pts += pad_label_pts + text_height
        top_buffer += pad_label_pts + text_height

    total_height_pts = bottom_buffer + (len(arys) - 1) * step_pts + top_buffer

    fig = ax.get_figure()
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()

    dpi_scale = 72 / fig.dpi
    current_ax_h = ax.get_window_extent(renderer).height * dpi_scale
    current_fig_height_pts = fig.get_figheight() * 72
    overhead_pts = current_fig_height_pts - current_ax_h

    margin_pts = ax.get_ymargin() * 100
    total_axes_h = total_height_pts + 2 * margin_pts

    target_fig_height_in = (total_axes_h + overhead_pts) / 72
    fig.set_size_inches(fig.get_figwidth(), target_fig_height_in)

    ax.set_ylim(-margin_pts, total_height_pts + margin_pts)

    colors = _get_n_colors(len(densities), colors, ax=ax)
    _iter = enumerate(bcast_zip(
        densities, splits, titles, side_labels, colors, alphas, fill_kwargs,
    ))
    for i, (pdf, split, title, side_label, color, alpha, _fill_kw) in _iter:
        baseline = total_height_pts - top_buffer - i * step_pts

        if side_label is not None:
            kw = {"ha": "right", "va": "center"} | side_label_kwargs
            ax.text(
                -side_label_pad, baseline, side_label,
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                color=color, alpha=alpha,
                **kw,
            )

        if pdf is None:
            continue

        pdf = pdf / pdf.max() / 2
        spl = CubicSpline(pdf.x, pdf)

        y_layout_trans = Affine2D().scale(1, density_height).translate(0, baseline)
        trans = blended_transform_factory(
            ax.transData,
            y_layout_trans + ax.transData,
        )

        sections = zip(
            np.concatenate([split[:1], split[1:] + quantile_gap / 2]),
            np.concatenate([split[1:-1] - quantile_gap / 2, split[-1:]])
        )
        _fill_kw = _init_kwargs_dict(_fill_kw)
        _fill_kw.setdefault("lw", 0)
        for x0, x1 in sections:
            _x = np.linspace(x0, x1, 400)
            _pdf = spl(_x)
            ax.fill_between(
                _x, -_pdf, _pdf,
                transform=trans, color=color, alpha=alpha, **_fill_kw,
            )

        if title is not None:
            if ci_kind == "upper_limit":
                q = split[-1]
                kw = {"ha": "left", "va": "center_baseline"} | title_kwargs
                ax.text(
                    max(q + text_xpad, min_x_upper_label), baseline, title,
                    color=color, alpha=alpha, transform=ax.transData, **kw,
                )
            elif ci_kind == "lower_limit":
                q = split[0]
                kw = {"ha": "right", "va": "center_baseline"} | title_kwargs
                ax.text(
                    min(q - text_xpad, max_x_lower_label), baseline, title,
                    color=color, alpha=alpha, transform=ax.transData, **kw,
                )
            elif ci_kind in ("eti", "hdi"):
                center = split[split.size // 2] if plot_ci else pdf.idxmax().item()
                text_y = baseline + density_height / 2 + pad_label_pts
                kw = {"ha": "center", "va": "bottom"} | title_kwargs
                ax.text(
                    center, text_y, title,
                    color=color, alpha=alpha, transform=ax.transData, **kw,
                )

    # infer whether current xaxis is shared and won't display labels
    tp = ax.xaxis.get_tick_params()
    # https://github.com/matplotlib/matplotlib/issues/27416
    if tp.get("labelbottom", tp.get("labelleft")) and not ax.get_xlabel():
        for ary in arys:
            try:
                if label := label_from_attrs(ary):
                    ax.set_xlabel(label)
                    break
            except AttributeError:
                pass  # not a DataArray

    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    return ax


def test_smoothing(dset, bins_unsmoothed=20, bins_smoothed=256, *, smooth=1,
                   color="k", color_unsmoothed="r",
                   fill_contours=False, shade_background=False,
                   limit_pad=0.5, kwargs_2d=None, **kwargs):
    kwargs_2d = _bcast_to_list(kwargs_2d, 2, {})
    for kw2d in kwargs_2d:
        kw2d.setdefault("contour_kwargs", {})
    lws = kwargs_2d[0]["contour_kwargs"].setdefault("linewidths", [1.5])
    kwargs_2d[1]["contour_kwargs"].setdefault("linewidths", np.array(lws) * 1/2)

    return compare_2d_dists(
        [dset, dset], bins=[bins_unsmoothed, bins_smoothed], smooth=[0, smooth],
        colors=[color_unsmoothed, color], limit_pad=limit_pad,
        fill_contours=fill_contours, shade_background=shade_background,
        kwargs_2d=kwargs_2d, **kwargs
    )


__all__ = [
    "get_1d_level",
    "get_2d_level",
    "get_inclusive_limits",
    "get_inclusive_limits_from_2d_levels",
    "plot_autocorr_evolution",
    "plot_trace_2d",
    "plot_1d_dist",
    "plot_2d_dist",
    "plot_joint_dist",
    "compare_2d_dists",
    "compare_1d_dists",
    "plot_1d_dists",
    "compare_violin",
    "plot_violin",
    "test_smoothing",
]
