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


import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy.stats import Normal
from excee.util import (
    ordered_union, label_from_attrs, get_long_names, _init_kwargs_dict,
)
from excee.density import compute_1d_density, detect_boundaries
from excee.plot.titles import (
    std_quantiles, add_stacked_title,
    measurement_from_log_pdf, measurement_from_sample, quantiles_from_log_pdf
)
from excee.plot.diagnostic import plot_autocorr_evolution, plot_trace_2d
from excee.plot.dist import (
    get_2d_level, sigma_from_2d_level, plot_1d_dist, plot_2d_dist, plot_joint_dist
)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def _get_n_colors(colors, n):
    from itertools import cycle

    if colors is not None:
        color_cycler = cycle(colors)
    else:
        color_cycler = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    colors = [next(color_cycler) for _ in range(n)]

    return colors


def process_dict_options_to_tuple(options, keys, default=None):
    if options is None:
        return tuple(default for _ in keys)
    return tuple(options.get(key, default) for key in keys)


def get_inclusive_limits(dsets, quantiles=None, sigma=2.5):
    if quantiles is None:
        quantiles = Normal().cdf([-sigma, sigma])

    bounded = xr.concat([detect_boundaries(ds) for ds in dsets], "ds")
    bounded = bounded.any("ds").rename(tail="quantile")
    lims = xr.concat(
        [
            xr.concat([ds.min() for ds in dsets], "ds").min(),
            xr.concat([ds.max() for ds in dsets], "ds").max(),
        ],
        "quantile",
    )
    ds_qs = xr.concat([ds.quantile(quantiles) for ds in dsets], "ds")
    qs = xr.concat(
        [ds_qs.isel(quantile=0).min("ds"), ds_qs.isel(quantile=1).max("ds")],
        "quantile",
    )
    return xr.where(bounded, lims, qs)


def get_inclusive_limits_from_2d_levels(dsets, levels, pad=1):
    sigma = sigma_from_2d_level(np.max(levels))
    return get_inclusive_limits(dsets, sigma=sigma+pad)


def compare_1d_dists(datasets, *, labels=None, var_names=None,
                     ncol=4, w=4, aspect=1,
                     axes_scale=None, limits="auto", limit_pad=1,
                     colors=None, show_titles=True, fig=None,
                     quantiles=std_quantiles, title_kwargs=None,
                     title_loc="center", title_stack_pad_frac=0.2,
                     include_long_names=True, **kwargs):
    if var_names is None:
        var_names = ordered_union([list(data.keys()) for data in datasets])
    if labels is None:
        labels = [None for _ in datasets]
    colors = _get_n_colors(colors, len(datasets))

    if limits == "auto":
        # ensure all quantiles are included in limits
        _sigma = (
            np.max(np.abs(Normal().icdf(quantiles))) + limit_pad
            if quantiles is not None else 3
        )
        _sigma = max(_sigma, 3)
        limits = get_inclusive_limits(datasets, sigma=_sigma)

    n = len(var_names)
    ncol = min(n, ncol)
    nrow = (n - 1) // ncol + 1

    if fig is not None:
        axes = np.array(fig.axes)
    else:
        if w is None:
            figsize = plt.rcParams["figure.figsize"]
        else:
            h = w / aspect
            figsize = (w*ncol, h*nrow)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        for ax in axes.flat:
            ax.set_box_aspect(1/aspect)

    axes_scale = _init_kwargs_dict(axes_scale)
    limits = _init_kwargs_dict(limits)

    title_quantiles = kwargs.pop(
        "title_quantiles",
        quantiles if quantiles is not None and len(quantiles) == 3
        else std_quantiles
    )

    for data, label, color in zip(datasets, labels, colors):
        xlabels = dict(zip(data.keys(), get_long_names(data)))
        weights = data.get("weights")
        for ax, key in zip(axes.flat, var_names):
            if key not in data:
                continue

            scale = axes_scale.get(key, "linear")
            ax.set_xlabel(xlabels[key])

            sample = data[key].values.ravel()
            plot_1d_dist(
                ax, sample, weights=weights, axes_scale=scale,
                label=label, color=color, quantiles=quantiles,
                **kwargs,
            )

            ax.set_xscale(scale)
            if (lims := limits.get(key)) is not None:
                ax.set_xlim(*lims)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax in axes.flat:
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(ymin=0)
        ax.tick_params(which="both", top=False, left=False, right=False)
        ax.spines[["left", "right", "top"]].set_visible(False)

    if show_titles:
        for ax, vn in zip(axes.flat, var_names):
            arys = [ds.get(vn) for ds in datasets]
            add_stacked_title(
                ax, arys, title_quantiles,
                colors=colors, title_loc=title_loc,
                title_kwargs=title_kwargs,
                title_stack_pad_frac=title_stack_pad_frac,
                include_long_names=include_long_names,
            )

    return fig, axes


def plot_1d_dists(data, **kwargs):
    if "color" in kwargs:
        kwargs["colors"] = [kwargs.pop("color")]
    return compare_1d_dists([data], **kwargs)


def compare_2d_dists(datasets, cols=None, *, rows=None, rowcols=None, colors=None,
                     show_titles=True, title_kwargs=None, title_loc="center",
                     title_stack_pad_frac=0.2, include_long_names=True,
                     exclude_1d_idx=None, exclude_2d_idx=None,
                     levels=None, limits="auto", limit_pad=1,
                     fig=None, **kwargs):
    exclude_1d_idx = exclude_1d_idx or []
    exclude_2d_idx = exclude_2d_idx or []
    default_contour_kwargs = _init_kwargs_dict(kwargs.pop("contour_kwargs", None))

    colors = _get_n_colors(colors, len(datasets))

    if levels is None:
        levels = get_2d_level(np.arange(1, 3))
    if isinstance(limits, str) and limits == "auto":
        limits = get_inclusive_limits_from_2d_levels(datasets, levels, pad=limit_pad)

    if rowcols is None:
        cols = cols if cols is not None else kwargs.pop("var_names", None)
        if cols is None:
            cols = ordered_union([list(data.keys()) for data in datasets])

        rows = rows if rows is not None else cols

        from excee.plot.dist import assemble_rowcols
        rowcols = assemble_rowcols(
            rows, cols,
            reverse=kwargs.get("reverse", False),
            ensure_1d_dists=kwargs.get("ensure_1d_dists", True),
        )

    bins = kwargs.pop("bins", None)
    smooth = kwargs.pop("smooth", None)
    bounds = kwargs.pop("bounds", None)
    _kwargs_1d = kwargs.pop("kwargs_1d", None)

    for i, (data, color) in enumerate(zip(datasets, colors)):
        ds_kw = {}
        ds_kw["color"] = color
        contour_kwargs = default_contour_kwargs.copy()
        contour_kwargs.setdefault("colors", [color])
        ds_kw["contour_kwargs"] = contour_kwargs
        ds_kw["kwargs_1d"] = _init_kwargs_dict(_kwargs_1d)

        if bins is not None:
            ds_kw["bins"] = bins[i] if isinstance(bins, list) else bins
        if smooth is not None:
            ds_kw["smooth"] = smooth[i] if isinstance(smooth, list) else smooth
        if bounds is not None:
            ds_kw["bounds"] = bounds[i] if isinstance(bounds, list) else bounds

        fig, axes = plot_joint_dist(
            data, rows=rows, cols=cols, rowcols=rowcols,
            levels=levels, limits=limits,
            fig=fig, show_titles=False,
            skip_1d=i in exclude_1d_idx,
            skip_2d=i in exclude_2d_idx,
            **kwargs, **ds_kw,
        )

    if show_titles:
        title_quantiles = kwargs.get(
            "title_quantiles",
            kwargs.get("quantiles", std_quantiles)
        )

        axes_var_names = [
            [axes[idx], rc[0]]
            for idx, rc in np.ndenumerate(rowcols)
            if rc[0] == rc[1] and rc[0] != ""
        ]
        for ax, vn in axes_var_names:
            arys = [
                ds.get(vn) if i not in exclude_1d_idx else None
                for i, ds in enumerate(datasets)
            ]
            add_stacked_title(
                ax, arys, title_quantiles,
                colors=colors, title_loc=title_loc,
                title_kwargs=title_kwargs,
                title_stack_pad_frac=title_stack_pad_frac,
                include_long_names=include_long_names,
            )

    return fig, axes


def test_smoothing(dset, bins_unsmoothed=20, bins_smoothed=256, *, smooth=1,
                   color="k", color_unsmoothed="r", contour_kwargs=None,
                   limit_pad=0.5, **kwargs):
    contour_kwargs = _init_kwargs_dict(contour_kwargs)
    lws = contour_kwargs.setdefault("linewidths", [1])
    fig, _ = compare_2d_dists(
        [dset], bins=bins_unsmoothed, smooth=0,
        colors=[color_unsmoothed], limit_pad=limit_pad, **kwargs
    )
    contour_kwargs["linewidths"] = np.array(lws) * 2/3
    return compare_2d_dists(
        [dset], bins=bins_smoothed, smooth=smooth,
        colors=[color], limit_pad=limit_pad,
        contour_kwargs=contour_kwargs,
        **(kwargs | {"fig": fig}),
    )


def plot_violin(ax, dsets, *,
                bins=512, smooth=1, density_kwargs=None,
                split_quantiles=None, extend_to=(np.inf, -np.inf),
                quantile_gap=None, gap_fraction=0.0025,
                violin_pad=0.1, text_dq=0.005, fill_alpha=1, lw=0,
                labels=None, label_kwargs=None, label_pad=0.005,
                measurement_kind=None, measurement_labels=None,
                measurement_kwargs=None, measurement_pad=0.05,
                min_q_upper_label=-np.inf):
    density_kwargs = _init_kwargs_dict(density_kwargs)

    violin_h = 1 - violin_pad

    if split_quantiles is None:
        from scipy.stats import norm
        split_quantiles = norm.cdf([-np.inf, *np.arange(-2, 3), np.inf])

    if quantile_gap is None:
        if ax.get_autoscale_on():
            xmin = min(ds.quantile(split_quantiles[0]).values for ds in dsets)
            xmax = max(ds.quantile(split_quantiles[-1]).values for ds in dsets)
        else:
            # axes limits have (presumably) already been set manually
            xmin, xmax = ax.get_xlim()
        quantile_gap = gap_fraction * (xmax - xmin)

    label_kwargs = _init_kwargs_dict(label_kwargs)
    label_kwargs.setdefault("fontsize", "small")

    measurement_kwargs = _init_kwargs_dict(measurement_kwargs)
    measurement_kwargs.setdefault("fontsize", "small")
    measurement_kwargs.setdefault("clip_on", True)
    meas_title_kwargs = {}
    meas_title_kwargs["err_prec"] = measurement_kwargs.pop("err_prec", 2)
    meas_title_kwargs["rescale_thresh"] = measurement_kwargs.pop("rescale_thresh", 3)
    meas_title_kwargs["style"] = measurement_kwargs.pop("style", "paren")
    meas_title_kwargs["quantiles"] = measurement_kwargs.pop(
        "quantiles", std_quantiles)

    if labels is None:
        labels = [None] * len(dsets)
    if measurement_labels is None:
        measurement_labels = [None] * len(dsets)

    prop_cycle = plt.rcParams["axes.prop_cycle"]

    y_center = 0.
    _iter = zip(prop_cycle, dsets, labels, measurement_labels)
    for props, ds, label, meas_label in _iter:
        if isinstance(ds, tuple) or (isinstance(ds, np.ndarray) and ds.ndim == 2):
            x, pdf = ds
            # truncate at ~ \pm 6 \sigma to avoid issues from pdf not integrating
            # quite to unity due to numerical error
            _cut = 1e-9
            split_quantiles = np.maximum(np.minimum(split_quantiles, 1-_cut), _cut)
            qs = quantiles_from_log_pdf(np.log(pdf), x, split_quantiles)
            median, = quantiles_from_log_pdf(np.log(pdf), x, (0.5,))
            title = measurement_from_log_pdf(np.log(pdf), x, **meas_title_kwargs)
        else:
            x, pdf = compute_1d_density(
                np.asarray(ds), bins, smooth, **density_kwargs)
            qs = ds.quantile(split_quantiles)
            median = ds.median().values
            title = measurement_from_sample(ds, **meas_title_kwargs)

        pdf = pdf / pdf.max() * violin_h / 2
        spl = CubicSpline(x, pdf)
        x = np.linspace(min(x[0], extend_to[0]), max(x[-1], extend_to[1]), x.size)

        sections = zip(
            np.concatenate([qs[:1], qs[1:] + quantile_gap / 2]),
            np.concatenate([qs[1:-1] - quantile_gap / 2, qs[-1:]])
        )
        for q0, q1 in sections:
            _x = np.linspace(q0, q1, 400)
            _pdf = spl(_x)

            props.setdefault("lw", lw)
            props.setdefault("alpha", fill_alpha)
            collection = ax.fill_between(
                _x, y_center - _pdf, y_center + _pdf,
                **props,
            )

        color = collection.get_facecolor()

        if measurement_kind == "upper":
            q = qs[-1]
            pre_title = f"{meas_label}: " if meas_label is not None else ""
            ax.text(
                max(q + text_dq, min_q_upper_label), y_center,
                f"{pre_title}${q:.3f}$",
                ha="left", va="center_baseline",
                color=color,
                **measurement_kwargs,
            )
        elif measurement_kind == "med_quant":
            pre_title = f"{meas_label}: " if meas_label is not None else ""
            ax.text(
                median,
                y_center + _pdf.max() + measurement_pad,
                pre_title + title,
                va="bottom", ha="center", color=color,
                **measurement_kwargs,
            )
        if label is not None:
            from matplotlib.transforms import blended_transform_factory
            ax.text(
                -label_pad, y_center, label,
                ha="right", va="center",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                **label_kwargs, color=color,
            )

        y_center += 1

    # infer whether current xaxis is shared and won't display labels
    tp = ax.xaxis.get_tick_params()
    # https://github.com/matplotlib/matplotlib/issues/27416
    if tp.get("labelbottom", tp.get("labelleft")) and not ax.get_xlabel():
        try:
            ax.set_xlabel(label_from_attrs(dsets[0]))
        except AttributeError:
            pass  # not a DataArray

    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    return ax


__all__ = [
    "get_2d_level",
    "plot_autocorr_evolution",
    "plot_trace_2d",
    "plot_1d_dist",
    "plot_2d_dist",
    "plot_joint_dist",
    "compare_1d_dists",
    "compare_2d_dists",
    "test_smoothing",
    "plot_1d_dists",
    "plot_violin",
]
