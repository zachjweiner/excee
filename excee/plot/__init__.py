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
from excee.plot.titles import make_ci_str, add_stacked_title
from excee.stats import compute_ci
from excee.plot.diagnostic import plot_autocorr_evolution, plot_trace_2d
from excee.plot.dist import (
    get_1d_level, get_2d_level, sigma_from_2d_level,
    plot_1d_dist, plot_2d_dist, plot_joint_dist,
    _init_dict_with_default
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


def compare_1d_dists(datasets, *, labels=None, var_names=None,
                     ncol=4, w=4, aspect=1, fig=None,
                     bins=None, smooth=1, bounds=None,
                     axes_scale=None, limits="auto", limit_sigma=3,
                     plot_ci=True, ci_prob=None,
                     ci_kind="auto", default_ci_kind="hdi",
                     colors=None, show_titles=True, title_kwargs=None,
                     **kwargs):
    if var_names is None:
        var_names = ordered_union([list(data.keys()) for data in datasets])
    if labels is None:
        labels = [None for _ in datasets]
    colors = _get_n_colors(colors, len(datasets))

    if limits == "auto":
        _dsets = [ds[[k for k in var_names if k in ds]] for ds in datasets]
        limits = get_inclusive_limits(_dsets, sigma=limit_sigma)
    else:
        limits = _init_dict_with_default(limits, var_names, None)

    bins = _init_dict_with_default(bins, var_names, None)
    smooth = _init_dict_with_default(smooth, var_names, None)
    axes_scale = _init_dict_with_default(axes_scale, var_names, "linear")
    bounds = _init_dict_with_default(bounds, var_names, None)
    ci_kind = _init_dict_with_default(ci_kind, var_names, "hdi")
    ci_prob = _init_dict_with_default(ci_prob, var_names, None)

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

    for data, label, color in zip(datasets, labels, colors):
        xlabels = dict(zip(data.keys(), get_long_names(data)))
        weights = data.get("weights")
        for ax, key in zip(axes.flat, var_names):
            if key not in data:
                continue

            scale = axes_scale.get(key, "linear")
            ax.set_xlabel(xlabels[key])

            x = np.asarray(data[key])
            plot_1d_dist(
                ax, x, weights=weights,
                label=label, color=color,
                bins=bins[key], smooth=smooth[key],
                axes_scale=axes_scale[key], bounds=bounds[key],
                plot_ci=plot_ci, ci_prob=ci_prob[key],
                ci_kind=ci_kind[key], default_ci_kind=default_ci_kind,
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

    title_kwargs = _init_kwargs_dict(title_kwargs)
    format_kwargs = {
        key: title_kwargs.pop(key)
        for key in ("err_prec", "rescale_thresh", "style", "include_long_names")
        if key in title_kwargs
    }
    if show_titles:
        for ax, key in zip(axes.flat, var_names):
            arys = [ds.get(key) for ds in datasets]
            ci_kwargs = format_kwargs | {
                "ci_kind": ci_kind[key],
                "ci_prob": ci_prob[key],
                "default_ci_kind": default_ci_kind,
                "weights": weights,
            }
            titles = [make_ci_str(ary, **ci_kwargs) for ary in arys]
            add_stacked_title(ax, titles, colors=colors, **title_kwargs)

    return fig, axes


def plot_1d_dists(data, **kwargs):
    if "color" in kwargs:
        kwargs["colors"] = [kwargs.pop("color")]
    return compare_1d_dists([data], **kwargs)


def compare_2d_dists(datasets, cols=None, *, rows=None, rowcols=None, var_names=None,
                     bins=None, smooth=None, bounds=None, kwargs_1d=None,
                     colors=None, show_titles=True, title_kwargs=None,
                     ci_kind="auto", default_ci_kind="hdi", ci_prob=None,
                     exclude_1d_idx=None, exclude_2d_idx=None,
                     levels=None, limits="auto", limit_pad=1,
                     fig=None, contour_kwargs=None, **kwargs):
    exclude_1d_idx = exclude_1d_idx or []
    exclude_2d_idx = exclude_2d_idx or []
    default_contour_kwargs = _init_kwargs_dict(contour_kwargs)

    colors = _get_n_colors(colors, len(datasets))

    if rowcols is None:
        cols = cols if cols is not None else var_names
        if cols is None:
            cols = ordered_union([list(data.keys()) for data in datasets])

        rows = rows if rows is not None else cols

        from excee.plot.dist import assemble_rowcols
        rowcols = assemble_rowcols(
            rows, cols,
            reverse=kwargs.get("reverse", False),
            ensure_1d_dists=kwargs.get("ensure_1d_dists", True),
        )

    if levels is None:
        levels = get_2d_level(np.arange(1, 3))
    if isinstance(limits, str) and limits == "auto":
        all_keys = np.unique(np.lib.recfunctions.structured_to_unstructured(rowcols))
        all_keys = [key for key in all_keys if key]
        _dsets = [ds[[k for k in all_keys if k in ds]] for ds in datasets]
        limits = get_inclusive_limits_from_2d_levels(_dsets, levels, pad=limit_pad)

    ci_kind = _init_dict_with_default(ci_kind, all_keys, "auto")
    ci_prob = _init_dict_with_default(ci_prob, all_keys, None)

    for i, (data, color) in enumerate(zip(datasets, colors)):
        ds_kw = {}
        ds_kw["color"] = color
        contour_kwargs = default_contour_kwargs.copy()
        contour_kwargs.setdefault("colors", [color])
        ds_kw["contour_kwargs"] = contour_kwargs
        ds_kw["kwargs_1d"] = _init_kwargs_dict(kwargs_1d)

        if bins is not None:
            ds_kw["bins"] = bins[i] if isinstance(bins, list) else bins
        if smooth is not None:
            ds_kw["smooth"] = smooth[i] if isinstance(smooth, list) else smooth
        if bounds is not None:
            ds_kw["bounds"] = bounds[i] if isinstance(bounds, list) else bounds

        fig, axes = plot_joint_dist(
            data, rows=rows, cols=cols, rowcols=rowcols,
            levels=levels, limits=limits,
            ci_kind=ci_kind, ci_prob=ci_prob,
            fig=fig, show_titles=False,
            skip_1d=i in exclude_1d_idx,
            skip_2d=i in exclude_2d_idx,
            **kwargs, **ds_kw,
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
            for idx, rc in np.ndenumerate(rowcols)
            if rc[0] == rc[1] and rc[0] != ""
        ]
        for ax, key in axes_var_names:
            arys = [
                ds.get(key) if i not in exclude_1d_idx else None
                for i, ds in enumerate(datasets)
            ]
            ci_kwargs = format_kwargs | {
                "ci_kind": ci_kind[key],
                "ci_prob": ci_prob[key],
                "default_ci_kind": default_ci_kind,
            }
            titles = [
                make_ci_str(ary, **ci_kwargs) if ary is not None else None
                for ary in arys
            ]
            add_stacked_title(ax, titles, colors=colors, **title_kwargs)

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


def plot_violin(ax, arys, *, weights=None,
                bins=1024, smooth=1, density_kwargs=None,
                quantile_gap=None, gap_fraction=0.0025,
                violin_pad=0.1, text_dq=0.005, fill_kwargs=None,
                side_labels=None, side_label_kwargs=None, side_label_pad=0.005,
                plot_ci=True, ci_kind="hdi",
                default_ci_kind="eti",  # applies to splits for ci_kind="limit"
                title_kwargs=None, title_pad=0.05,
                include_long_names=False, label=None,
                min_x_upper_label=-np.inf, max_x_lower_label=np.inf):
    density_kwargs = _init_kwargs_dict(density_kwargs)

    def _get_density(ary):
        if (
            isinstance(ary, tuple)
            or (isinstance(ary, np.ndarray) and ary.ndim == 2)
        ):
            coord, pdf = ary
        else:
            coord, pdf = compute_1d_density(
                np.asarray(ary), bins, smooth, weights=weights, **density_kwargs)
        da = xr.DataArray(pdf, dims="x", coords={"x": coord})
        try:
            da = da.assign_attrs(ary.attrs)
        except AttributeError:
            pass
        return da

    densities = [_get_density(ary) for ary in arys]

    def _get_splits(da):
        xmin, xmax = da.x[0].values, da.x[-1].values
        if not plot_ci:
            return np.array([xmin, xmax])

        ci1 = compute_ci(
            da, "density",
            ci_kind=default_ci_kind if "limit" in ci_kind else ci_kind,
            ci_prob=get_1d_level(1), weights=weights,
        )
        if ci_kind in ("eti", "hdi"):
            ci2 = compute_ci(
                da, "density", ci_kind=ci_kind, ci_prob=get_1d_level(2),
                weights=weights,
            )
            splits = [*ci1, *ci2, xmin, xmax]
        elif "limit" in ci_kind:
            lim = compute_ci(
                da, "density", ci_kind=ci_kind, ci_prob=get_1d_level(2),
                weights=weights,
            )
            splits = [*ci1, lim, xmin if "upper" in ci_kind else xmax]
        elif ci_kind is None:
            splits = [xmin, xmax]
        else:
            raise NotImplementedError(f"{ci_kind=}")

        return np.sort(np.unique(splits))

    splits = [_get_splits(density) for density in densities]

    title_kwargs = _init_kwargs_dict(title_kwargs)
    ci_fmt_kw = {"include_long_names": include_long_names} | {
        key: title_kwargs.pop(key)
        for key in ("err_prec", "rescale_thresh", "style")
        if key in title_kwargs
    }
    title_kwargs.setdefault("fontsize", "small")
    title_kwargs.setdefault("clip_on", True)

    def _get_title(da):
        if ci_kind is not None:
            return make_ci_str(
                da, input_kind="density", ci_kind=ci_kind,
                ci_prob=get_1d_level(2 if "limit" in ci_kind else 1),
                default_ci_kind=default_ci_kind, weights=weights,
                label=label or label_from_attrs(da), **ci_fmt_kw,
            )
        else:
            return None

    titles = [_get_title(density) for density in densities]

    violin_h = 1 - violin_pad

    if quantile_gap is None:
        # FIXME: drop limits setting in favor of compare_violin wrapper
        if ax.get_autoscale_on():
            xmin = np.min(splits)
            xmax = np.max(splits)
        else:
            # axes limits have (presumably) already been set manually
            xmin, xmax = ax.get_xlim()
        quantile_gap = gap_fraction * (xmax - xmin)

    side_label_kwargs = _init_kwargs_dict(side_label_kwargs)
    side_label_kwargs.setdefault("fontsize", "small")

    if side_labels is None:
        side_labels = [None] * len(arys)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    fill_kwargs = _init_kwargs_dict(fill_kwargs)

    y_center = 0.
    _iter = zip(densities, splits, titles, prop_cycle, side_labels)
    for pdf, split, title, props, side_label in _iter:
        pdf = pdf / pdf.max() * violin_h / 2
        spl = CubicSpline(pdf.x, pdf)

        sections = zip(
            np.concatenate([split[:1], split[1:] + quantile_gap / 2]),
            np.concatenate([split[1:-1] - quantile_gap / 2, split[-1:]])
        )
        for x0, x1 in sections:
            _x = np.linspace(x0, x1, 400)
            _pdf = spl(_x)

            _fill_kwargs = {"lw": 0, "alpha": 1} | props | fill_kwargs
            collection = ax.fill_between(
                _x, y_center - _pdf, y_center + _pdf,
                **_fill_kwargs,
            )

        color = collection.get_facecolor()

        if ci_kind == "upper_limit":
            q = split[-1]
            ax.text(
                max(q + text_dq, min_x_upper_label), y_center,
                title,
                ha="left", va="center_baseline",
                color=color,
                **title_kwargs,
            )
        elif ci_kind == "lower_limit":
            q = split[0]
            ax.text(
                min(q - text_dq, max_x_lower_label), y_center,
                title,
                ha="right", va="center_baseline",
                color=color,
                **title_kwargs,
            )
        elif ci_kind in ("eti", "hdi"):
            center = split[split.size // 2] if plot_ci else pdf.idxmax().item()
            ax.text(
                center,
                y_center + pdf.max() + title_pad,
                title,
                va="bottom", ha="center", color=color,
                **title_kwargs,
            )
        if side_label is not None:
            from matplotlib.transforms import blended_transform_factory
            ax.text(
                -side_label_pad, y_center, side_label,
                ha="right", va="center",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                **side_label_kwargs, color=color,
            )

        y_center += 1

    # infer whether current xaxis is shared and won't display labels
    tp = ax.xaxis.get_tick_params()
    # https://github.com/matplotlib/matplotlib/issues/27416
    if tp.get("labelbottom", tp.get("labelleft")) and not ax.get_xlabel():
        try:
            ax.set_xlabel(label_from_attrs(arys[0]))
        except AttributeError:
            pass  # not a DataArray

    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    return ax


__all__ = [
    "get_1d_level",
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
