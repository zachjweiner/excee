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


def compare_1d_dists(datasets, style="standard", *, var_names=None,
                     labels=None, side_labels=None,
                     ncol=4, w=None, aspect=1, fig=None,
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

    limits = _init_dict_with_default(limits, var_names, "auto")
    _autos = [key for key in var_names if limits[key] == "auto"]
    if _autos:
        _dsets = [ds[[k for k in _autos if k in ds]] for ds in datasets]
        _alims = get_inclusive_limits(_dsets, sigma=limit_sigma)
        limits |= {key: np.asarray(_alims[key]) for key in _autos}

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
        w = 4 if w is None and style != "violin" else w
        if w is None:
            figsize = plt.rcParams["figure.figsize"]
        else:
            h = 20 if style == "violin" else w / aspect
            figsize = (w*ncol, h*nrow)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        if style != "violin":
            for ax in axes.flat:
                ax.set_box_aspect(1/aspect)

    for ax in axes.flat[n:]:
        ax.axis("off")

    if style == "violin":
        if side_labels is None:
            side_labels = [None] * len(datasets)

        for col, (ax, key) in enumerate(zip(axes.flat, var_names)):
            if (lims := limits.get(key, None)) is not None:
                ax.set_xlim(lims)
            _ = plot_violin(
                ax, [ds[key] for ds in datasets[::-1]],
                side_labels=side_labels[::-1] if col % ncol == 0 else None,
                # bins=bins[key], smooth=smooth[key],
                # axes_scale=axes_scale[key], bounds=bounds[key],
                plot_ci=plot_ci,
                # ci_prob=ci_prob[key],
                ci_kind=ci_kind[key] if ci_kind[key] != "auto" else default_ci_kind,
                default_ci_kind=default_ci_kind,
                title_kwargs=title_kwargs,
                **kwargs,
            )
    else:
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

    all_keys = np.unique(np.lib.recfunctions.structured_to_unstructured(rowcols))
    all_keys = [key for key in all_keys if key]
    ci_kind = _init_dict_with_default(ci_kind, all_keys, "auto")
    ci_prob = _init_dict_with_default(ci_prob, all_keys, None)

    if levels is None:
        levels = get_2d_level(np.arange(1, 3))

    limits = _init_dict_with_default(limits, all_keys, "auto")
    _autos = [key for key in all_keys if limits[key] == "auto"]
    if _autos:
        _dsets = [ds[[k for k in _autos if k in ds]] for ds in datasets]
        _alims = get_inclusive_limits_from_2d_levels(_dsets, levels, pad=limit_pad)
        limits |= {key: np.asarray(_alims[key]) for key in _autos}

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
                relative_height=1, fill_kwargs=None, gap_fraction=0.0025,
                title_pad=0.3, interviolin_pad=0.5, title_kwargs=None,
                plot_ci=True, ci_kind="hdi",
                default_ci_kind="eti",  # applies to splits for ci_kind="limit"
                include_long_names=False, label=None, limit_xpad_fraction=0.01,
                min_x_upper_label=-np.inf, max_x_lower_label=np.inf,
                side_labels=None, side_label_kwargs=None, side_label_pad=0.0075):

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
    title_kwargs.setdefault("clip_on", True)

    from matplotlib.font_manager import FontProperties
    fontsize_spec = title_kwargs.get("fontsize", plt.rcParams["font.size"])
    fs_pts = FontProperties(size=fontsize_spec).get_size_in_points()

    pad_label_pts = title_pad * fs_pts
    pad_next_pts = interviolin_pad * fs_pts
    density_height = relative_height * fs_pts
    text_height = 1.25 * fs_pts

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

    xspan = (
        np.max(splits) - np.min(splits) if ax.get_autoscalex_on()
        else np.diff(ax.get_xlim())[0]
        # axes limits have (presumably) already been set manually
    )

    quantile_gap = gap_fraction * xspan
    text_xpad = limit_xpad_fraction * xspan

    side_label_kwargs = _init_kwargs_dict(side_label_kwargs)
    if side_labels is None:
        side_labels = [None] * len(arys)

    has_top_labels = ci_kind in ("eti", "hdi") and any(t is not None for t in titles)

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

    prop_cycle = plt.rcParams["axes.prop_cycle"]()
    fill_kwargs = _init_kwargs_dict(fill_kwargs)

    from matplotlib.transforms import Affine2D, blended_transform_factory

    _iter = enumerate(zip(densities, splits, titles, prop_cycle, side_labels))
    for i, (pdf, split, title, props, side_label) in _iter:
        pdf = pdf / pdf.max() / 2
        spl = CubicSpline(pdf.x, pdf)

        baseline = bottom_buffer + i * step_pts

        y_layout_trans = Affine2D().scale(1, density_height).translate(0, baseline)
        trans = blended_transform_factory(
            ax.transData,
            y_layout_trans + ax.transData,
        )

        sections = zip(
            np.concatenate([split[:1], split[1:] + quantile_gap / 2]),
            np.concatenate([split[1:-1] - quantile_gap / 2, split[-1:]])
        )
        for x0, x1 in sections:
            _x = np.linspace(x0, x1, 400)
            _pdf = spl(_x)
            collection = ax.fill_between(
                _x, -_pdf, _pdf,
                transform=trans,
                **({"lw": 0, "alpha": 1} | props | fill_kwargs),
            )

        color = collection.get_facecolor()

        if title is not None:
            if ci_kind == "upper_limit":
                q = split[-1]
                kw = {"ha": "left", "va": "center_baseline"} | title_kwargs
                ax.text(
                    max(q + text_xpad, min_x_upper_label), baseline, title,
                    color=color, transform=ax.transData, **kw,
                )
            elif ci_kind == "lower_limit":
                q = split[0]
                kw = {"ha": "right", "va": "center_baseline"} | title_kwargs
                ax.text(
                    min(q - text_xpad, max_x_lower_label), baseline, title,
                    color=color, transform=ax.transData, **kw,
                )
            elif ci_kind in ("eti", "hdi"):
                center = split[split.size // 2] if plot_ci else pdf.idxmax().item()
                text_y = baseline + density_height / 2 + pad_label_pts
                kw = {"ha": "center", "va": "bottom"} | title_kwargs
                ax.text(
                    center, text_y, title,
                    color=color, transform=ax.transData, **kw,
                )

        if side_label is not None:
            kw = {"ha": "right", "va": "center"} | side_label_kwargs
            ax.text(
                -side_label_pad, baseline, side_label,
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                color=color,
                **kw,
            )

    # infer whether current xaxis is shared and won't display labels
    tp = ax.xaxis.get_tick_params()
    # https://github.com/matplotlib/matplotlib/issues/27416
    if tp.get("labelbottom", tp.get("labelleft")) and not ax.get_xlabel():
        for ary in arys:
            try:
                ax.set_xlabel(label_from_attrs(ary))
                break
            except AttributeError:
                pass  # not a DataArray

    ax.set_yticks([])
    ax.set_yticks([], minor=True)

    return ax


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
    "compare_1d_dists",
    "compare_2d_dists",
    "test_smoothing",
    "plot_1d_dists",
    "plot_violin",
]
