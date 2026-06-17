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


from collections.abc import Mapping
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


def compare_2d_dists(dsets, cols=None, *, rows=None, rowcols=None, var_names=None,
                     bins=None, smooth=None, bounds=None, axes_scale="linear",
                     levels=None, limits="auto", limit_pad=1,
                     kwargs_1d=None, kwargs_2d=None, density_kwargs=None,
                     exclude_1d_idx=None, exclude_2d_idx=None,
                     ci_kind="auto", default_ci_kind="hdi", ci_prob=None,
                     colors=None, show_titles=True, title_kwargs=None,
                     fig=None, contour_kwargs=None, **kwargs):
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
            ("contour_kwargs", contour_kwargs, {}),
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
        ds_kw["contour_kwargs"].setdefault("colors", [ds_kw["color"]])
        fig, axes = plot_joint_dist(
            data, rows=rows, cols=cols, rowcols=rowcols,
            levels=levels, limits=limits,
            fig=fig, show_titles=False,
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


def compare_1d_dists(dsets, *, var_names=None,
                     smooth=1, limits="auto", limit_sigma=3,
                     ncol=4, remove_1d_spines=True, **kwargs):
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
        dsets, rowcols=rowcols, smooth=smooth,
        limits=limits, remove_1d_spines=remove_1d_spines,
        **kwargs,
    )

    return fig, axes


def plot_1d_dists(data, **kwargs):
    if "color" in kwargs:
        kwargs["colors"] = [kwargs.pop("color")]
    return compare_1d_dists([data], **kwargs)


def compare_violin(dsets, *, var_names=None,
                   side_labels=None, ncol=4, fig=None,
                   limits="auto", limit_sigma=3,
                   bins=1024, smooth=1, bounds=None, axes_scale="linear",
                   ci_kind="hdi", default_ci_kind="eti",  # ci_prob=None,
                   **kwargs):
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
            **kwargs,
        )
    for ax in axes.flat[n:]:
        ax.axis("off")

    return fig, axes


def plot_violin(ax, arys, *, input_kind="sample", colors=None, alphas=1,
                bins=1024, smooth=1, bounds=None, axes_scale="linear",
                density_kwargs=None, fill_kwargs=None,
                relative_height=1, title_pad=0.3, interviolin_pad=0.5,
                gap_fraction=0.0025,
                show_titles=True, title_kwargs=None,
                plot_ci=True, ci_kind="hdi",  # ci_prob=None,
                default_ci_kind="eti",  # applies to splits for ci_kind="limit"
                include_long_names=False, label=None, limit_xpad_fraction=0.01,
                min_x_upper_label=-np.inf, max_x_lower_label=np.inf,
                side_labels=None, side_label_kwargs=None, side_label_pad=0.0075):

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
