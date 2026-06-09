__copyright__ = """
Copyright (c) 2013-2022 Dan Foreman-Mackey
Copyright (C) 2024 Zachary J Weiner
"""

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
from numpy.lib import recfunctions
from scipy.interpolate import CubicSpline
from arviz_stats.base import array_stats
from excee.stats import autocorr_time, hdi, eti
from excee.density import compute_1d_density, compute_2d_density
from excee.plot.titles import measurement_from_sample, parse_ci_input
from excee.util import _init_kwargs_dict, label_from_attrs

_find_hdi_contours = array_stats._find_hdi_contours

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    mpl = None
    plt = None

import logging
logger = logging.getLogger(__name__)


def get_2d_level(sigma):
    return 1 - np.exp(-1/2 * np.asarray(sigma)**2)


def sigma_from_2d_level(level):
    return np.sqrt(- 2 * np.log(1 - np.asarray(level)))


def plot_2d_density(ax, X, Y, pdf, color,
                    *, levels=None,
                    plot_contours=True, fill_contours=True, shade_background=True,
                    plot_density=False, density_kwargs=None,
                    gapcolor=None, gap_linestyle="--", fill_alphas=None,
                    contour_kwargs=None, contourf_kwargs=None, alpha_xx=0.5):
    contour_kwargs = _init_kwargs_dict(contour_kwargs)
    contour_kwargs.setdefault("colors", [color])
    contourf_kwargs = _init_kwargs_dict(contourf_kwargs)
    contourf_kwargs.setdefault("antialiased", False)

    if levels is None:
        levels = get_2d_level(np.arange(1, 3))

    V = _find_hdi_contours(pdf, levels[::-1])

    from matplotlib.colors import LinearSegmentedColormap, colorConverter

    if shade_background:
        base_color = ax.get_facecolor()
        base_cmap = LinearSegmentedColormap.from_list(
            "base_cmap", [base_color, base_color], N=2
        )
        ax.contourf(
            X, Y, pdf, [V.min(), pdf.max()],
            cmap=base_cmap,
            antialiased=False,
        )

    if fill_contours:
        rgba_color = colorConverter.to_rgba(color)
        contour_cmap = [list(rgba_color) for _ in levels] + [rgba_color]
        if fill_alphas is None:
            _n = len(levels)
            fill_alphas = (np.arange(_n) + 1 + alpha_xx) / (_n + alpha_xx)
        for i, _alpha in enumerate(fill_alphas):
            contour_cmap[i][-1] *= _alpha

        ax.contourf(
            X, Y, pdf, np.concatenate([V, [pdf.max()]]),
            colors=contour_cmap,
            **contourf_kwargs,
        )
    elif plot_density:
        density_cmap = LinearSegmentedColormap.from_list(
            "density_cmap",
            [color, colorConverter.to_rgba(ax.get_facecolor(), alpha=0)]
        )
        _default = {"cmap": density_cmap, "antialiased": True, "rasterized": True}
        density_kwargs = _default | _init_kwargs_dict(density_kwargs)
        ax.pcolormesh(X, Y, pdf.max() - pdf, **density_kwargs)

    if plot_contours:
        ax.contour(X, Y, pdf, V[:], **contour_kwargs)
        if gapcolor is not None:
            kw = contour_kwargs | {
                "linestyles": [gap_linestyle], "colors": [gapcolor],
            }
            ax.contour(X, Y, pdf, V[:], **kw)

    return ax


def plot_2d_dist(ax, data, color, *, weights=None, bw_method="isj",
                 axes_scale="linear", bins=None, smooth=None,
                 cholesky_whitening=True, bounds=None, force_bounds=False,
                 lcv_threshold=0.22, lcv_frac=0.15, pad_nstd=None,
                 plot_datapoints=False, datapoint_kwargs=None,
                 **kwargs):
    axes_scale = [axes_scale]*2 if isinstance(axes_scale, str) else axes_scale
    if any(scale != "linear" for scale in axes_scale):
        _data = data.copy()
        for i, scale in enumerate(axes_scale):
            if scale == "log":
                _data[i] = np.log(_data[i])
    else:
        _data = data

    smooth = np.full((2,), smooth)
    smooth = np.where(smooth == None, 0, smooth)  # noqa: E711
    bins = np.full((2,), bins)
    bins = np.where(bins == None, np.where(smooth == 0, 20, 256), bins)  # noqa: E711
    logger.info(
        f"smooth = ({smooth[0]}, {smooth[1]}), bins = ({bins[0]}, {bins[1]})"
    )

    if plot_datapoints:
        _defaults = {
            "color": color, "alpha": 0.1, "linestyle": "None",
            "marker": "o", "markersize": 2, "markeredgecolor": "None",
            "rasterized": True, "zorder": -1,
        }
        data_kwargs = _defaults | _init_kwargs_dict(datapoint_kwargs)
        ax.plot(np.ravel(_data[0]), np.ravel(_data[1]), **data_kwargs)

    X, Y, Z = compute_2d_density(
        _data, bins, smooth, weights=weights, bw_method=bw_method,
        cholesky_whitening=cholesky_whitening,
        bounds=bounds, force_bounds=force_bounds,
        lcv_threshold=lcv_threshold, lcv_frac=lcv_frac, pad_nstd=pad_nstd,
    )
    if axes_scale[0] == "log":
        X = np.exp(X)
    if axes_scale[1] == "log":
        Y = np.exp(Y)

    return plot_2d_density(ax, X, Y, Z, color=color, **kwargs)


def plot_1d_dist(ax, data, *, weights=None, ess=None, bw_method="isj",
                 axes_scale="linear", bins=None, smooth=None,
                 bounds=None, force_bounds=False, boundary_correction="linear",
                 lcv_threshold=0.22, lcv_frac=0.15, pad_nstd=None,
                 plot_ci=True, ci_kind="auto", default_ci_kind="hdi",
                 ci_prob=None, quantile_kwargs=None,
                 norm="relative", side="bottom", label=None,
                 color=None, alpha=None, ci_alpha=None,
                 line_kwargs=None, fill_kwargs=None,
                 **kwargs):
    ci_kind, ci_prob = parse_ci_input(data, ci_kind, default_ci_kind, ci_prob)
    quantile_kwargs = _init_kwargs_dict(quantile_kwargs)

    alpha = alpha if alpha is not None else 0.2 if not plot_ci else 0.1
    ci_alpha = ci_alpha if ci_alpha is not None else 2 * alpha

    _data = np.log(data) if axes_scale == "log" else data

    smooth = 0 if smooth is None else smooth
    bins = (40 if smooth == 0 else 512) if bins is None else bins

    # FIXME: unify branches?
    if smooth == 0:
        if side != "bottom":
            raise NotImplementedError()

        hist, bin_edges = np.histogram(
            np.ravel(_data), bins=bins, density=norm == "density", weights=weights,
        )
        if axes_scale == "log":
            bin_edges = np.exp(bin_edges)
        if norm == "relative":
            hist = hist / np.max(hist)

        ax.bar(
            bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge",
            color=color, label=label, alpha=alpha, **kwargs,
        )
    else:
        if weights is not None:
            raise NotImplementedError("KDE with weights")

        x, y = compute_1d_density(
            _data, bins, smooth, weights=weights, ess=ess, bw_method=bw_method,
            bounds=bounds, force_bounds=force_bounds,
            boundary_correction=boundary_correction,
            lcv_threshold=lcv_threshold, lcv_frac=lcv_frac, pad_nstd=pad_nstd,
        )

        if axes_scale == "log":
            x = np.exp(x)
        if norm == "relative":
            y = y / np.max(y)

        rdata = np.ravel(data)
        med = np.quantile(rdata, 0.5, method="inverted_cdf", weights=weights)
        if ci_kind == "eti":
            low, high = eti(rdata, ci_prob, weights=weights)
        elif ci_kind == "hdi":
            low, high = hdi(rdata, ci_prob)
        elif ci_kind == "upper_limit":
            low = x[0]
            high = np.quantile(
                rdata, ci_prob, weights=weights, method="inverted_cdf")
        elif ci_kind == "lower_limit":
            low = np.quantile(
                rdata, 1-ci_prob, weights=weights, method="inverted_cdf")
            high = x[-1]
        elif ci_kind is not None:
            raise NotImplementedError(f"{ci_kind=}")

        if ci_kind is not None and plot_ci:
            # pylint: disable=E0606
            if axes_scale == "log":
                x_ci = np.geomspace(low, high, bins)
                spl = CubicSpline(np.log(x), y)
                y_ci = spl(np.log(x_ci))
                y_med = spl(np.log(med))
            else:
                x_ci = np.linspace(low, high, bins)
                spl = CubicSpline(x, y)
                y_ci = spl(x_ci)
                y_med = spl(med)

        if side == "top":
            y = -y
        elif side == "left":
            y, x = x, y
        elif side == "right":
            y, x = x, -y

        line_kwargs = _init_kwargs_dict(line_kwargs)
        lines = ax.plot(x, y, label=label, color=color, **line_kwargs, **kwargs)
        line_z = lines[0].get_zorder()
        _color = lines[0].get_color()

        fill_kwargs = _init_kwargs_dict(fill_kwargs)
        fill_kwargs.setdefault("zorder", line_z)
        fill_kwargs.setdefault("color", _color)
        fill_kwargs.setdefault("linewidth", 0)
        fill_alpha = fill_kwargs.setdefault("alpha", alpha)
        if side in ("left", "right"):
            ax.fill_betweenx(y, 0, x, **fill_kwargs, **kwargs)
        else:
            ax.fill_between(x, 0, y, **fill_kwargs, **kwargs)

        quantile_kwargs.setdefault("color", _color)
        quantile_kwargs.setdefault("alpha", (1 + fill_alpha) / 2)
        quantile_kwargs.setdefault("zorder", line_z)

        if ci_kind is not None and plot_ci:
            if side == "top":
                y_ci = -y_ci
            elif side == "left":
                y_ci, x_ci = x_ci, y_ci
            elif side == "right":
                y_ci, x_ci = x_ci, -y_ci

            fill_kwargs["alpha"] = ci_alpha
            if side in ("left", "right"):
                ax.fill_betweenx(y_ci, 0, x_ci, **fill_kwargs, **kwargs)
            else:
                ax.fill_between(x_ci, 0, y_ci, **fill_kwargs, **kwargs)

            if side == "bottom":
                ax.plot([med, med], [0, y_med], **quantile_kwargs)
            elif side == "top":
                ax.plot([med, med], [0, -y_med], **quantile_kwargs)
            elif side == "left":
                ax.plot([0, y_med], [med, med], **quantile_kwargs)
            elif side == "right":
                ax.plot([0, -y_med], [med, med], **quantile_kwargs)


def _set_xlim(ax, new_xlim, force=False):
    if force:
        return ax.set_xlim(new_xlim)
    xlim = ax.get_xlim()
    return ax.set_xlim([min(xlim[0], new_xlim[0]), max(xlim[1], new_xlim[1])])


def _set_ylim(ax, new_ylim, force=False):
    if force:
        return ax.set_ylim(new_ylim)
    ylim = ax.get_ylim()
    return ax.set_ylim([min(ylim[0], new_ylim[0]), max(ylim[1], new_ylim[1])])


def _init_dict_with_default(inpt, keys, default):
    if not isinstance(inpt, dict):
        default = inpt if inpt is not None else default
        kwargs = {}
    else:
        kwargs = inpt.copy()
    for key in keys:
        kwargs.setdefault(key, default)
    return kwargs


rowcol_dt = [("row", "<U32"), ("col", "<U32")]


def rc_dt(r, c):
    return np.array((r, c), dtype=rowcol_dt)


def assemble_rowcols(rows, cols, reverse=False, ensure_1d_dists=True):
    rslc = slice(None, None, -1) if reverse else slice(None)
    rowcols = [
        [
            (row, col)
            if not (
                col in rows
                and row in (cols[:j] if not reverse else cols[j+1:])
            )
            else ("", "")
            for j, col in enumerate(cols)
        ]
        for row in rows
    ]
    rowcols = np.asarray(rowcols, dtype=rowcol_dt)

    if ensure_1d_dists:
        all_keys = np.unique(recfunctions.structured_to_unstructured(rowcols))
        missing_1d = [
            var for var in all_keys
            if not np.isin(rc_dt(var, var), rowcols)
        ]
        need_new_row = [
            var
            for j, var in enumerate(cols[rslc])
            if (
                var in missing_1d
                and not np.isin(rc_dt("", ""), rowcols[:, j])
            )
        ]
        if need_new_row:
            items = [np.full(rowcols.shape[1], rc_dt("", "")), rowcols]
            if reverse:
                items = items[::-1]
            rowcols = np.vstack(items)
        for j, (var, col) in enumerate(zip(cols, rowcols.T)):
            if var not in rows:
                for i, rc in enumerate(col if reverse else col[::-1]):
                    if rc == rc_dt("", ""):
                        rowcols[i if reverse else -i-1, j] = (var, var)
                        break

    return rowcols


def set_figure_layout(fig, nrow, ncol, reverse, panel_dim=None, whspace=0.05):
    # FIXME: remove this entirely once constrained layout can be manipulated
    # as needed

    rc = np.array([ncol, nrow])
    # specifying in units of panel_dim; converting to subplots_adjust accordingly
    lbdim, trdim = (0.2, 0.5) if reverse else (0.5, 0.2)
    plotdim = rc + (rc - 1) * whspace
    figsize = lbdim + plotdim + trdim

    if panel_dim is None:
        width = plt.rcParams["figure.figsize"][0]
        fig.set_size_inches((width, width * figsize[1] / figsize[0]))
    else:
        fig.set_size_inches(panel_dim * (lbdim + plotdim + trdim))

    # figsize = (width, height) likewise for lb and tr
    lb = lbdim / figsize
    tr = (lbdim + plotdim) / figsize
    fig.subplots_adjust(
        # fractions of figure width
        left=lb[0], bottom=lb[1],
        right=tr[0], top=tr[1],
        # fractions of panel spacing
        wspace=whspace, hspace=whspace
    )

    if reverse:
        for ax in fig.axes:
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    return fig


def axis_has_content(ax):
    return bool(ax.lines + ax.images + ax.collections + ax.patches)


def get_ess(x):
    issue_warning = False
    import xarray as xr
    if isinstance(x, xr.DataArray):
        if "ess" in x.attrs:
            _ess = x.attrs["ess"]
        elif {"chain", "draw"} <= set(x.dims):
            N = x.sizes["chain"] * x.sizes["draw"]
            _ess = N / autocorr_time(x)[0].values[()]
        elif "sample" in x.dims:
            issue_warning = True
            _ess = x.sizes["sample"]
        else:
            raise RuntimeError()
    elif np.ndim(x) == 1:
        issue_warning = True
        _ess = np.shape(x)[-1]
    else:
        _ess = np.prod(np.shape(x)[-2:]) / autocorr_time(x)[0]

    if issue_warning and not get_ess.has_warned:
        logger.warning(
            "flattened chain detected; assuming all samples independent")
        get_ess.has_warned = True

    return float(_ess)


def plot_joint_dist(
    data,
    rows=None, cols=None,
    *,
    weights=None, ess=None, bounds=None,
    skip_1d=False, skip_2d=False,
    # alternative panel specification
    var_names=None, rowcols=None, ensure_1d_dists=True, reverse=False,
    # distributions
    bins=None, smooth=None,
    # plot style
    color=None, limits=None, axes_scale="linear", sideways_hists=False,
    # ticks
    ticks=None, max_n_ticks=5,
    top_ticks=False, rotate_ticks=True, configure_tick_locators=True,
    # labels and titles
    labels=None, label_kwargs=None, show_titles=False, title_kwargs=None,
    plot_ci=True, ci_kind="auto", ci_prob=None,
    # truths
    truths=None, truth_marker="s", truth_kwargs=None,
    # figure config
    fig=None, resize_fig=False, whspace=0.05, panel_dim=2,
    # kwargs passed along to plot_Nd_dist
    kwargs_1d=None,
    **kwargs_2d,
):
    if isinstance(data, np.ndarray):
        if labels is not None:
            data = dict(zip(labels, data.T))
        else:
            data = dict(zip(map(str, np.arange(data.shape[-1])), data.T))

    cols = (
        cols if cols is not None
        else var_names if var_names is not None  # for compat
        else list(data.keys())
    )
    rows = rows if rows is not None else cols

    if reverse:
        # to match corner's old behavior
        rows = rows[::-1]
        cols = cols[::-1]

    if rowcols is None:
        rowcols = assemble_rowcols(
            rows, cols,
            reverse=reverse, ensure_1d_dists=ensure_1d_dists,
        )
    nrow, ncol = rowcols.shape
    all_keys = np.unique(recfunctions.structured_to_unstructured(rowcols))
    all_keys = [key for key in all_keys if key]
    plot_keys = list(set(all_keys) & set(data.keys()))

    if color is None:
        color = mpl.rcParams["ytick.color"]

    kwargs_1d = _init_kwargs_dict(kwargs_1d)
    kwargs_1d.setdefault("color", color)
    kwargs_2d.setdefault("color", color)

    if weights is False:
        weights = None
    elif "weights" in data:
        weights = np.asarray(data["weights"])

    bins = _init_dict_with_default(bins, plot_keys, None)
    smooth = _init_dict_with_default(smooth, plot_keys, None)
    axes_scale = _init_dict_with_default(axes_scale, plot_keys, "linear")
    minmax = {k: np.asarray([data[k].min(), data[k].max()]) for k in plot_keys}
    bounds = _init_dict_with_default(bounds, plot_keys, None)
    ci_kind = _init_dict_with_default(ci_kind, plot_keys, "hdi")
    ci_prob = _init_dict_with_default(ci_prob, plot_keys, None)

    get_ess.has_warned = False
    if ess is None:
        ess = {str(key): get_ess(data[key]) for key in plot_keys}
        logger.info(f"ess: {ess}")
    else:
        for key in set(plot_keys) - set(ess.keys()):
            ess[key] = get_ess(data[key])

    try:
        label_dict = {
            key: label_from_attrs(data[key]) if key in data else key
            for key in plot_keys
        }
    except AttributeError:
        label_dict = {key: key if labels is not None else None for key in plot_keys}

    xlabel_kwargs = _init_kwargs_dict(label_kwargs)
    ylabel_kwargs = _init_kwargs_dict(label_kwargs)
    if reverse:
        ylabel_kwargs["rotation"] = -90
        ylabel_kwargs["va"] = "bottom"

    title_kwargs = _init_kwargs_dict(title_kwargs)
    if reverse:
        title_kwargs.setdefault("y", 0)
        title_kwargs.setdefault("va", "top")
        title_kwargs.setdefault("pad", -plt.rcParams["axes.titlepad"])

    measurement_kwargs = {
        key: val for key in ("err_prec", "rescale_thresh", "style")
        if (val := title_kwargs.pop(key, None)) is not None
    }

    new_fig = fig is None
    if fig is None:
        with plt.style.context({"figure.constrained_layout.use": False}):
            fig, axes = plt.subplots(nrow, ncol, squeeze=False)
    else:
        axes = np.array(fig.axes).reshape((nrow, ncol))
    if new_fig or resize_fig:
        fig = set_figure_layout(
            fig, nrow, ncol, reverse, panel_dim=panel_dim, whspace=whspace)

    if truths is not None:
        try:
            _ = truths.get(cols[0], None)
        except (TypeError, IndexError):
            truths = dict(zip(cols, truths))
    truth_kwargs = _init_kwargs_dict(truth_kwargs)
    truth_kwargs.setdefault("color", color)
    truth_kwargs.setdefault("linestyle", "None")

    for (i, j), (row, col) in np.ndenumerate(rowcols):
        ax = axes[i, j]
        axis_had_no_content = not axis_has_content(ax)

        if row not in data or col not in data:
            if axis_had_no_content:
                ax.axis("off")
            continue
        else:
            ax.axis("on")

        y = np.asarray(data[row])
        x = np.asarray(data[col])

        side = (
            None if row != col
            else "left" if sideways_hists and not reverse and j == ncol-1
            else "right" if sideways_hists and reverse and j == 0
            else "bottom"
        )

        if row != col:
            if skip_2d:
                continue
            logger.info(f"plotting 2D dist for ({row}, {col}) on axes[{i}, {j}]")
            plot_2d_dist(
                ax,
                np.stack([x, y], axis=0),
                bins=(bins[col], bins[row]),
                smooth=(smooth[col], smooth[row]),
                weights=weights,
                bounds=(bounds[col], bounds[row]),
                axes_scale=(axes_scale[col], axes_scale[row]),
                **kwargs_2d,
            )
        else:
            if skip_1d:
                continue
            logger.info(f"plotting 1D dist for {row} on axes[{i}, {j}]")
            plot_1d_dist(
                ax, x, weights=weights, ess=ess[col],
                bins=bins[col], smooth=smooth[col],
                axes_scale=axes_scale[col], bounds=bounds[col],
                plot_ci=plot_ci, ci_kind=ci_kind[col], ci_prob=ci_prob[col],
                side=side, **kwargs_1d,
            )
            if side in ("left", "right"):
                ax.autoscale(axis="x")  # to recalculate xmax
                ax.set_xlim(xmin=0)
            else:
                ax.autoscale(axis="y")  # to recalculate ymax
                ax.set_ylim(ymin=0)

        if truths is not None:
            if col in truths:
                if side in ("left", "right"):
                    axes[i, j].axhline(truths[col], **truth_kwargs)
                else:
                    axes[i, j].axvline(truths[col], **truth_kwargs)
            if row in truths and row != col:
                axes[i, j].axhline(truths[row], **truth_kwargs)
                if col in truths:
                    axes[i, j].plot(
                        truths[col], truths[row],
                        marker=truth_marker, **truth_kwargs,
                    )

        from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator

        def _locator(scale):
            return (
                NullLocator() if max_n_ticks == 0
                else MaxNLocator(max_n_ticks, prune="lower") if scale == "linear"
                else LogLocator(numticks=max_n_ticks) if scale == "log"
                else None
            )

        # titles, limits and tick locators
        if row == col:
            if show_titles:
                # FIXME: auto align titles to left/right if reverse when too wide
                title = measurement_from_sample(
                    x, weights=weights, ci_kind=ci_kind[col], ci_prob=ci_prob[col],
                    label=label_dict[col], **measurement_kwargs,
                )
                ax.set_title(title, **title_kwargs)

            if side in ("left", "right"):
                ax.set_yscale(axes_scale[col])

                if limits is not None and col in limits:
                    ax.set_ylim(limits[col])
                else:
                    _set_ylim(ax, minmax[col], force=axis_had_no_content)

                ax.xaxis.set_major_locator(NullLocator())
                if configure_tick_locators:
                    ax.yaxis.set_major_locator(_locator(axes_scale[col]))
            else:
                ax.set_xscale(axes_scale[col])

                if limits is not None and col in limits:
                    ax.set_xlim(limits[col])
                else:
                    _set_xlim(ax, minmax[col], force=axis_had_no_content)

                ax.yaxis.set_major_locator(NullLocator())
                if configure_tick_locators:
                    ax.xaxis.set_major_locator(_locator(axes_scale[col]))
        else:
            ax.set_xscale(axes_scale[col])
            ax.set_yscale(axes_scale[row])

            if limits is not None and col in limits:
                ax.set_xlim(limits[col])
            else:
                _set_xlim(ax, minmax[col], force=axis_had_no_content)
            if limits is not None and row in limits:
                ax.set_ylim(limits[row])
            else:
                _set_ylim(ax, minmax[row], force=axis_had_no_content)

            if configure_tick_locators:
                ax.xaxis.set_major_locator(_locator(axes_scale[col]))
                ax.yaxis.set_major_locator(_locator(axes_scale[row]))

        # tick positioning/removal
        if (i < nrow - 1 and not reverse) or (i > 0 and reverse):
            if top_ticks and row == col:
                if side in ("left", "right"):
                    ax.yaxis.set_ticks_position(
                        "left" if side == "right" else "right")
                else:
                    ax.xaxis.set_ticks_position(
                        "top" if side == "bottom" else "bottom")
            elif side in ("top", "bottom", None):
                ax.set_xticklabels([])
                ax.set_xticklabels([], minor=True)
        else:  # noqa: PLR5501
            if row != col or side in ("top", "bottom"):
                ax.set_xlabel(label_dict[col], **xlabel_kwargs)
            elif side in ("left", "right"):
                ax.set_yticklabels([])
                ax.set_yticklabels([], minor=True)

        if ((j > 0 and not reverse) or (j < ncol - 1 and reverse)) and row != col:
            ax.set_yticklabels([])
            ax.set_yticklabels([], minor=True)
        elif row != col:
            ax.set_ylabel(label_dict[row], **ylabel_kwargs)

        if ticks is not None:
            if col in ticks:
                if side in ("left", "right"):
                    ax.set_yticks(ticks[col])
                else:
                    ax.set_xticks(ticks[col])
            if row in ticks and row != col:
                ax.set_yticks(ticks[row])

    if rotate_ticks:
        for ax in axes.flat:
            ax.tick_params(which="both", labelrotation=45)

    return fig, axes
