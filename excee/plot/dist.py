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


from itertools import pairwise
import numpy as np
from numpy.lib import recfunctions
from scipy.ndimage import gaussian_filter
from arviz_stats.base import array_stats
from excee.density import compute_1d_density, compute_2d_density
from excee.plot.titles import measurement_from_sample
from excee.util import _init_kwargs_dict

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


def plot_2d_density(ax, X, Y, pdf, color,
                    *, levels=None,
                    plot_contours=True, fill_contours=True, shade_background=True,
                    plot_density=False, plot_datapoints=False,
                    gapcolor=None, gap_linestyle="--",
                    contour_kwargs=None, contourf_kwargs=None, alpha_xx=0.5):
    contour_kwargs = _init_kwargs_dict(contour_kwargs)
    contour_kwargs.setdefault("colors", [color])
    contourf_kwargs = _init_kwargs_dict(contourf_kwargs)
    contourf_kwargs.setdefault("antialiased", False)

    if levels is None:
        levels = get_2d_level(np.arange(1, 3))

    V = _find_hdi_contours(pdf, levels[::-1])

    if shade_background:
        base_color = ax.get_facecolor()
        from matplotlib.colors import LinearSegmentedColormap
        base_cmap = LinearSegmentedColormap.from_list(
            "base_cmap", [base_color, base_color], N=2
        )
        ax.contourf(
            X, Y, pdf, [V.min(), pdf.max()],
            cmap=base_cmap,
            antialiased=False,
        )

    if plot_contours:
        ax.contour(X, Y, pdf, V[:], **contour_kwargs)
        if gapcolor is not None:
            kw = contour_kwargs | {
                "linestyles": [gap_linestyle], "colors": [gapcolor],
            }
            ax.contour(X, Y, pdf, V[:], **kw)

    if fill_contours:
        from matplotlib.colors import colorConverter
        rgba_color = colorConverter.to_rgba(color)
        contour_cmap = [list(rgba_color) for _ in levels] + [rgba_color]
        for i, _ in enumerate(levels):
            contour_cmap[i][-1] *= (i + 1 + alpha_xx) / (len(levels) + alpha_xx)

        ax.contourf(
            X, Y, pdf, np.concatenate([V, [pdf.max()]]),
            colors=contour_cmap,
            **contourf_kwargs,
        )

    if plot_density:
        raise NotImplementedError("plot_density")

    if plot_datapoints:
        raise NotImplementedError("plot_datapoints")

    return ax


def plot_2d_dist(ax, data, color, *, weights=None,
                 bins=256, smooth_factor=None, use_kdepy=False,
                 pad_nstd=4, axes_scale="linear", _cholesky=True, **kwargs):
    X, Y, Z = compute_2d_density(
        data, weights=weights, bins=bins, smooth_factor=smooth_factor,
        use_kdepy=use_kdepy, pad_nstd=pad_nstd, axes_scale=axes_scale,
        _cholesky=_cholesky,
    )
    return plot_2d_density(ax, X, Y, Z, color=color, **kwargs)


def plot_1d_dist(ax, sample, *, weights=None, kind="kde", axes_scale="linear",
                 relative=False, density=True, bins=20,
                 quantiles=(), quantile_kwargs=None, side="bottom",
                 label=None, color=None, line_kwargs=None, fill_kwargs=None,
                 kde_kwargs=None, **kwargs):
    quantile_kwargs = _init_kwargs_dict(quantile_kwargs)
    kde_kwargs = _init_kwargs_dict(kde_kwargs)

    _sample = np.log(sample) if axes_scale == "log" else sample
    qvalues = (
        np.quantile(_sample, quantiles, weights=weights, method="inverted_cdf")
        if quantiles is not None else ()
    )
    if axes_scale == "log":
        qvalues = np.exp(qvalues)

    if kind == "hist":
        if side != "bottom":
            raise NotImplementedError()

        hist, bin_edges = np.histogram(
            _sample, bins=bins, density=density, weights=weights,
        )
        if axes_scale == "log":
            bin_edges = np.exp(bin_edges)
        if relative:
            hist /= np.max(hist)

        ax.bar(
            bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge",
            color=color, label=label, **kwargs,
        )
        ax.set_xscale(axes_scale)

        quantile_kwargs.setdefault("ls", "dashed")

        ytick_color = mpl.rcParams["ytick.color"]
        quantile_kwargs.setdefault("color", color or ytick_color)
        for q in qvalues:
            ax.axvline(q, **quantile_kwargs)
    else:
        if weights is not None:
            raise NotImplementedError("KDE with weights")

        x, y = compute_1d_density(_sample, **kde_kwargs)

        if axes_scale == "log":
            x = np.exp(x)
        if relative:
            y /= np.max(y)

        ymaxes = (
            np.interp(np.log(qvalues), np.log(x), y) if axes_scale == "log"
            else np.interp(qvalues, x, y)
        )

        line_kwargs = _init_kwargs_dict(line_kwargs)
        if side == "top":
            y = -y
        elif side == "left":
            y, x = x, y
        elif side == "right":
            y, x = x, -y
        lines = ax.plot(x, y, label=label, color=color, **line_kwargs, **kwargs)
        line_z = lines[0].get_zorder()
        _color = lines[0].get_color()

        fill_kwargs = _init_kwargs_dict(fill_kwargs)
        fill_kwargs.setdefault("zorder", line_z)
        fill_kwargs.setdefault("color", _color)
        fill_alpha = fill_kwargs.setdefault("alpha", kwargs.pop("alpha", 0.2))
        if side in ("left", "right"):
            ax.fill_betweenx(y, 0, x, **fill_kwargs, **kwargs)
        else:
            ax.fill_between(x, 0, y, **fill_kwargs, **kwargs)

        # quantile_kwargs.setdefault("color", "white")
        quantile_kwargs.setdefault("color", _color)
        quantile_kwargs.setdefault("alpha", (1 + fill_alpha) / 2)
        quantile_kwargs.setdefault("zorder", line_z)
        for q, ymax in zip(qvalues, ymaxes):
            if side == "bottom":
                ax.plot([q, q], [0, ymax], **quantile_kwargs)
            elif side == "top":
                ax.plot([q, q], [0, -ymax], **quantile_kwargs)
            elif side == "left":
                ax.plot([0, ymax], [q, q], **quantile_kwargs)
            elif side == "right":
                ax.plot([0, -ymax], [q, q], **quantile_kwargs)


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
        default = inpt or default
        kwargs = {}
    else:
        kwargs = inpt.copy()
    for key in keys:
        kwargs.setdefault(key, default)
    return kwargs


rowcol_dt = [("row", "<U32"), ("col", "<U32")]


def rc_dt(r, c):
    return np.array((r, c), dtype=rowcol_dt)


def assemble_rowcols(rows, cols, reverse=False, ensure_1d_hists=True):
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

    if ensure_1d_hists:
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


def plot_joint_dist(
    data,
    rows=None,
    cols=None,
    *,
    var_names=None,
    rowcols=None,
    ensure_1d_hists=True,
    hist_kind="kde",
    bins=20,
    limits=None,
    ticks=None,
    axes_scale="linear",
    weights=None,
    color=None,
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    labels=None,
    label_kwargs=None,
    show_titles=False,
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    truth_marker_kwargs=None,
    quantiles=None,
    title_quantiles=None,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    rotate_ticks=True,
    configure_tick_locators=True,
    reverse=False,
    hist_kwargs=None,
    resize_fig=False,
    sideways_hists=False,
    whspace=0.05,
    panel_dim=2,
    skip_1d=False,
    skip_2d=False,
    **hist2d_kwargs,
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
            reverse=reverse, ensure_1d_hists=ensure_1d_hists,
        )
    nrow, ncol = rowcols.shape
    all_keys = np.unique(recfunctions.structured_to_unstructured(rowcols))
    all_keys = [key for key in all_keys if key]

    quantiles = quantiles if quantiles is not None else []
    title_quantiles = (
        title_quantiles if title_quantiles is not None
        else quantiles if quantiles is not None
        else [0.15865525, 0.5, 0.84134475]
    )

    if show_titles and len(title_quantiles) != 3:
        raise ValueError(
            "'title_quantiles' must contain exactly three values; "
            "pass a length-3 list or array using the 'title_quantiles' argument"
        )

    try:
        from excee.util import label_from_attrs
        label_dict = {
            key: label_from_attrs(data[key]) if key in data else key
            for key in all_keys
        }
    except AttributeError:
        label_dict = {key: key if labels is not None else None for key in all_keys}

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

    err_prec = title_kwargs.pop("err_prec", 2)
    rescale_thresh = title_kwargs.pop("rescale_thresh", 2)
    title_style = title_kwargs.pop("style", "paren")

    bins = _init_dict_with_default(bins, all_keys, 20)
    axes_scale = _init_dict_with_default(axes_scale, all_keys, "linear")

    _keys = list(set(all_keys) & set(data.keys()))
    minmax = {k: np.asarray([data[k].min(), data[k].max()]) for k in _keys}
    hist_bin_factor = _init_dict_with_default(hist_bin_factor, all_keys, 1)

    if color is None:
        color = mpl.rcParams["ytick.color"]

    # FIXME: unify, put behind plot_1d_dist
    if hist_kind == "hist":
        hist_kwargs = _init_kwargs_dict(hist_kwargs)
        hist_kwargs.setdefault("color", color)
        hist_kwargs.setdefault("density", True)
        if smooth1d is None:
            hist_kwargs.setdefault("histtype", "stepfilled")
            hist_kwargs.setdefault("alpha", 0.2)

        kde_kwargs = {}
    elif hist_kind == "kde":
        kde_kwargs = _init_kwargs_dict(hist_kwargs)
        kde_kwargs.setdefault("color", color)
        hist_kwargs = {}
    else:
        raise ValueError(f"{hist_kind=}")

    if weights is False:
        weights = None
    elif "weights" in data:
        weights = np.asarray(data["weights"]).ravel()

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
    truth_marker_kwargs = _init_kwargs_dict(truth_marker_kwargs)
    truth_marker_kwargs.setdefault("marker", "s")
    truth_marker_kwargs.setdefault("color", truth_color)
    truth_marker_kwargs.setdefault("linestyle", "None")

    for (i, j), (row, col) in np.ndenumerate(rowcols):
        ax = axes[i, j]
        axis_had_no_content = not axis_has_content(ax)

        if row not in data or col not in data:
            if axis_had_no_content:
                ax.axis("off")
            continue
        else:
            ax.axis("on")

        y = np.asarray(data[row]).ravel()
        x = np.asarray(data[col]).ravel()

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
                np.stack([x, y], axis=-1),
                bins=[bins[col], bins[row]],
                axes_scale=[axes_scale[col], axes_scale[row]],
                weights=weights,
                color=color,
                smooth_factor=smooth if smooth is not None else 0,
                **hist2d_kwargs,
            )
        elif hist_kind == "hist" and not skip_1d:
            if sideways_hists:
                raise NotImplementedError()
            # Plot the histograms.
            n_bins_1d = int(max(1, np.round(hist_bin_factor[col] * bins[col])))
            if axes_scale[col] == "linear":
                bins_1d = np.linspace(
                    min(minmax[col]), max(minmax[col]), n_bins_1d + 1
                )
            elif axes_scale[col] == "log":
                bins_1d = np.logspace(
                    np.log10(min(minmax[col])),
                    np.log10(max(minmax[col])),
                    n_bins_1d + 1
                )
            else:
                raise ValueError(
                    f"Scale {axes_scale[col]} for dimension {col} not supported."
                    + " Use 'linear' or 'log'."
                )
            if smooth1d is None:
                n, _, _ = ax.hist(x, bins=bins_1d, weights=weights, **hist_kwargs)
            else:
                n, _ = np.histogram(x, bins=bins_1d, weights=weights)
                n = gaussian_filter(n, smooth1d)
                x0 = np.array(list(pairwise(bins_1d))).flatten()
                y0 = np.array(list(zip(n, n))).flatten()
                ax.plot(x0, y0, **hist_kwargs)

            # Plot quantiles if wanted.
            if len(quantiles) > 0:
                qvalues = np.quantile(
                    x, quantiles, weights=weights, method="inverted_cdf")
                for q in qvalues:
                    ax.axvline(q, ls="dashed", color=color)

            _set_ylim(ax, [0, 1.1 * np.max(n)], force=axis_had_no_content)

        elif hist_kind == "kde" and not skip_1d:
            logger.info(f"plotting 1D dist for {row} on axes[{i}, {j}]")
            # FIXME: subsume hist plotting branch into call to plot_1d_dist
            plot_1d_dist(
                ax, x, weights=weights,
                kind="kde", axes_scale=axes_scale[col],
                quantiles=quantiles, side=side, **kde_kwargs,
            )
            if side in ("left", "right"):
                ax.autoscale(axis="x")  # to recalculate xmax
                ax.set_xlim(xmin=0)
            else:
                ax.autoscale(axis="y")  # to recalculate ymax
                ax.set_ylim(ymin=0)

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
                    x, title_quantiles, weights=weights,
                    err_prec=err_prec, rescale_thresh=rescale_thresh,
                    label=label_dict[col],
                    style=title_style,
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

        if truths is not None:
            if col in truths:
                if side in ("left", "right"):
                    axes[i, j].axhline(truths[col], color=truth_color)
                else:
                    axes[i, j].axvline(truths[col], color=truth_color)
            if row in truths and row != col:
                axes[i, j].axhline(truths[row], color=truth_color)
                if col in truths:
                    axes[i, j].plot(
                        truths[col], truths[row],
                        **truth_marker_kwargs,
                    )

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
