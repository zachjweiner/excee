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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator
from corner.core import (
    hist2d, _set_xlim, _set_ylim, gaussian_filter, quantile,
)
from excee.analysis import plot_1d_hist, _make_title, _init_kwargs_dict


def _init_dict_with_default(inpt, keys, default):
    if not isinstance(inpt, dict):
        default = inpt or default
        kwargs = {}
    else:
        kwargs = inpt
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


def corner_impl(
    data,
    rows=None,
    cols=None,
    *,
    ensure_1d_hists=True,
    hist_kind="kde",
    bins=20,
    ranges=None,
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
    scale_hist=False,
    quantiles=None,
    title_quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    rotate_ticks=True,
    configure_tick_locators=True,
    reverse=False,
    hist_kwargs=None,
    resize_fig=False,
    force_range=None,
    whspace=0.05,
    panel_dim=2,
    **hist2d_kwargs,
):
    if isinstance(data, np.ndarray):
        if labels is not None:
            data = dict(zip(labels, data.T))
        else:
            data = dict(zip(map(str, np.arange(data.shape[-1])), data.T))

    cols = cols or list(data.keys())
    rows = rows or cols

    if reverse:
        # to match corner's old behavior
        rows = rows[::-1]
        cols = cols[::-1]

    rowcols = assemble_rowcols(
        rows, cols,
        reverse=reverse, ensure_1d_hists=ensure_1d_hists,
    )
    nrow, ncol = rowcols.shape
    all_keys = np.unique(recfunctions.structured_to_unstructured(rowcols))
    all_keys = [key for key in all_keys if key]

    quantiles = quantiles or []
    title_quantiles = title_quantiles or quantiles or [0.15865525, 0.5, 0.84134475]

    if show_titles and len(title_quantiles) != 3:
        raise ValueError(
            "'title_quantiles' must contain exactly three values; "
            "pass a length-3 list or array using the 'title_quantiles' argument"
        )

    try:
        label_dict = {
            key: data[key].attrs.get("long_name", key)
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

    if force_range is None:
        # if force_range is not passed, default to True if ranges are passed
        force_range = ranges is not None
    _keys = list(set(all_keys) & set(data.keys()))
    minmax = {k: (data[k].min(), data[k].max()) for k in _keys}
    ranges = minmax | _init_kwargs_dict(ranges)
    hist_bin_factor = _init_dict_with_default(hist_bin_factor, all_keys, 1)

    if color is None:
        color = mpl.rcParams["ytick.color"]

    # Set up the default histogram keywords.
    if hist_kind == "hist":
        hist_kwargs = _init_kwargs_dict(hist_kwargs)
        hist_kwargs.setdefault("color", color)
        if smooth1d is None:
            hist_kwargs.setdefault("histtype", "step")
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
        weights = np.asarray(data["weights"])

    new_fig = fig is None
    if fig is None:
        with plt.style.context({"figure.constrained_layout.use": False}):
            fig, axes = plt.subplots(nrow, ncol, squeeze=False)
    else:
        axes = np.array(fig.axes).reshape((nrow, ncol))
    if new_fig or resize_fig:
        fig = set_figure_layout(
            fig, nrow, ncol, reverse, panel_dim=panel_dim, whspace=whspace)

    for (i, j), (row, col) in np.ndenumerate(rowcols):
        ax = axes[i, j]
        axis_had_no_content = not axis_has_content(ax)

        if row not in data or col not in data:
            if axis_had_no_content:
                ax.axis("off")
            continue
        else:
            ax.axis("on")

        y = data[row]
        x = data[col]

        if row != col:
            hist2d(
                np.asarray(x),
                np.asarray(y),
                ax=ax,
                range=[ranges[col], ranges[row]],
                axes_scale=[axes_scale[col], axes_scale[row]],
                weights=weights,
                color=color,
                smooth=smooth,
                bins=[bins[col], bins[row]],
                new_fig=new_fig,
                force_range=force_range or axis_had_no_content,
                **hist2d_kwargs,
            )
        elif hist_kind == "hist":
            # Plot the histograms.
            n_bins_1d = int(max(1, np.round(hist_bin_factor[col] * bins[col])))
            if axes_scale[col] == "linear":
                bins_1d = np.linspace(
                    min(ranges[col]), max(ranges[col]), n_bins_1d + 1
                )
            elif axes_scale[col] == "log":
                bins_1d = np.logspace(
                    np.log10(min(ranges[col])),
                    np.log10(max(ranges[col])),
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
                if gaussian_filter is None:
                    raise ImportError("Please install scipy for smoothing")
                n, _ = np.histogram(x, bins=bins_1d, weights=weights)
                n = gaussian_filter(n, smooth1d)
                x0 = np.array(list(zip(bins_1d[:-1], bins_1d[1:]))).flatten()
                y0 = np.array(list(zip(n, n))).flatten()
                ax.plot(x0, y0, **hist_kwargs)

            # Plot quantiles if wanted.
            if len(quantiles) > 0:
                qvalues = quantile(x, quantiles, weights=weights)
                for q in qvalues:
                    ax.axvline(q, ls="dashed", color=color)

                if verbose:
                    print("Quantiles:")  # noqa: T201
                    print(list(zip(quantiles, qvalues)))  # noqa: T201

            if scale_hist:
                maxn = np.max(n)
                _set_ylim(
                    force_range or axis_had_no_content, new_fig,
                    ax, [-0.1 * maxn, 1.1 * maxn]
                )
            else:
                _set_ylim(
                    force_range or axis_had_no_content, new_fig,
                    ax, [0, 1.1 * np.max(n)]
                )

        elif hist_kind == "kde":
            # FIXME: subsume hist plotting branch into call to plot_1d_hist
            plot_1d_hist(
                ax, np.asarray(data[col]), weights=weights,
                kind="kde", axes_scale=axes_scale[col],
                quantiles=quantiles, **kde_kwargs,
            )
            ax.set_ylim(ymin=0)

        if row == col:
            if show_titles:
                # FIXME: auto align titles to left/right if reverse when too wide
                title = _make_title(
                    np.asarray(data[col]),
                    title_quantiles, weights=weights,
                    err_prec=err_prec, rescale_thresh=rescale_thresh,
                    label=label_dict[col],
                    style=title_style,
                )
                ax.set_title(title, **title_kwargs)

            ax.set_xscale(axes_scale[col])
            _set_xlim(force_range or axis_had_no_content, new_fig, ax, ranges[col])
            ax.yaxis.set_major_locator(NullLocator())

        # formatting
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        elif configure_tick_locators:
            if axes_scale[col] == "linear":
                ax.xaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )
            elif axes_scale[col] == "log":
                ax.xaxis.set_major_locator(
                    LogLocator(numticks=max_n_ticks)
                )
            if row != col:
                if axes_scale[row] == "linear":
                    ax.yaxis.set_major_locator(
                        MaxNLocator(max_n_ticks, prune="lower")
                    )
                elif axes_scale[row] == "log":
                    ax.yaxis.set_major_locator(
                        LogLocator(numticks=max_n_ticks)
                    )

        if (i < nrow - 1 and not reverse) or (i > 0 and reverse):
            if top_ticks and row == col:
                ax.xaxis.set_ticks_position("top")
            else:
                ax.set_xticklabels([])
                ax.set_xticklabels([], minor=True)
        else:
            ax.set_xlabel(label_dict[col], **xlabel_kwargs)

        if ((j > 0 and not reverse) or (j < ncol - 1 and reverse)) and row != col:
            ax.set_yticklabels([])
            ax.set_yticklabels([], minor=True)
        elif row != col:
            ax.set_ylabel(label_dict[row], **ylabel_kwargs)

    if rotate_ticks:
        for ax in axes.flat:
            ax.tick_params(which="both", labelrotation=45)

    if truths is not None:
        try:
            _ = truths[cols[0]]
        except (TypeError, IndexError):
            truths = dict(zip(cols, truths))

        for (i, j), (row, col) in np.ndenumerate(rowcols):
            ax = axes[i, j]

            if row not in truths and col not in truths:
                continue

            if col in truths:
                axes[i, j].axvline(truths[col], color=truth_color)
            if row in truths and row != col:
                axes[i, j].axhline(truths[row], color=truth_color)
                axes[i, j].plot(
                    truths[col], truths[row],
                    color=truth_color,
                    linestyle="None",
                    marker="s",
                )

    # ranges controls the actual binning
    # limits independently sets/overrides axes limits
    if limits is not None:
        for (i, j), (row, col) in np.ndenumerate(rowcols):
            ax = axes[i, j]
            if row in limits and row != col:
                ax.set_ylim(*limits[row])
            if col in limits:
                ax.set_xlim(*limits[col])

    if ticks is not None:
        for (i, j), (row, col) in np.ndenumerate(rowcols):
            ax = axes[i, j]
            if row in ticks and row != col:
                ax.set_yticks(ticks[row])
            if col in ticks:
                ax.set_xticks(ticks[col])

    return fig, axes
