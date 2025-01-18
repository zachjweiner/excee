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
import arviz as az
from xarray.plot.utils import label_from_attrs
from excee.util import ordered_union

_std_quantiles = (0.15865525, 0.5, 0.84134475)


def _get_long_names(data):
    return [label_from_attrs(da) for da in data.values()]
    # return [da.attrs.get("long_name", key) for key, da in data.items()]


def plot_autocorr_evolution(data, n0=100, nn=20, **kwargs):
    from excee.analysis import autocorr_time_over_time

    ns = np.geomspace(kwargs.get("discard", 0) + n0, data.sizes["draw"], nn)
    ns = ns.astype(int)
    tau = autocorr_time_over_time(data, ns, **kwargs)

    _names = _get_long_names(data)
    labels = [
        fr"{name}: {round(t) if np.isfinite(t) else 'NAN'}"
        for t, name in zip(tau[:, -1], _names)
    ]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.loglog(ns, tau.T, ".-", label=labels)
    ax.legend(title=r"$\tau_f$", loc="center left", bbox_to_anchor=(1, 0.5))
    return fig, ax


def plot_trace_2d(data, width=8, height=2, split_at=None, ratio=None,
                  cbar_kwargs=None, **kwargs):
    import matplotlib.pyplot as plt

    n = len(data)
    ncol = 1 if split_at is None else 2
    if ratio is None:
        ratio = 1 if split_at is None else split_at / data.sizes["draw"]

    fig, axes = plt.subplots(
        n, ncol, figsize=(width, n*height),
        sharex="col", sharey=True, squeeze=False,
        width_ratios=None if split_at is None else (ratio, 1)
    )

    cbar_kwargs = _init_kwargs_dict(cbar_kwargs)
    cbar_kwargs.setdefault("aspect", 10)
    if split_at is None:
        cbar_kwargs.setdefault("pad", 0.025)

    cbar_kwargs_left = cbar_kwargs.copy()
    cbar_kwargs_left.setdefault("location", "left")

    cbar_kwargs_right = cbar_kwargs.copy()
    cbar_kwargs_right.setdefault("label", None)
    cbar_kwargs_right.setdefault("pad", ratio * 0.1)

    for row, key in enumerate(data):
        arr = data[key]
        if split_at is not None:
            arr.isel(draw=slice(split_at)).plot(
                ax=axes[row, 0],
                cbar_kwargs=cbar_kwargs_left, **kwargs,
            )
            arr.isel(draw=slice(split_at, None)).plot(
                ax=axes[row, 1],
                cbar_kwargs=cbar_kwargs_right, **kwargs,
            )
        else:
            arr.plot(ax=axes[row, 0], cbar_kwargs=cbar_kwargs, **kwargs)

    for ax in axes.flat:
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    if split_at is None:
        wspace = 0
        for ax in axes[:, 0]:
            ax.set_ylabel("chain")
        axes[-1, 0].set_xlabel("draw")
    else:
        wspace = 0.025
        # fig.supxlabel("draw", y=0.125, va="top")
        for ax in axes.flat:
            ax.yaxis.set_ticklabels([])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=wspace)

    return fig, axes


def _init_kwargs_dict(kwargs):
    return {} if kwargs is None else kwargs.copy()


def plot_1d_hist(ax, sample, *, weights=None, kind="hist", axes_scale="linear",
                 relative=False, density=True, bins=20, range=None,
                 quantiles=(), quantile_kwargs=None, side="bottom",
                 label=None, color=None, line_kwargs=None, fill_kwargs=None,
                 kde_kwargs=None, **kwargs):
    quantile_kwargs = _init_kwargs_dict(quantile_kwargs)
    kde_kwargs = _init_kwargs_dict(kde_kwargs)

    from corner.core import quantile
    if range is not None:
        sample = sample[(range[0] < sample) & (sample < range[1])]
    _sample = np.log(sample) if axes_scale == "log" else sample
    qvalues = quantile(_sample, quantiles, weights=weights) if quantiles else ()
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

        import matplotlib as mpl
        ytick_color = mpl.rcParams["ytick.color"]
        quantile_kwargs.setdefault("color", color or ytick_color)
        for q in qvalues:
            ax.axvline(q, **quantile_kwargs)
    else:
        if weights is not None:
            raise NotImplementedError("KDE with weights")

        x, y = az.kde(_sample, **kde_kwargs)

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


def _exponent(x):
    return np.floor(np.log10(np.abs(x))).astype(int)


def format_measurement(x, quantiles, err_prec=2, rescale_thresh=2, weights=None):
    from corner.core import quantile
    q_lo, q_mid, q_hi = quantile(x, quantiles, weights=weights)
    q_m, q_p = q_mid - q_lo, q_hi - q_mid

    _exps = _exponent([q_m, q_p])
    if (
        max(*(_exps + 1), 0) <= max(err_prec, rescale_thresh)
        and max(*(-_exps - 1), 0) <= rescale_thresh
    ):
        rescale_exp = 0
    else:
        rescale_exp = max(*_exps, _exponent(q_mid))

    m_digits, p_digits = np.maximum(0, err_prec - (_exps - rescale_exp) - 1)
    mid_digits = max(m_digits, p_digits)

    m_str = f"{{:.{m_digits}f}}".format(q_m / 10.**rescale_exp)
    p_str = f"{{:.{p_digits}f}}".format(q_p / 10.**rescale_exp)
    mid_str = f"{{:.{mid_digits}f}}".format(q_mid / 10.**rescale_exp)
    title = fr"{mid_str}_{{-{m_str}}}^{{+{p_str}}}"

    return title, rescale_exp


def _make_title(x, quantiles, label=None, style="paren", **kwargs):
    title, rescale_exp = format_measurement(x, quantiles, **kwargs)
    lhs = f"{label} = " if label else ""
    if rescale_exp != 0:
        if style == "paren":
            title = lhs + fr"$\left( {title} \right) \times 10^{{{rescale_exp}}}$"
        elif style == "multiply" and label is not None:
            title = fr"$10^{{{-rescale_exp}}}$\,{{{label}}} = ${title}$"
        else:
            raise ValueError()
    else:
        title = (f"{label} = " if label else "") + f"${title}$"

    return title


def add_stacked_titles(axes, datasets, title_quantiles, var_names=None, colors=None,
                       title_loc="center", title_kwargs=None,
                       title_stack_pad_frac=0.2):
    labels = [_get_long_names(data) for data in datasets]

    title_kwargs = _init_kwargs_dict(title_kwargs)
    err_prec = title_kwargs.pop("err_prec", 2)
    rescale_thresh = title_kwargs.pop("rescale_thresh", 2)
    title_style = title_kwargs.pop("style", "paren")

    import matplotlib.pyplot as plt
    title_kwargs.setdefault("fontsize", plt.rcParams["axes.titlesize"])
    change_colors = "color" not in title_kwargs

    for i, ax in enumerate(axes):
        xycoords = None
        for data, _labels, color in zip(
            datasets[::-1], labels[::-1], colors[::-1]
        ):
            weights = data.get("weights")
            if var_names is not None:
                if var_names[i] not in data:
                    continue
                else:
                    x = data[var_names[i]].values.ravel()
                    label = _labels[list(data.keys()).index(var_names[i])]
            else:
                x = list(data.values())[i].values.ravel()
                label = _labels[i]

            title = _make_title(
                x, title_quantiles, weights=weights, label=label, err_prec=err_prec,
                rescale_thresh=rescale_thresh, style=title_style,
            )
            if change_colors:
                title_kwargs["color"] = color
            if xycoords is None:
                xycoords = ax.set_title(title, loc=title_loc, **title_kwargs)
            else:
                xycoords = ax.annotate(
                    title, (0, 1 + title_stack_pad_frac), xycoords=xycoords,
                    va="bottom", ha="left", **title_kwargs,
                )


def set_corner_limits(axes, limits):
    for i, lims in enumerate(limits):
        if lims is None:
            continue
        for ax in axes[i, :i]:
            ax.set_ylim(*lims)
        for ax in axes[i:, i]:
            ax.set_xlim(*lims)


def set_corner_ticks(axes, ticks, **kwargs):
    for i, tick in enumerate(ticks):
        if tick is None:
            continue
        for ax in axes[i, :i]:
            ax.set_yticks(tick, **kwargs)
        for ax in axes[i:, i]:
            ax.set_xticks(tick, **kwargs)


def plot_corner(data, *, color=None, quantiles=_std_quantiles, fill_contours=True,
                plot_contours=True, plot_density=False, plot_datapoints=False,
                hist_kind="kde", hist_kwargs=None, contour_kwargs=None,
                show_titles=True, title_kwargs=None, **kwargs):
    if color is None:
        import matplotlib as mpl
        color = mpl.rcParams["ytick.color"]

    from excee.analysis import expand_sample_to_chain_and_draw

    if hasattr(data, "sizes") and not set(data.sizes).issuperset({"chain", "draw"}):
        data = expand_sample_to_chain_and_draw(data)

    if hist_kind == "hist":
        hist_kwargs = _init_kwargs_dict(hist_kwargs)
        hist_kwargs.setdefault("histtype", "stepfilled")
        hist_kwargs.setdefault("alpha", 0.2)
        hist_kwargs.setdefault("density", True)
        hist_kwargs.setdefault("color", color)
    elif hist_kind == "kde":
        pass
    else:
        raise ValueError(f"{hist_kind=}")

    contour_kwargs = _init_kwargs_dict(contour_kwargs)
    contour_kwargs.setdefault("linewidths", 1)

    title_kwargs = _init_kwargs_dict(title_kwargs)

    if "cols" not in kwargs:
        kwargs["cols"] = kwargs.pop("var_names", None)

    from excee.corner import corner_impl
    fig, axes = corner_impl(
        data, quantiles=quantiles, color=color,
        fill_contours=fill_contours, plot_contours=plot_contours,
        plot_density=plot_density, plot_datapoints=plot_datapoints,
        hist_kwargs=hist_kwargs, hist_kind=hist_kind,
        show_titles=show_titles, title_kwargs=title_kwargs,
        contour_kwargs=contour_kwargs,
        **kwargs,
    )

    return fig, axes


def _get_n_colors(colors, n):
    import matplotlib.pyplot as plt
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


def compare_1d_posteriors(datasets, *, labels=None, var_names=None,
                          ncol=4, w=4, aspect=1,
                          axes_scale=None, ranges=None, limits=None,
                          colors=None, kind="hist", relative_hist=False,
                          show_titles=True, fig=None,
                          quantiles=_std_quantiles, title_kwargs=None,
                          title_loc="center", title_stack_pad_frac=0.2, **kwargs):
    if var_names is None:
        var_names = ordered_union([list(data.keys()) for data in datasets])
    if labels is None:
        labels = [None for _ in datasets]
    colors = _get_n_colors(colors, len(datasets))

    n = len(var_names)
    ncol = min(n, ncol)
    nrow = (n - 1) // ncol + 1

    import matplotlib.pyplot as plt

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
    ranges = _init_kwargs_dict(ranges)
    limits = _init_kwargs_dict(limits)

    title_quantiles = kwargs.pop("title_quantiles", quantiles or _std_quantiles)

    for data, label, color in zip(datasets, labels, colors):
        xlabels = dict(zip(data.keys(), _get_long_names(data)))
        weights = data.get("weights")
        for ax, key in zip(axes.flat, var_names):
            if key not in data:
                continue

            scale = axes_scale.get(key, "linear")
            ax.set_xlabel(xlabels[key])

            sample = data[key].values.ravel()
            plot_1d_hist(
                ax, sample, weights=weights, kind=kind, axes_scale=scale,
                relative=relative_hist, label=label, **kwargs, color=color,
                quantiles=quantiles, range=ranges.get(key, None),
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
        add_stacked_titles(
            axes.flat[:n], datasets, title_quantiles,
            var_names=var_names, colors=colors, title_loc=title_loc,
            title_kwargs=title_kwargs,
            title_stack_pad_frac=title_stack_pad_frac,
        )

    return fig, axes


def plot_1d_posterior(data, **kwargs):
    return compare_1d_posteriors([data], **kwargs)


def get_2d_level(sigma):
    return 1 - np.exp(-1/2 * sigma**2)


def compare_2d_posteriors(datasets, cols=None, rows=None,
                          colors=None, hist_kind="kde", relative_hist=False,
                          show_titles=True, title_kwargs=None, title_loc="center",
                          title_stack_pad_frac=0.2, fig=None, **kwargs):
    default_contour_kwargs = _init_kwargs_dict(kwargs.get("contour_kwargs"))
    kwargs.setdefault("levels", get_2d_level(np.arange(1, 2.1, 1)))

    colors = _get_n_colors(colors, len(datasets))

    cols = cols if cols is not None else kwargs.pop("var_names", None)
    if cols is None:
        cols = ordered_union([list(data.keys()) for data in datasets])

    rows = rows if rows is not None else cols

    ranges = kwargs.pop("ranges", None)
    bins = kwargs.pop("bins", None)
    smooth = kwargs.pop("smooth", None)

    for i, (data, color) in enumerate(zip(datasets, colors)):
        ds_kw = {}
        ds_kw["color"] = color
        contour_kwargs = default_contour_kwargs.copy()
        contour_kwargs.setdefault("colors", [color])
        ds_kw["contour_kwargs"] = contour_kwargs

        if hist_kind == "kde":
            ds_kw["hist_kwargs"] = {
                "line_kwargs": {"zorder": 2+i/1e3},
                "relative": relative_hist,
            }

        if bins:
            ds_kw["bins"] = bins[i] if isinstance(bins, list) else bins
        if ranges:
            ds_kw["ranges"] = ranges[i] if isinstance(ranges, list) else ranges
        if smooth:
            ds_kw["smooth"] = smooth[i] if isinstance(smooth, list) else smooth

        fig, axes = plot_corner(
            data, rows=rows, cols=cols, fig=fig, show_titles=False,
            hist_kind=hist_kind,
            force_range=i == 0,  # only force range the first time
            **kwargs, **ds_kw,
        )

    title_quantiles = kwargs.get(
        "title_quantiles",
        kwargs.get("quantiles", _std_quantiles)
    )

    from excee.corner import assemble_rowcols
    rowcols = assemble_rowcols(
        rows, cols,
        reverse=kwargs.get("reverse", False),
        ensure_1d_hists=kwargs.get("ensure_1d_hists", True),
    )

    hists = [
        [axes[idx], rc[0]]
        for idx, rc in np.ndenumerate(rowcols)
        if rc[0] == rc[1] and rc[0] != ""
    ]
    title_axes = [hist[0] for hist in hists]
    title_names = [hist[1] for hist in hists]
    if show_titles:
        add_stacked_titles(
            title_axes, datasets, title_quantiles,
            var_names=title_names, colors=colors, title_loc=title_loc,
            title_kwargs=title_kwargs,
            title_stack_pad_frac=title_stack_pad_frac,
        )

    return fig, axes
