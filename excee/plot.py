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
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
import arviz_stats as az
from excee.util import ordered_union, label_from_attrs
from excee.density import get_2d_level

_std_quantiles = (0.15865525, 0.5, 0.84134475)

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    mpl = None
    plt = None


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

    fig, ax = plt.subplots()
    ax.loglog(ns, tau.T, ".-", label=labels)
    ax.legend(title=r"$\tau_f$", loc="center left", bbox_to_anchor=(1, 0.5))
    return fig, ax


def plot_trace_2d(data, width=8, height=2, split_at=None, ratio=None,
                  cbar_kwargs=None, subplots_layout="tight", **kwargs):

    n = len(data)
    ncol = 1 if split_at is None else 2
    if ratio is None:
        ratio = 1 if split_at is None else split_at / data.sizes["draw"]

    fig, axes = plt.subplots(
        n, ncol, figsize=(width, n*height),
        sharex="col", sharey=True, squeeze=False,
        width_ratios=None if split_at is None else (ratio, 1),
        layout=subplots_layout,
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

    if subplots_layout == "tight":
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=wspace)

    return fig, axes


def _init_kwargs_dict(kwargs):
    return {} if kwargs is None else kwargs.copy()


def plot_1d_hist(ax, sample, *, weights=None, kind="kde", axes_scale="linear",
                 relative=False, density=True, bins=20, range=None,
                 quantiles=(), quantile_kwargs=None, side="bottom",
                 label=None, color=None, line_kwargs=None, fill_kwargs=None,
                 kde_kwargs=None, **kwargs):
    quantile_kwargs = _init_kwargs_dict(quantile_kwargs)
    kde_kwargs = _init_kwargs_dict(kde_kwargs)

    if range is not None:
        sample = sample[(range[0] < sample) & (sample < range[1])]
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

        x, y, _ = az.kde(np.asarray(_sample), **kde_kwargs)

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


def quantiles_from_log_pdf(log_pdf, x, quantiles):
    pdf = np.exp(log_pdf - log_pdf.max())
    slc = np.where(pdf > 1e-20)  # FIXME: smarter way?
    pdf = pdf[slc]
    x = x[slc]
    pdf /= simpson(pdf, x=x)
    cdf = CubicSpline(x, pdf).antiderivative()

    return np.array([cdf.solve(q, extrapolate=False).squeeze() for q in quantiles])


def _exponent(x):
    return np.floor(np.log10(np.abs(x))).astype(int)


def format_measurement(quantiles, err_prec=2, rescale_thresh=2,
                       label=None, style="paren"):
    q_lo, q_mid, q_hi = quantiles
    q_m, q_p = q_mid - q_lo, q_hi - q_mid

    _exps = _exponent([q_m, q_p])
    if (
        max(*(_exps + 1), 0) <= min(err_prec, rescale_thresh)
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
    meas = fr"{mid_str}_{{-{m_str}}}^{{+{p_str}}}"

    lhs = f"{label} = " if label else ""
    if rescale_exp != 0:
        if style == "paren":
            meas = lhs + fr"$\left( {meas} \right) \times 10^{{{rescale_exp}}}$"
        elif style == "multiply" and label is not None:
            meas = fr"$10^{{{-rescale_exp}}}$\,{{{label}}} = ${meas}$"
        else:
            raise ValueError()
    else:
        meas = (f"{label} = " if label else "") + f"${meas}$"

    return meas


def measurement_from_sample(sample, quantiles=_std_quantiles, weights=None,
                            **kwargs):
    qs = np.quantile(sample, quantiles, weights=weights, method="inverted_cdf")
    return format_measurement(qs, **kwargs)


def measurement_from_log_pdf(log_pdf, x, quantiles=_std_quantiles, **kwargs):
    qs = quantiles_from_log_pdf(log_pdf, x, quantiles)
    return format_measurement(qs, **kwargs)


def add_stacked_titles(axes, datasets, title_quantiles, var_names=None, colors=None,
                       title_loc="center", title_kwargs=None,
                       title_stack_pad_frac=0.2, include_long_names=True):
    labels = [_get_long_names(data) for data in datasets]

    title_kwargs = _init_kwargs_dict(title_kwargs)
    err_prec = title_kwargs.pop("err_prec", 2)
    rescale_thresh = title_kwargs.pop("rescale_thresh", 2)
    title_style = title_kwargs.pop("style", "paren")

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
            if not include_long_names:
                label = None

            title = measurement_from_sample(
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
                          colors=None, kind="kde", relative_hist=False,
                          show_titles=True, fig=None,
                          quantiles=_std_quantiles, title_kwargs=None,
                          title_loc="center", title_stack_pad_frac=0.2,
                          include_long_names=True, **kwargs):
    if var_names is None:
        var_names = ordered_union([list(data.keys()) for data in datasets])
    if labels is None:
        labels = [None for _ in datasets]
    colors = _get_n_colors(colors, len(datasets))

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
    ranges = _init_kwargs_dict(ranges)
    limits = _init_kwargs_dict(limits)

    title_quantiles = kwargs.pop(
        "title_quantiles",
        quantiles if quantiles is not None and len(quantiles) == 3
        else _std_quantiles
    )

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
            include_long_names=include_long_names,
        )

    return fig, axes


def plot_1d_posterior(data, **kwargs):
    return compare_1d_posteriors([data], **kwargs)


def compare_2d_posteriors(datasets, cols=None, rows=None, rowcols=None,
                          colors=None, hist_kind="kde", relative_hist=False,
                          show_titles=True, title_kwargs=None, title_loc="center",
                          title_stack_pad_frac=0.2, include_long_names=True,
                          exclude_1d_idx=None, exclude_2d_idx=None,
                          fig=None, **kwargs):
    exclude_1d_idx = exclude_1d_idx or []
    exclude_2d_idx = exclude_2d_idx or []
    default_contour_kwargs = _init_kwargs_dict(kwargs.pop("contour_kwargs", None))
    kwargs.setdefault("levels", get_2d_level(np.arange(1, 2.1, 1)))

    colors = _get_n_colors(colors, len(datasets))

    if rowcols is None:
        cols = cols if cols is not None else kwargs.pop("var_names", None)
        if cols is None:
            cols = ordered_union([list(data.keys()) for data in datasets])

        rows = rows if rows is not None else cols

        from excee.corner import assemble_rowcols
        rowcols = assemble_rowcols(
            rows, cols,
            reverse=kwargs.get("reverse", False),
            ensure_1d_hists=kwargs.get("ensure_1d_hists", True),
        )

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

        if bins is not None:
            ds_kw["bins"] = bins[i] if isinstance(bins, list) else bins
        if ranges is not None:
            ds_kw["ranges"] = ranges[i] if isinstance(ranges, list) else ranges
        if smooth is not None:
            ds_kw["smooth"] = smooth[i] if isinstance(smooth, list) else smooth

        fig, axes = plot_corner(
            data, rows=rows, cols=cols, rowcols=rowcols,
            fig=fig, show_titles=False,
            hist_kind=hist_kind,
            # only force range the first time
            # FIXME: drop this and just let corner autodetect no content?
            force_range=i == 0,
            skip_1d=i in exclude_1d_idx,
            skip_2d=i in exclude_2d_idx,
            **kwargs, **ds_kw,
        )

    if show_titles:
        title_quantiles = kwargs.get(
            "title_quantiles",
            kwargs.get("quantiles", _std_quantiles)
        )

        hists = [
            [axes[idx], rc[0]]
            for idx, rc in np.ndenumerate(rowcols)
            if rc[0] == rc[1] and rc[0] != ""
        ]
        title_axes = [hist[0] for hist in hists]
        title_names = [hist[1] for hist in hists]
        _datasets = [ds for i, ds in enumerate(datasets) if i not in exclude_1d_idx]
        _colors = [c for i, c in enumerate(colors) if i not in exclude_1d_idx]
        add_stacked_titles(
            title_axes, _datasets, title_quantiles,
            var_names=title_names, colors=_colors, title_loc=title_loc,
            title_kwargs=title_kwargs,
            title_stack_pad_frac=title_stack_pad_frac,
            include_long_names=include_long_names,
        )

    return fig, axes


def plot_violin(ax, dsets, *,
                split_quantiles=None, extend_to=(np.inf, -np.inf),
                quantile_gap=None, gap_fraction=0.0025,
                violin_pad=0.1, text_dq=0.005, fill_alpha=1, lw=0,
                labels=None, label_kwargs=None, label_pad=0.005,
                measurement_kind=None, measurement_labels=None,
                measurement_kwargs=None, measurement_pad=0.05,
                min_q_upper_label=-np.inf):
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
    meas_title_kwargs = {}
    meas_title_kwargs["err_prec"] = measurement_kwargs.pop("err_prec", 2)
    meas_title_kwargs["rescale_thresh"] = measurement_kwargs.pop("rescale_thresh", 3)
    meas_title_kwargs["style"] = measurement_kwargs.pop("style", "paren")
    meas_title_kwargs["quantiles"] = measurement_kwargs.pop(
        "quantiles", _std_quantiles)

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
            x, pdf, _ = az.kde(np.asarray(ds))
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
                clip_on=True,
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
