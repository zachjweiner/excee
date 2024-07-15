__copyright__ = "Copyright (C) 2023 Zachary J Weiner"

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


from dataclasses import dataclass, field
from functools import cached_property
import re
import numpy as np
import xarray as xr
import arviz as az
from emcee import EnsembleSampler
from emcee.autocorr import integrated_time
from emcee.backends import HDFBackend
from excee.util import (
    union_dicts, ordered_union, ordered_intersection, read_pickle_from_h5,
    grouped_map
)


def autocorr_time(data, discard=0, thin=1, n=None, quiet=True, **kwargs):
    dat = data.sel(draw=slice(discard, n, thin))
    nchain = dat.sizes["chain"]
    ndraw = dat.sizes["draw"]
    dat = dat.transpose("draw", "chain", ...)

    x = np.concatenate(
        [da.values.reshape(ndraw, nchain, -1) for da in dat.values()],
        axis=-1
    )
    return thin * integrated_time(x, quiet=quiet, **kwargs)


def autocorr_time_over_time(data, ns, tol=0, **kwargs):
    result = np.empty((len(data), len(ns)))
    for i, n in enumerate(ns):
        result[:, i] = autocorr_time(data, n=n, tol=tol, **kwargs)

    return result


def get_random_sample(data, axis, num_samples, rng):
    rng = np.random.default_rng(None if rng is True else rng)
    slc = rng.choice(len(data[axis]), size=num_samples, replace=False)
    return data.isel({axis: slc})


def get_sample(data, discard, thin, flat=False, rng=False):
    if rng is False:  # 0 is a valid seed
        data = data.isel(draw=slice(discard, None, thin))
        if flat:
            data = data.stack(sample=("chain", "draw"))
    else:
        data = data.isel(draw=slice(discard, None))

        if flat:
            data = data.stack(sample=("chain", "draw"))
            axis = "sample"
        else:
            axis = "draw"

        num_samples = data.sizes[axis] // thin
        data = get_random_sample(data, axis, num_samples, rng)

    return data


def filter_outliers(sample, nstd, thresh=0.99, max_iter=10, min_iter=2):
    if sample.ndim == 1:
        sample = sample[:, None]

    for i in range(max_iter):
        nsamples = sample.shape[0]
        _thresh = min(thresh, 1 - 1 / nsamples)

        mean = np.median(sample, axis=0)
        std = np.std(sample, axis=0)
        sample = sample[np.all(abs(sample - mean) < nstd * std, axis=1)]

        if sample.shape[0] / nsamples > _thresh and i + 1 >= min_iter:
            break

    return sample.squeeze()


def expand_sample_to_chain_and_draw(dset):
    n = dset.sizes["sample"]
    dset = dset.drop_vars(["chain", "sample", "draw"])
    dset = dset.rename_dims({"sample": "draw"})
    dset = dset.assign_coords(draw=np.arange(n))
    dset = dset.expand_dims({"chain": [1]}, axis=0)
    return dset


def filter_outliers_dset(dset, nstd, thresh=0.99, max_iter=10, min_iter=2):
    if isinstance(nstd, float | int):
        nstd = [-nstd, nstd]

    if "sample" not in dset.dims:
        dset = dset.stack(sample=["chain", "draw"])

    for i in range(max_iter):
        nsamples = dset.sizes["sample"]
        _thresh = min(thresh, 1 - 1 / nsamples)

        med = dset.median()
        std = dset.std()
        delta = (dset - med) / std
        mask = (nstd[0] < delta) & (delta < nstd[1])
        mask = mask.to_array().all(["variable"])
        dset = dset.where(mask, drop=True)
        if dset.sizes["sample"] / nsamples > _thresh and i + 1 >= min_iter:
            break

    return expand_sample_to_chain_and_draw(dset)


def split_vector_vars(data, keep_dims=("chain", "draw")):
    if set(data.dims) == set(keep_dims):
        return data

    def split_one(da):
        dims_to_split = set(da.dims) - set(keep_dims)
        if len(dims_to_split) > 1:
            raise NotImplementedError("multi-dimensional splitting")
        elif dims_to_split:
            dim, = dims_to_split

            if da[dim].dtype.kind == "i":
                prefix = re.sub("_dim_[0-9]", "", dim)
                da[dim] = [f"{prefix}_{i}" for i in da[dim].values]

            dset = da.to_dataset(dim).copy()

            for name, var in dset.items():
                prefix, idx = re.findall("([a-zA-z]+)_([0-9]+)", name)[0]
                if long_name := da.attrs.get("long_name"):
                    prefix = long_name.replace("$", "")
                var.attrs["long_name"] = f"${prefix}_{{{idx}}}$"
        else:
            dset = da.copy()

        return dset

    das = [split_one(da) for da in data.values()]

    return xr.merge(das)


def _get_long_names(data, labeller=None):
    return [
        da.attrs.get(
            "long_name",
            key if labeller is None
            else labeller.var_name_to_str(key)
        )
        for key, da in data.items()
    ]


def plot_autocorr_evolution(data, n0=100, nn=20, labeller=None, **kwargs):
    ns = np.geomspace(kwargs.get("discard", 0) + n0, data.sizes["draw"], nn)
    ns = ns.astype(int)
    tau = autocorr_time_over_time(data, ns, **kwargs)

    _names = _get_long_names(data, labeller)
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
                 quantiles=(), quantile_kwargs=None,
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

        line_kwargs = _init_kwargs_dict(line_kwargs)
        lines = ax.plot(x, y, label=label, color=color, **line_kwargs, **kwargs)
        line_z = lines[0].get_zorder()
        _color = lines[0].get_color()

        fill_kwargs = _init_kwargs_dict(fill_kwargs)
        fill_kwargs.setdefault("zorder", line_z)
        fill_kwargs.setdefault("color", _color)
        fill_alpha = fill_kwargs.setdefault("alpha", kwargs.pop("alpha", 0.2))
        ax.fill_between(x, 0, y, **fill_kwargs, **kwargs)

        # quantile_kwargs.setdefault("color", "white")
        quantile_kwargs.setdefault("color", _color)
        quantile_kwargs.setdefault("alpha", (1 + fill_alpha) / 2)
        quantile_kwargs.setdefault("zorder", line_z)
        ymaxes = (
            np.interp(np.log(qvalues), np.log(x), y) if axes_scale == "log"
            else np.interp(qvalues, x, y)
        )
        for q, ymax in zip(qvalues, ymaxes):
            ax.plot([q, q], [0, ymax], **quantile_kwargs)


def _exponent(x):
    return np.floor(np.log10(np.abs(x))).astype(int)


def format_measurement(x, quantiles, err_prec=2, rescale_thresh=2, weights=None):
    from corner.core import quantile
    q_lo, q_mid, q_hi = quantile(x, quantiles, weights=weights)
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
                       title_stack_pad_frac=0.2, labeller=None):
    labels = [_get_long_names(data, labeller) for data in datasets]

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


def plot_corner(data, *, color=None, quantiles=(0.16, 0.5, 0.84), fill_contours=True,
                plot_contours=True, plot_density=False, plot_datapoints=False,
                hist_kind="kde", hist_kwargs=None, contour_kwargs=None,
                show_titles=True, title_kwargs=None, panel_dim=None, **kwargs):
    if color is None:
        import matplotlib as mpl
        color = mpl.rcParams["ytick.color"]

    if not set(data.sizes).issuperset({"chain", "draw"}):
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
        hist_kwargs=hist_kwargs, contour_kwargs=contour_kwargs,
        show_titles=show_titles, title_kwargs=title_kwargs,
        panel_dim=panel_dim, **kwargs,
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
                          labeller=None, show_titles=True,
                          quantiles=(0.16, 0.5, 0.84), title_kwargs=None,
                          title_loc="center", title_stack_pad_frac=0.2, **kwargs):
    if var_names is None:
        var_names = ordered_union([list(data.keys()) for data in datasets])
    if labels is None:
        labels = [None for _ in datasets]
    colors = _get_n_colors(colors, len(datasets))

    n = len(var_names)
    ncol = min(n, ncol)
    nrow = (n - 1) // ncol + 1
    h = w / aspect

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrow, ncol, figsize=(w*ncol, h*nrow), squeeze=False)

    axes_scale = _init_kwargs_dict(axes_scale)
    ranges = _init_kwargs_dict(ranges)
    limits = _init_kwargs_dict(limits)

    title_quantiles = kwargs.pop("title_quantiles", quantiles or (0.16, 0.5, 0.84))

    for data, label, color in zip(datasets, labels, colors):
        xlabels = dict(zip(data.keys(), _get_long_names(data, labeller)))
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

    fig.tight_layout()
    if show_titles:
        add_stacked_titles(
            axes.flat[:n], datasets, title_quantiles,
            var_names=var_names, colors=colors, title_loc=title_loc,
            title_kwargs=title_kwargs, labeller=labeller,
            title_stack_pad_frac=title_stack_pad_frac,
        )

    return fig, axes


def plot_1d_posterior(data, **kwargs):
    return compare_1d_posteriors([data], **kwargs)


def compare_2d_posteriors(datasets, cols=None, rows=None,
                          colors=None, hist_kind="kde", relative_hist=False,
                          show_titles=True, title_kwargs=None, title_loc="center",
                          title_stack_pad_frac=0.2, fig=None, **kwargs):
    default_contour_kwargs = _init_kwargs_dict(kwargs.get("contour_kwargs"))
    kwargs.setdefault("levels", 1 - np.exp(-1/2 * np.arange(1, 2.1, 1)**2))

    colors = _get_n_colors(colors, len(datasets))

    cols = cols or kwargs.pop("var_names", None)
    if cols is None:
        cols = ordered_union([list(data.keys()) for data in datasets])

    rows = rows or cols

    fig = None
    for i, (data, color) in enumerate(zip(datasets, colors)):
        kwargs["color"] = color
        contour_kwargs = default_contour_kwargs.copy()
        contour_kwargs.setdefault("colors", [color])
        kwargs["contour_kwargs"] = contour_kwargs

        if hist_kind == "kde":
            kwargs["hist_kwargs"] = {
                "line_kwargs": {"zorder": 2+i/1e3},
                "relative": relative_hist,
            }

        fig, axes = plot_corner(
            data, rows=rows, cols=cols, fig=fig, show_titles=False,
            hist_kind=hist_kind,
            force_range=i == 0,  # only force range the first time
            **kwargs,
        )

    title_quantiles = kwargs.get(
        "title_quantiles",
        kwargs.get("quantiles", (0.16, 0.5, 0.84))
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


def gelman_rubin(sample):
    nsteps, nwalkers, _ = sample.shape

    # variance of the per-walker means
    interchain_var = np.var(np.mean(sample, axis=0), axis=0, ddof=1)
    # mean of the per-walker variances
    intrachain_var = np.mean(np.var(sample, axis=0, ddof=1), axis=0)

    # FIXME: kombine computes the below
    # net_var = (
    #     intrachain_var * (nsteps - 1) / nsteps
    #     + interchain_var * (nwalkers + 1) / nwalkers
    # )
    net_var = intrachain_var * (nsteps - 1) / nsteps + interchain_var

    return np.sqrt(net_var / intrachain_var)


@dataclass
class EmceeResult:
    sampler: EnsembleSampler
    sample_parameters: list
    log_prob_names: list = field(default_factory=list)
    blob_names: list = field(default_factory=list)  # FIXME: rename to derived_names?
    var_name_map: dict = field(default_factory=dict)
    fixed_parameters: dict = field(default_factory=dict)
    _autocorr_discard: int = field(default=100, repr=False)

    var_names: list = field(default_factory=list, init=False)  # FIXME: rename?
    all_names: list = field(default_factory=list, init=False)  # FIXME: rename?
    nwalkers: int = field(init=False)
    ndim: int = field(init=False)
    nsteps: int = field(init=False)
    data: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        self.nwalkers = self.sampler.nwalkers
        self.ndim = self.sampler.ndim
        self.nsteps = self.sampler.iteration
        self.var_names = [par.name for par in self.sample_parameters]
        self.all_names = (
            tuple(self.var_names)
            + tuple(self.log_prob_names)
            + tuple(self.blob_names)
        )

        _sample_map = {par.name: par.latex for par in self.sample_parameters}
        self.var_name_map = _sample_map | self.var_name_map
        _blob_names = self.log_prob_names + self.blob_names

        # TODO: remove usage of az.from_emcee
        from excee.sampling import sample_pars_to_par_names
        # FIXME: the below
        try:
            slices = sample_pars_to_par_names(self.sample_parameters).values()
        except AttributeError:
            slices = None
        idata = az.from_emcee(
            self.sampler,
            var_names=self.var_names,
            slices=slices,
            blob_names=_blob_names if _blob_names else None,
        )
        self.idata = idata
        data = idata.posterior  # pylint: disable=E1101

        try:
            data = data.merge(idata.log_likelihood)  # pylint: disable=E1101
        except AttributeError:
            pass

        for key in self.var_names:
            data[key].attrs["kind"] = "sampled"
        for key in self.log_prob_names:
            data[key].attrs["kind"] = "log_prob"
        for key in self.blob_names:
            data[key].attrs["kind"] = "derived"

        data["log_prob"] = idata.sample_stats.lp  # pylint: disable=E1101
        data["log_prob"].attrs["kind"] = "log_prob"

        for key, val in data.items():
            val.attrs["long_name"] = self.var_name_map.get(key, key)

        self.data = data

    @cached_property
    def autocorr_time(self):
        tau = autocorr_time(
            self.data[self.var_names], discard=self._autocorr_discard)
        if not np.all(np.isfinite(tau)):
            from warnings import warn
            warn(f"nonfinite autocorrelation time: {tau}", stacklevel=2)
        return tau

    @classmethod
    def from_file(cls, fname):
        backend = HDFBackend(fname, read_only=True)
        backend.nwalkers, backend.ndim = backend.shape

        # FIXME: this
        with backend.open("r") as f:
            sample_parameters = read_pickle_from_h5(f["sample_parameters"])
            fixed_parameters = read_pickle_from_h5(f["fixed_parameters"])
            log_prob_names = list(f.attrs["log_prob_names"])
            blob_names = list(f.attrs["blob_names"])
            var_name_map = read_pickle_from_h5(f["var_name_map"])

        return cls(
            backend,
            sample_parameters,
            log_prob_names,
            blob_names,
            var_name_map,
            fixed_parameters=fixed_parameters,
        )

    def get_sample(self, discard_per_autocorr, thin_per_autocorr, *,
                   var_names=None, filter_std=None, tau=None,
                   split_vectors=False, **kwargs):
        if tau is None:
            tau = np.nanmax(self.autocorr_time)

        thin = round(thin_per_autocorr * tau)
        discard = round(discard_per_autocorr * tau)

        data = get_sample(self.data, discard, thin, **kwargs)

        if var_names is not None:
            data = data[var_names]

        if filter_std is not None:
            data = filter_outliers_dset(data, filter_std)

        if split_vectors:
            data = split_vector_vars(data)

        return data

    def get_random_sample(self, nsamples, rng=None):
        # FIXME: remove "sample" dimension but preserve coords?
        sample = self.get_sample(10, 1, flat=True)

        return get_random_sample(sample, "sample", nsamples, rng)

    @cached_property
    def best_sample(self):
        # N.B. *not* necessarily the best/optimal fit!
        idxmax = self.data.log_prob.argmax(...)
        return self.data[idxmax]

    def get_best_sample_array(self):
        return self.best_sample[self.var_names].to_array().values

    def get_bounds_array(self, clip=0.025):
        data = self.get_sample(10, 1, var_names=self.var_names, split_vectors=True)
        return data.quantile([clip, 1-clip]).to_array().values

    @cached_property
    def best_fit(self):
        try:
            ds = xr.load_dataset(
                self.sampler.filename, engine="h5netcdf", group="best_fit")

            return ds[list(self.data.keys())]
        except (OSError, AttributeError):
            return None

    def summary(self, discard_per_autocorr, thin_per_autocorr, var_names=None,
                rng=False, filter_std=None, hdi_prob=0.95, **kwargs):
        var_names = var_names or self.var_names
        if set(var_names) != set(self.var_names):
            tau = autocorr_time(self.data[var_names], discard=self._autocorr_discard)
        else:
            tau = self.autocorr_time

        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names,
            filter_std=filter_std, tau=np.nanmax(tau), rng=rng, split_vectors=True,
        )

        import arviz as az
        summary = az.summary(data, round_to="none", hdi_prob=hdi_prob, **kwargs)
        summary["tau"] = tau

        if self.best_fit is not None:
            best = self.best_fit[var_names]
            summary["best"] = split_vector_vars(best).to_array().values
        else:
            best = self.best_sample[var_names]
            summary["best*"] = split_vector_vars(best).to_array().values

        return summary

    def stats(self, discard_per_autocorr, thin_per_autocorr, **kwargs):
        df1 = self.summary(
            discard_per_autocorr, thin_per_autocorr,
            kind="stats", stat_focus="median", **kwargs)
        df2 = self.summary(
            discard_per_autocorr, thin_per_autocorr,
            kind="stats", stat_focus="mean", **kwargs)
        merged = df2.merge(df1)

        return merged.set_index(df1.index)

    @cached_property
    def arviz_labeller(self):
        # FIXME: use dset attrs instead, convert when needed
        from arviz.labels import MapLabeller
        return MapLabeller(var_name_map=self.var_name_map)

    def plot_autocorr_evolution(self, n0=100, nn=20, var_names=None,
                                discard=200, thin=1, **kwargs):
        var_names = var_names or self.var_names
        data = self.data[var_names]
        data = split_vector_vars(data)

        return plot_autocorr_evolution(
            data, n0=n0, nn=nn, labeller=self.arviz_labeller,
            discard=discard, thin=thin, **kwargs)

    def plot_corner(self, discard_per_autocorr=10, thin_per_autocorr=1,
                    *, var_names=None, filter_std=None, tau=None, rng=False,
                    **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names,
            filter_std=filter_std, tau=tau, rng=rng, split_vectors=True,
        )

        return plot_corner(data, labeller=self.arviz_labeller, **kwargs)

    def plot_trace_2d(self, var_names=None, draw=None, split_at_per_autocorr=10,
                      ratio=1/4, **kwargs):
        var_names = var_names or self.var_names
        data = self.data[var_names]
        if draw is not None:
            data = data.sel(draw=draw)
        data = split_vector_vars(data)

        if split_at_per_autocorr is not None:
            split_at = round(split_at_per_autocorr * np.nanmax(self.autocorr_time))
        else:
            split_at = None

        return plot_trace_2d(data, split_at=split_at, ratio=ratio, **kwargs)

    def plot_1d_posterior(self, discard_per_autocorr=10, thin_per_autocorr=1,
                          *, var_names=None, filter_std=None, tau=None, rng=False,
                          **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names or self.var_names,
            filter_std=filter_std, tau=tau, rng=rng, split_vectors=True,
        )

        return plot_1d_posterior(data, labeller=self.arviz_labeller, **kwargs)

    @cached_property
    def covariance_matrix(self):
        # FIXME: arguments?
        sample = self.get_sample(
            discard_per_autocorr=10, thin_per_autocorr=1,
            var_names=self.var_names, flat=True, split_vectors=True)
        return np.cov(sample.to_array().values)

    @cached_property
    def errors(self):
        return np.sqrt(np.diagonal(self.covariance_matrix))

    @cached_property
    def correlation_matrix(self):
        sig_sig = np.outer(self.errors, self.errors)
        return self.covariance_matrix / sig_sig

    def project_sample(self, sample, func, nthreads=None, **kwargs):
        from functools import partial
        func = partial(func, **(self.fixed_parameters | kwargs))

        from multiprocessing import Pool

        # FIXME: select only var_names?
        with Pool(nthreads) as pool:
            result = grouped_map(sample, "sample", func, mapper=pool.map)

        return result


def compare_results_1d(results, labels=None,
                       discard_per_autocorr=10, thin_per_autocorr=1,
                       posterior=True, log_probs=False, blobs=False,
                       filter_std=None, rng=False, var_names=None, **kwargs):
    labeller = az.labels.MapLabeller(
        union_dicts([res.var_name_map for res in results])
    )

    def _get_names(res):
        if var_names is not None:
            return ordered_intersection([var_names, res.all_names])

        names = []
        if posterior:
            names.extend(res.var_names)
        if log_probs:
            names.extend(res.log_prob_names)
        if blobs:
            names.extend(res.blob_names)

        return names

    datasets = [
        res.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=_get_names(res),
            filter_std=filter_std, rng=rng, split_vectors=True,
        )
        for res in results
    ]

    return compare_1d_posteriors(
        datasets,
        labels=labels,
        labeller=labeller,
        **kwargs,
    )


def compare_results_2d(results, discard_per_autocorr=10, thin_per_autocorr=1,
                       posterior=True, log_probs=False, blobs=False,
                       filter_std=None, rng=False, var_names=None,
                       **kwargs):
    labeller = az.labels.MapLabeller(
        union_dicts([res.var_name_map for res in results])
    )

    def _get_names(res):
        names = []
        if posterior:
            names.extend(res.var_names)
        if log_probs:
            names.extend(res.log_prob_names)
        if blobs:
            names.extend(res.blob_names)

        return names

    # FIXME: won't work with vector variables
    if var_names is None:
        var_names = ordered_union([_get_names(res) for res in results])

    datasets = [
        res.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=ordered_intersection([var_names, res.all_names]),
            filter_std=filter_std, rng=rng, split_vectors=True,
        )
        for res in results
    ]

    kwargs.setdefault("labeller", labeller)
    return compare_2d_posteriors(datasets, var_names=var_names, **kwargs)
