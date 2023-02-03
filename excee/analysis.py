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
import numpy as np
import xarray as xr
import arviz as az
from emcee import EnsembleSampler
from emcee.autocorr import integrated_time
from emcee.backends import HDFBackend


def autocorr_time(data, discard=0, thin=1, n=None, quiet=True, **kwargs):
    x = data.sel(draw=slice(discard, n, thin))
    x = x.to_array().values.T
    return thin * integrated_time(x, quiet=quiet, **kwargs)


def autocorr_time_over_time(data, ns, tol=0, **kwargs):
    result = np.empty((len(data), len(ns)))
    for i, n in enumerate(ns):
        result[:, i] = autocorr_time(
            data, discard=0, thin=1, n=n, tol=tol, **kwargs)

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

        num_samples = data.dims[axis] // thin
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


def filter_outliers_dset(dset, nstd, thresh=0.99, max_iter=10, min_iter=2):
    if isinstance(nstd, (float, int)):
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

    n = dset.dims["sample"]
    dset = dset.drop_vars(["chain", "sample", "draw"])
    dset = dset.rename_dims({"sample": "draw"})
    dset = dset.assign_coords(draw=np.arange(n))
    dset = dset.expand_dims({"chain": [1]}, axis=0)

    return dset


def plot_autocorr_evolution(data, n0=100, nn=20, labeller=None, **kwargs):
    ns = np.geomspace(n0, data.dims["draw"], nn).astype(int)
    tau = autocorr_time_over_time(data, ns, **kwargs)

    labels = [
        fr"{name}: ${round(t)}$" if labeller is None
        else fr"{labeller.var_name_map[name]}: ${round(t)}$"
        for t, name in zip(tau[:, -1], data)
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
        ratio = 1 if split_at is None else split_at / data.dims["draw"]

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


def plot_corner(data, quantiles=(0.16, 0.5, 0.84), fill_contours=True,
                plot_contours=True, plot_density=False, plot_datapoints=False,
                bins=20, hist_kwargs=None, contour_kwargs=None, show_titles=True,
                title_kwargs=None, **kwargs):
    hist_kwargs = _init_kwargs_dict(hist_kwargs)
    hist_kwargs.setdefault("histtype", "stepfilled")
    hist_kwargs.setdefault("alpha", 0.2)

    contour_kwargs = _init_kwargs_dict(contour_kwargs)
    contour_kwargs.setdefault("linewidths", 0.4)

    title_kwargs = _init_kwargs_dict(title_kwargs)
    title_kwargs.setdefault("fontsize", 14)

    import corner
    return corner.corner(
        data, quantiles=quantiles, bins=bins,
        fill_contours=fill_contours, plot_contours=plot_contours,
        plot_density=plot_density, plot_datapoints=plot_datapoints,
        hist_kwargs=hist_kwargs, contour_kwargs=contour_kwargs,
        show_titles=show_titles, title_kwargs=title_kwargs,
        **kwargs
    )


def compare_1d_posteriors(datasets, labels=None, var_names=None, ncol=4, w=4,
                          kind="hist", fill_kwargs=None, labeller=None, **kwargs):
    if var_names is None:
        from excee.util import ordered_union
        var_names = ordered_union([list(data.keys()) for data in datasets])
    if labels is None:
        labels = [None for _ in datasets]

    n = len(var_names)
    nrow = (n - 1) // ncol + 1

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrow, ncol, figsize=(w*ncol, w*nrow))
    prop_cycler = plt.rcParams["axes.prop_cycle"]

    for data, label, props in zip(datasets, labels, prop_cycler):
        for key, ax in zip(var_names, axes.flat):
            if key not in data:
                continue

            if kind == "hist":
                ax.hist(
                    data[key].values.ravel(),
                    label=label, **kwargs, **props,
                )
            elif kind == "kde":
                x, y = az.kde(data[key].values.ravel())
                ax.plot(
                    x, y,
                    label=label, **kwargs, **props
                )
                ax.fill_between(
                    x, 0, y,
                    **fill_kwargs, **props,
                )

            if labeller is not None:
                ax.set_xlabel(labeller.var_name_to_str(key))

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax in axes.flat:
        ax.get_yaxis().set_visible(False)
        ax.tick_params(which="both", top=False, left=False, right=False)
        ax.spines[["left", "right", "top"]].set_visible(False)

    fig.tight_layout()

    return fig, axes


def plot_1d_posterior(data, **kwargs):
    return compare_1d_posteriors([data], **kwargs)


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
        self.all_names = self.var_names + self.log_prob_names + self.blob_names

        _sample_map = {par.name: par.latex for par in self.sample_parameters}
        self.var_name_map = _sample_map | self.var_name_map
        _blob_names = self.log_prob_names + self.blob_names

        # TODO: remove usage of az.from_emcee
        idata = az.from_emcee(
            self.sampler,
            var_names=self.var_names,
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
        return autocorr_time(self.data[self.var_names])

    @classmethod
    def from_file(cls, fname):
        backend = HDFBackend(fname, read_only=True)
        backend.nwalkers, backend.ndim = backend.shape

        from excee.util import read_pickle_from_h5

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
                   var_names=None, filter_std=None, tau=None, **kwargs):
        if tau is None:
            tau = np.max(self.autocorr_time)

        thin = round(thin_per_autocorr * tau)
        discard = round(discard_per_autocorr * tau)

        data = get_sample(self.data, discard, thin, **kwargs)

        if var_names is not None:
            data = data[var_names]

        if filter_std is not None:
            data = filter_outliers_dset(data, filter_std)

        return data

    def get_random_sample(self, nsamples, rng=None):
        # FIXME: remove "sample" dimension but preserve coords?
        sample = self.get_sample(10, 1, flat=True)

        return get_random_sample(sample, "sample", nsamples, rng)

    def summary(self, discard_per_autocorr, thin_per_autocorr, var_names=None,
                rng=False, filter_std=None, **kwargs):
        tau = self.autocorr_time
        var_names = var_names or self.var_names

        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names,
            filter_std=filter_std, tau=np.max(tau), rng=rng,
        )

        import arviz as az
        summary = az.summary(data, round_to="none", **kwargs)
        summary["tau"] = tau

        return summary

    @cached_property
    def arviz_labeller(self):
        # FIXME: use dset attrs instead, convert when needed
        from arviz.labels import MapLabeller
        return MapLabeller(var_name_map=self.var_name_map)

    def plot_autocorr_evolution(self, n0=100, nn=20, var_names=None, **kwargs):
        var_names = var_names or self.var_names
        data = self.data[var_names]

        return plot_autocorr_evolution(
            data, n0=n0, nn=nn, labeller=self.arviz_labeller, **kwargs)

    def plot_corner(self, discard_per_autocorr=10, thin_per_autocorr=1,
                    *, var_names=None, filter_std=None, tau=None, rng=False,
                    **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names,
            filter_std=filter_std, tau=tau, rng=rng,
        )

        return plot_corner(data, labeller=self.arviz_labeller, **kwargs)

    def plot_trace_2d(self, var_names=None, draw=None, split_at_per_autocorr=10,
                      ratio=1/4, **kwargs):
        var_names = var_names or self.var_names
        data = self.data[var_names]
        if draw is not None:
            data = data.sel(draw=draw)

        if split_at_per_autocorr is not None:
            split_at = round(split_at_per_autocorr * np.max(self.autocorr_time))
        return plot_trace_2d(data, split_at=split_at, ratio=ratio, **kwargs)

    def plot_1d_posterior(self, discard_per_autocorr=10, thin_per_autocorr=1,
                          *, var_names=None, filter_std=None, tau=None, rng=False,
                          **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names or self.var_names,
            filter_std=filter_std, tau=tau, rng=rng,
        )

        return plot_1d_posterior(data, labeller=self.arviz_labeller, **kwargs)

    @cached_property
    def covariance_matrix(self):
        # FIXME: arguments?
        sample = self.get_sample(
            discard_per_autocorr=10, thin_per_autocorr=1,
            var_names=self.var_names, flat=True)
        return np.cov(sample.to_array().values)

    @cached_property
    def errors(self):
        return np.sqrt(np.diagonal(self.covariance_matrix))

    @cached_property
    def correlation_matrix(self):
        sig_sig = np.outer(self.errors, self.errors)
        return self.covariance_matrix / sig_sig


def compare_results_1d(results, labels=None,
                       discard_per_autocorr=10, thin_per_autocorr=1,
                       posterior=True, log_probs=False, blobs=False,
                       filter_std=None, rng=False, **kwargs):

    from excee.util import union_dicts
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

    datasets = [
        res.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=_get_names(res),
            filter_std=filter_std, rng=rng,
        )
        for res in results
    ]

    return compare_1d_posteriors(
        datasets,
        labels=labels,
        labeller=labeller,
        **kwargs,
    )
