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


def plot_autocorr_evolution(data, n0=100, nn=20, labeller=None, **kwargs):
    ns = np.geomspace(n0, data.dims["draw"], nn).astype(int)
    tau = autocorr_time_over_time(data, ns, **kwargs)

    labels = [
        fr"{name}: ${int(t)}$" if labeller is None
        else fr"{labeller.var_name_map[name]}: ${int(t)}$"
        for t, name in zip(tau[:, -1], data)
    ]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.loglog(ns, tau.T, ".-", label=labels)
    ax.legend(title=r"$\tau_f$", loc="center left", bbox_to_anchor=(1, 0.5))
    return fig, ax


def plot_trace_2d(data, width=8, height=3, **kwargs):
    import matplotlib.pyplot as plt

    n = len(data)
    fig, axes = plt.subplots(
        n, 1, figsize=(width, n*height), sharex=True, squeeze=False)

    for key, ax in zip(data, axes.flat):
        data[key].plot(ax=ax, **kwargs)

    for ax in axes[:-1, 0]:
        ax.set_xlabel(None)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    return fig, axes


def get_sample(data, discard, thin, flat=True, rng=None):
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

        rng = np.random.default_rng(None if rng is True else rng)
        slc = rng.choice(len(data[axis]), size=num_samples, replace=False)
        data = data.isel({axis: slc})

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


def _init_kwargs_dict(kwargs):
    return {} if kwargs is None else kwargs.copy()


def corner(data, quantiles=(0.16, 0.5, 0.84), fill_contours=True, plot_contours=True,
           plot_density=False, plot_datapoints=False, bins=20,
           hist_kwargs=None, contour_kwargs=None, show_titles=True,
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

    var_names: list = field(default_factory=list, init=False)  # FIXME: rename?
    nwalkers: int = field(init=False)
    ndim: int = field(init=False)
    nsteps: int = field(init=False)
    data: xr.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        self.nwalkers = self.sampler.nwalkers
        self.ndim = self.sampler.ndim
        self.nsteps = self.sampler.iteration
        self.var_names = [par.name for par in self.sample_parameters]

        _sample_map = {par.name: par.latex for par in self.sample_parameters}
        self.var_name_map = _sample_map | self.var_name_map

        # TODO: remove usage of az.from_emcee
        idata = az.from_emcee(
            self.sampler,
            var_names=self.var_names,
            blob_names=self.log_prob_names+self.blob_names,
        )
        self.idata = idata
        data = idata.posterior.merge(idata.log_likelihood)  # pylint: disable=E1101
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

    @classmethod
    def from_file(cls, fname):
        backend = HDFBackend(fname, read_only=True)
        backend.nwalkers, backend.ndim = backend.shape

        from excee.util import read_pickle_from_h5

        # FIXME: this
        with backend.open("r") as f:
            sample_parameters = read_pickle_from_h5(f["sample_parameters"])
            # res.fixed_parameters = read_pickle_from_h5(f["fixed_parameters"])
            log_prob_names = list(f.attrs["log_prob_names"])
            blob_names = list(f.attrs["blob_names"])
            var_name_map = read_pickle_from_h5(f["var_name_map"])

        return cls(
            backend,
            sample_parameters,
            log_prob_names,
            blob_names,
            var_name_map
        )

    def get_sample(self, discard_per_autocorr, thin_per_autocorr,
                   tau=None, rng=False, **kwargs):
        if tau is None:
            tau = np.max(autocorr_time(self.data[self.var_names]))

        thin = int(thin_per_autocorr * tau)
        discard = int(discard_per_autocorr * tau)

        return get_sample(self.data, discard, thin, rng=rng, **kwargs)

    def summary(self, discard_per_autocorr, thin_per_autocorr, var_names=None,
                rng=None, **kwargs):
        var_names = var_names or self.var_names
        tau = autocorr_time(self.data[var_names])

        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr, tau=np.max(tau), flat=False,
            rng=rng)
        data = data[var_names]

        import arviz as az
        summary = az.summary(data, round_to="none", **kwargs)
        summary["tau"] = tau

        return summary

    @cached_property
    def arviz_labeller(self):
        from arviz.labels import MapLabeller
        return MapLabeller(var_name_map=self.var_name_map)

    def plot_autocorr_evolution(self, n0=100, nn=20, var_names=None, **kwargs):
        var_names = var_names or self.var_names
        data = self.data[var_names]

        return plot_autocorr_evolution(
            data, n0=n0, nn=nn, labeller=self.arviz_labeller, **kwargs)

    def plot_corner(self, discard_per_autocorr=10, thin_per_autocorr=1,
                    *, tau=None, filter_std=None, rng=None,
                    **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr, tau=tau, rng=rng)

        if filter_std is not None:
            data = filter_outliers_dset(data, filter_std)

        return corner(data, labeller=self.arviz_labeller, **kwargs)

    def plot_trace_2d(self, var_names=None, **kwargs):
        var_names = var_names or self.var_names
        return plot_trace_2d(self.data[var_names], **kwargs)

    @cached_property
    def covariance_matrix(self):
        # FIXME: arguments?
        sample = self.get_sample(discard_per_autocorr=10, thin_per_autocorr=1)
        return np.cov(sample.to_array().values)

    @cached_property
    def errors(self):
        return np.sqrt(np.diagonal(self.covariance_matrix))

    @cached_property
    def correlation_matrix(self):
        sig_sig = np.outer(self.errors, self.errors)
        return self.covariance_matrix / sig_sig
