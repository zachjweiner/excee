__copyright__ = "Copyright (C) 2022 Zachary J Weiner"

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


from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Iterable, Callable
from typing import Any
from time import time
from functools import cached_property
import numpy as np
from scipy import stats
from emcee import EnsembleSampler
from emcee.autocorr import integrated_time

data_root = Path(__file__).parents[1] / "data"


@dataclass
class SampleParameter:
    name: str
    low: float
    high: float
    latex: str = None

    prior: Callable = field(init=False, repr=False)

    def __post_init__(self):
        self.prior = stats.uniform(self.low, self.high - self.low)


@dataclass
class GaussianSampleParameter:
    name: str
    mean: float
    std: float
    latex: str = None

    prior: Callable = field(init=False, repr=False)

    def __post_init__(self):
        self.prior = stats.norm(self.mean, self.std)


@dataclass
class FixedParameter:
    name: str
    value: float
    latex: str = None


@dataclass
class GaussianLikelihood:
    means: np.ndarray
    cov: np.ndarray
    mvn: Any = field(init=False)

    def __post_init__(self):
        self.mvn = stats.multivariate_normal(
            mean=self.means, cov=self.cov,
            allow_singular=True,
        )

    def __call__(self, pars):
        return self.mvn.logpdf(pars)


import emcee.moves
default_moves = [
    (emcee.moves.KDEMove(), 1),
    # (emcee.moves.DEMove(), 0.4),
    # (emcee.moves.DESnookerMove(), 0.2)
]


class LikelihoodSampler:
    def __init__(self,
                 sample_parameters: list,
                 log_prob: Callable,
                 nblobs: int = 0,
                 vectorize: bool = False,
                 kwargs=None):
        self.sample_parameters = sample_parameters
        self.ndim = len(self.sample_parameters)
        self.names = [par.name for par in self.sample_parameters]
        self.log_prob = log_prob
        self.nblobs = nblobs
        self.vectorize = vectorize
        self.kwargs = kwargs

    def log_prior(self, pars, *args, **kwargs):
        lnp = 0
        for i, par in enumerate(self.sample_parameters):
            if self.vectorize:
                lnp += par.prior.logpdf(pars[:, i])
            else:
                lnp += par.prior.logpdf(pars[par.name])

        return lnp

    def get_p0(self, nwalkers):
        p0 = np.empty((nwalkers, self.ndim))
        for i, par in enumerate(self.sample_parameters):
            p0[:, i] = par.prior.rvs(nwalkers)

        return p0

    def log_prob_wrap_vec(self, *args, **kwargs):
        log_prior = self.log_prior(*args, **kwargs)

        if self.nblobs > 0:
            log_prob, *blobs = self.log_prob(*args, **kwargs)
            return log_prior + log_prob, *blobs
        else:
            log_prob = self.log_prob(*args, **kwargs)
            return log_prior + log_prob

    def log_prob_wrap(self, *args, **kwargs):
        log_prior = self.log_prior(*args, **kwargs)
        if not np.isfinite(log_prior):
            if self.nblobs > 0:
                return (log_prior,) + (0,)*self.nblobs
            else:
                return log_prior

        if self.nblobs > 0:
            log_prob, *blobs = self.log_prob(*args, **kwargs)
            return log_prior + log_prob, *blobs
        else:
            log_prob = self.log_prob(*args, **kwargs)
            return log_prior + log_prob

    def __call__(self, nwalkers, nsteps, p0=None, progress="notebook",
                 moves: Iterable = None, pool=None,
                 **kwargs):
        sampler = EnsembleSampler(
            nwalkers, self.ndim,
            self.log_prob_wrap if not self.vectorize else self.log_prob_wrap_vec,
            moves=moves or default_moves,
            parameter_names=self.names if not self.vectorize else None,
            pool=pool,
            vectorize=self.vectorize,
            blobs_dtype=None if self.nblobs == 0 else float,
            kwargs=self.kwargs,
        )

        if p0 is None:
            p0 = self.get_p0(nwalkers)

        ss = time()
        sampler.run_mcmc(p0, nsteps, progress=progress, **kwargs)
        ee = time()

        return EmceeResult(sampler, None, None, ee-ss)


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


@dataclass
class EmceeResult:
    sampler: EnsembleSampler
    fiducial_parameters: np.ndarray
    fiducial_model: np.ndarray
    time: float

    nwalkers: int = field(init=False)
    ndim: int = field(init=False)
    nsteps: int = field(init=False)

    def __post_init__(self):
        self.nwalkers = self.sampler.nwalkers
        self.ndim = self.sampler.ndim
        self.nsteps = self.sampler.iteration

    def get_sample(self, discard_per_autocorr, thin_per_autocorr,
                   return_log_prob=False, blobs=False, flat=True):
        autocorr = np.max(self.autocorr_time())
        thin = int(thin_per_autocorr * autocorr)
        discard = int(discard_per_autocorr * autocorr)
        samples = self.sampler.get_chain(discard=discard, thin=thin, flat=flat)

        results = (samples,)
        if return_log_prob:
            log_likelihoods = self.sampler.get_log_prob(
                discard=discard, thin=thin, flat=True)
            results = results + (log_likelihoods,)
        if blobs:
            blobs = self.sampler.get_blobs(
                discard=discard, thin=thin, flat=True)
            if (names := blobs.dtype.names):
                blobs = blobs.view((float, len(names)))
            results = results + (blobs,)

        return np.hstack(results)

    def gelman_rubin(self, discard=0, thin=1, n=None):
        sample = self.sampler.get_chain(discard=discard, thin=thin, flat=False)[:n]
        return gelman_rubin(sample)

    def autocorr_time(self, discard=0, thin=1, n=None, quiet=True, **kwargs):
        x = self.sampler.get_chain(discard=discard, thin=thin)[:n]
        return thin * integrated_time(x, quiet=quiet, **kwargs)

    def autocorr_time_over_time(self, ns, tol=0, **kwargs):
        result = np.empty((self.sampler.ndim, ns.size))
        for i, n in enumerate(ns):
            result[:, i] = self.autocorr_time(n=n, tol=tol, **kwargs)
        return result

    def plot_autocorr_over_time(self, n0=100, nn=20):
        ns = np.geomspace(n0, self.sampler.iteration, nn).astype(int)
        tau = self.autocorr_time_over_time(ns)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(ns, tau.T, ".-", label=[fr"${int(t)}$" for t in tau[:, -1]])
        ax.legend(title=r"$\tau_f$", loc="center left", bbox_to_anchor=(1, 0.5))
        return fig, ax

    def gelman_rubin_over_time(self, ns, **kwargs):
        result = np.empty((self.sampler.ndim, ns.size))
        for i, n in enumerate(ns):
            result[:, i] = self.gelman_rubin(n=n)
        return result

    def plot_gelman_rubin_over_time(self, n0=100, nn=20):
        ns = np.geomspace(n0, self.sampler.iteration, nn).astype(int)
        R = self.gelman_rubin_over_time(ns)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.loglog(
            ns, np.abs(R.T**2 - 1),
            ".-"
        )
        # ax.legend(title=r"$\tau_f$", loc="center left", bbox_to_anchor=(1, 0.5))
        return fig, ax

    @cached_property
    def covariance_matrix(self):
        # FIXME: arguments?
        sample = self.get_sample(discard_per_autocorr=10, thin_per_autocorr=1)
        return np.cov(sample.T)

    @cached_property
    def errors(self):
        return np.sqrt(np.diagonal(self.covariance_matrix))

    @cached_property
    def correlation_matrix(self):
        sig_sig = np.outer(self.errors, self.errors)
        return self.covariance_matrix / sig_sig

    def plot_corner(self, transform=lambda x: x,
                    discard_per_autocorr=10, thin_per_autocorr=1, **kwargs):
        sample = self.get_sample(discard_per_autocorr, thin_per_autocorr)
        fig = corner(transform(sample), **kwargs)
        return fig, fig.axes


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


__all__ = [
    "EmceeResult",
]
