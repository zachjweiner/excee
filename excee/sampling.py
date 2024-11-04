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
from typing import Protocol
from abc import abstractmethod
from collections.abc import Sequence, Callable, Iterable
from typing import Any
import numpy as np
import xarray as xr
from scipy import stats, optimize
from emcee import EnsembleSampler
from emcee.backends import HDFBackend


class PriorInterface(Protocol):
    @abstractmethod
    def rvs(self, size=None, random_state=None) -> np.ndarray:
        pass

    @abstractmethod
    def mean(self) -> np.ndarray:
        pass

    @abstractmethod
    def std(self) -> np.ndarray:
        pass

    @abstractmethod
    def ppf(self, q: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


@dataclass
class ExponentialDistribution:
    """
    The distribution of a parameter whose exponentiatial is uniformly
    distributed beteween `low` and `high` (each positive and nonzero).
    """
    low: float
    high: float

    def __post_init__(self):
        if self.low < 0 or self.high < 0:
            raise ValueError("low and high must be positive and nonzero")

        self.dist = stats.truncexpon(
            b=np.log(self.high / self.low),
            loc=np.log(1 / self.high)
        )

    def rvs(self, size=None, random_state=None) -> np.ndarray:
        return - self.dist.rvs(size=size, random_state=random_state)

    def mean(self) -> np.ndarray:
        return - self.dist.mean()

    def std(self) -> np.ndarray:
        return self.dist.std()

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return - self.dist.ppf(q)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.logpdf(-x)


class SampleParameterInterface(Protocol):
    @property
    def name(self) -> str:
        pass

    @property
    def prior(self) -> PriorInterface:
        pass

    @property
    def size(self) -> int:
        pass


@dataclass(frozen=True)
class SampleParameter:
    name: str
    low: float
    high: float
    latex: str | None = None

    prior: PriorInterface = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        super().__setattr__("prior", self._make_prior())

    def _make_prior(self):
        return stats.uniform(self.low, self.high - self.low)

    @property
    def size(self):
        return self.prior.mean().size


class LogUniformSampleParameter(SampleParameter):
    def _make_prior(self):
        return stats.loguniform(self.low, self.high)


class ExpUniformSampleParameter(SampleParameter):
    def _make_prior(self):
        return ExponentialDistribution(self.low, self.high)


@dataclass(frozen=True)
class GaussianSampleParameter:
    name: str
    mean: float
    std: float
    latex: str | None = None
    low: float = -np.inf
    high: float = np.inf

    prior: PriorInterface = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        super().__setattr__("prior", self._make_prior())

    def _make_prior(self):
        if np.any(np.isfinite(self.low)) or np.any(np.isfinite(self.low)):
            a = (self.low - self.mean) / self.std
            b = (self.high - self.mean) / self.std
            return stats.truncnorm(loc=self.mean, scale=self.std, a=a, b=b)
        else:
            return stats.norm(self.mean, self.std)

    @property
    def size(self):
        return self.prior.mean().size


def sample_parameter_rvs(parameters, nsamples):
    return np.hstack([
        par.prior.rvs((nsamples, par.size))
        for par in parameters
    ])


def parameters_to_normal(parameters, reduce_uniform_std_by=1, reduce_norm_std_by=1):
    mean = np.hstack([par.prior.mean() for par in parameters])
    std = np.hstack([
        par.prior.std() / (
            reduce_uniform_std_by if isinstance(par, SampleParameter)
            else reduce_norm_std_by if isinstance(par, GaussianSampleParameter)
            else 1
        )
        for par in parameters
    ])
    low = np.hstack([par.low for par in parameters])
    high = np.hstack([par.high for par in parameters])

    return GaussianSampleParameter("all", mean, std, low=low, high=high)


def delta_chi2_per_dof(dist, sample):
    return - 2 * np.mean(dist.logpdf(sample) - dist.logpdf(dist.mean()), axis=-1)


def sample_parameters_within(parameters, chi2_per_par, nsamples, **kwargs):
    mvn = parameters_to_normal(parameters, **kwargs)

    sample = mvn.prior.rvs((nsamples, mvn.size))
    while True:
        mask = delta_chi2_per_dof(mvn.prior, sample) > chi2_per_par
        if not np.any(mask):
            return sample
        sample[mask] = mvn.prior.rvs((sum(mask), mvn.size))


@dataclass(frozen=True)
class FixedParameter:
    name: str
    value: float
    latex: str | None = None


@dataclass
class GaussianLikelihood:
    means: np.ndarray
    cov: np.ndarray
    mvn: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.mvn = stats.multivariate_normal(
            mean=self.means, cov=self.cov,
            allow_singular=True,
        )

    def __call__(self, pars):
        return self.mvn.logpdf(pars)


def _compare_arrays(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False

    return np.all(a == b)


import emcee.moves
default_moves = [
    (emcee.moves.KDEMove(), 1),
    # (emcee.moves.DEMove(), 0.4),
    # (emcee.moves.DESnookerMove(), 0.2)
]


def sample_pars_to_par_names(
        sample_parameters: Sequence[SampleParameterInterface]):
    parameter_names = {}
    i0 = 0
    for par in sample_parameters:
        i1 = i0 + par.size
        parameter_names[par.name] = slice(i0, i1) if par.size > 1 else i0
        i0 = i1

    return parameter_names


@dataclass
class LikelihoodSampler:
    sample_parameters: list[SampleParameterInterface]
    log_prob: Callable
    vectorize: bool = False
    kwargs: dict = field(default_factory=dict)
    var_name_map: dict = field(default_factory=dict)
    derived_priors: Iterable[Callable] = field(default_factory=dict)

    def __post_init__(self):
        self.ndim = sum(par.size for par in self.sample_parameters)
        self.names = [par.name for par in self.sample_parameters]

        p0 = {par.name: par.prior.mean() for par in self.sample_parameters}
        test = self.log_prob(p0, **self.kwargs)
        if isinstance(test, tuple):
            log_probs, blobs = test
            self.nblobs = len(log_probs) + len(blobs)
            self.log_prob_names = list(log_probs.keys())
            self.blob_names = list(blobs.keys())
        else:
            self.nblobs = 0
            self.log_prob_names = ()
            self.blob_names = ()

    def log_prior(self, pars, *args, **kwargs):
        lnp = 0
        for par in self.sample_parameters:
            lnp_par = par.prior.logpdf(pars[par.name])
            if par.size > 1:
                lnp_par = lnp_par.sum(axis=-1)
            lnp += lnp_par

        for fn in self.derived_priors:
            lnp += fn(pars, *args, **kwargs)

        return lnp

    def get_p0(self, nwalkers):
        return sample_parameter_rvs(self.sample_parameters, nwalkers)

    def log_prob_wrap(self, *args, **kwargs):
        log_prior = self.log_prior(*args, **kwargs)
        if not self.vectorize and not np.isfinite(log_prior):
            if self.nblobs > 0:
                return (log_prior,) + (0,)*self.nblobs
            else:
                return log_prior

        if self.nblobs > 0:
            log_prob_dict, blobs_dict = self.log_prob(*args, **kwargs)
            if self.vectorize:
                log_probs = np.array(list(log_prob_dict.values()))
                blobs = np.array(list(blobs_dict.values()))
                log_prob_plus_prior = log_prior + np.sum(log_probs, axis=0)
                return np.vstack([log_prob_plus_prior, log_probs, blobs]).T
            else:
                log_probs = tuple(log_prob_dict.values())
                blobs = tuple(blobs_dict.values())
                log_prob = sum(log_probs)
                return log_prior + log_prob, *log_probs, *blobs
        else:
            log_prob = self.log_prob(*args, **kwargs)
            return log_prior + log_prob

    def log_prob_wrap_optimize(self, x, **kwargs):
        pars = dict(zip(self.names, x))
        log_prior = self.log_prior(pars, **kwargs)
        res = self.log_prob_wrap(pars, **kwargs)
        log_post = - res[0] if isinstance(res, tuple) else - res
        return log_post - log_prior

    def __call__(self, nwalkers, nsteps, p0=None, progress="notebook",
                 moves: Sequence | None = None, pool=None, backend=None,
                 **kwargs):
        sampler = EnsembleSampler(
            nwalkers, self.ndim,
            self.log_prob_wrap,
            moves=moves or default_moves,
            parameter_names=sample_pars_to_par_names(self.sample_parameters),
            pool=pool,
            vectorize=self.vectorize,
            backend=backend,
            blobs_dtype=None if self.nblobs == 0 else float,
            kwargs=self.kwargs,
        )

        if isinstance(backend, HDFBackend):
            with backend.open("a") as file:
                if backend.iteration == 0:
                    from excee.util import write_pickle_to_h5

                    write_pickle_to_h5(
                        file, self.sample_parameters, "sample_parameters")
                    write_pickle_to_h5(file, self.kwargs, "fixed_parameters")
                    write_pickle_to_h5(file, self.var_name_map, "var_name_map")
                    file.attrs["log_prob_names"] = self.log_prob_names
                    file.attrs["blob_names"] = self.blob_names
                else:
                    from excee.util import read_pickle_from_h5

                    _sample = read_pickle_from_h5(file["sample_parameters"])
                    _fixed = read_pickle_from_h5(file["fixed_parameters"])
                    _lp = file.attrs["log_prob_names"]
                    _blob = file.attrs["blob_names"]
                    consistent = (
                        _compare_arrays(self.sample_parameters, _sample)
                        and _compare_arrays(self.kwargs, _fixed)
                        and _compare_arrays(self.log_prob_names, _lp)
                        and _compare_arrays(self.blob_names, _blob)
                    )

                    if not consistent:
                        raise RuntimeError(
                            "Existing backend file and current sampler "
                            "are not consistent"
                        )

        if p0 is None and (backend is None or backend.iteration == 0):
            p0 = self.get_p0(nwalkers)
        if isinstance(p0, dict):
            p0 = {key: p0[key] for key in self.names}

        sampler.run_mcmc(p0, nsteps, progress=progress, **kwargs)

        from excee import SamplingResult
        result = SamplingResult.from_emcee(
            sampler,
            self.sample_parameters,
            self.kwargs,
            self.log_prob_names,
            self.blob_names,
            self.var_name_map
        )
        return result

    def _optimization_config(self, x0, bounds):
        from functools import partial
        func = partial(self.log_prob_wrap_optimize, **self.kwargs)

        if x0 is None:
            x0 = np.array([par.prior.mean() for par in self.sample_parameters])

        if bounds is None:
            delta = 1e-3
            bounds = np.array([
                [par.prior.ppf(x) for x in (delta, 1-delta)]
                for par in self.sample_parameters
            ])

        return func, x0, bounds

    def save_optimize_result(self, result, backend, group):
        if isinstance(backend, HDFBackend):
            pars = dict(zip(self.names, result.x))
            if self.nblobs > 0:
                log_prob_dict, blobs_dict = self.log_prob(pars.copy(), **self.kwargs)
                prior = self.log_prior(pars.copy(), **self.kwargs)
                log_prob_dict["log_prob"] = sum(log_prob_dict.values()) + prior
                ds = xr.Dataset(pars | log_prob_dict | blobs_dict)
            else:
                ds = xr.Dataset(pars)

            for key, val in result.items():
                if key != "x":
                    try:
                        ds.attrs[key] = val
                    except Exception:
                        pass

            with backend.open("a") as file:
                if group in file:
                    del file[group]

            ds.to_netcdf(
                backend.filename, engine="h5netcdf", group=group, mode="a",
                invalid_netcdf=True
            )

    def minimize(self, x0=None, bounds=None, backend=None, group="best_fit",
                 **kwargs):
        func, x0, bounds = self._optimization_config(x0, bounds)
        result = optimize.minimize(func, x0, bounds=bounds, **kwargs)
        self.save_optimize_result(result, backend, group)

        return result

    def nelder_mead(self, x0=None, bounds=None, backend=None, group="best_fit",
                    xatol=1e-3, fatol=1e-1, **kwargs):
        options = {"xatol": xatol, "fatol": fatol} | kwargs
        return self.minimize(
            x0=x0, bounds=bounds, backend=backend, group=group,
            method="Nelder-Mead", options=options,
        )

    def powell(self, x0=None, bounds=None, backend=None, group="best_fit",
               xtol=1e-4, ftol=1e-4, **kwargs):
        options = {"xtol": xtol, "ftol": ftol} | kwargs
        return self.minimize(
            x0=x0, bounds=bounds, backend=backend, group=group,
            method="Powell", options=options,
        )

    def differential_evolution(self, x0=None, bounds=None, backend=None,
                               group="best_fit", workers=-1, tol=1e-3, **kwargs):
        func, x0, bounds = self._optimization_config(x0, bounds)
        result = optimize.differential_evolution(
            func, bounds=bounds, x0=x0, tol=tol, workers=workers, **kwargs
        )
        self.save_optimize_result(result, backend, group)

        return result

    def direct(self, x0=None, bounds=None, backend=None, group="best_fit",
               f_min_rtol=1e-3, len_tol=1e-3, **kwargs):
        func, x0, bounds = self._optimization_config(x0, bounds)
        result = optimize.direct(
            func, bounds=list(bounds), f_min_rtol=f_min_rtol, len_tol=len_tol,
            **kwargs
        )
        self.save_optimize_result(result, backend, group)

        return result

    def dual_annealing(self, x0=None, bounds=None, backend=None, group=None,
                       **kwargs):
        func, x0, bounds = self._optimization_config(x0, bounds)
        result = optimize.dual_annealing(
            func, x0=x0, bounds=bounds, **kwargs
        )
        if group is not None:
            self.save_optimize_result(result, backend, group)

        return result

    def shgo(self, x0=None, bounds=None, backend=None, group="best_fit", **kwargs):
        func, x0, bounds = self._optimization_config(x0, bounds)
        result = optimize.shgo(
            func, bounds=bounds, **kwargs
        )
        self.save_optimize_result(result, backend, group)

        return result

    def basinhopping(self, x0=None, bounds=None, backend=None, group="best_fit",
                     **kwargs):
        func, x0, bounds = self._optimization_config(x0, bounds)
        result = optimize.basinhopping(
            func, x0=x0, **kwargs
        )
        self.save_optimize_result(result, backend, group)

        return result
