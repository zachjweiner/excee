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
from collections.abc import Iterable, Callable
from typing import Any
import numpy as np
from scipy import stats
from emcee import EnsembleSampler
from emcee.backends import HDFBackend


@dataclass
class SampleParameter:
    name: str
    low: float
    high: float
    latex: str = None

    prior: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.prior = stats.uniform(self.low, self.high - self.low)


@dataclass
class GaussianSampleParameter:
    name: str
    mean: float
    std: float
    latex: str = None

    prior: Callable = field(init=False, repr=False, compare=False)

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


class LikelihoodSampler:
    def __init__(self,
                 sample_parameters: list,
                 log_prob: Callable,
                 vectorize: bool = False,
                 kwargs: dict = None,
                 var_name_map: dict = None):
        self.sample_parameters = sample_parameters
        self.ndim = len(self.sample_parameters)
        self.names = [par.name for par in self.sample_parameters]
        self.log_prob = log_prob
        self.vectorize = vectorize
        self.kwargs = kwargs or {}
        self.var_name_map = var_name_map or {}

        p0 = {par.name: par.prior.rvs(size=1)[0] for par in self.sample_parameters}
        test = log_prob(p0, **self.kwargs)
        if isinstance(test, tuple):
            log_probs, blobs = test
            self.nblobs = len(log_probs) + len(blobs)
            self.log_prob_names = tuple(log_probs.keys())
            self.blob_names = tuple(blobs.keys())
        else:
            self.nblobs = 0
            self.log_prob_names = ()
            self.blob_names = ()

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
            log_prob_dict, blobs_dict = self.log_prob(*args, **kwargs)
            log_probs = tuple(log_prob_dict.values())
            blobs = tuple(blobs_dict.values())
            log_prob = sum(log_probs)
            return log_prior + log_prob, *log_probs, *blobs
        else:
            log_prob = self.log_prob(*args, **kwargs)
            return log_prior + log_prob

    def __call__(self, nwalkers, nsteps, p0=None, progress="notebook",
                 moves: Iterable = None, pool=None, backend=None,
                 **kwargs):
        sampler = EnsembleSampler(
            nwalkers, self.ndim,
            self.log_prob_wrap if not self.vectorize else self.log_prob_wrap_vec,
            moves=moves or default_moves,
            parameter_names=self.names if not self.vectorize else None,
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

        sampler.run_mcmc(p0, nsteps, progress=progress, **kwargs)

        from excee import EmceeResult
        result = EmceeResult(
            sampler,
            self.sample_parameters,
            self.log_prob_names,
            self.blob_names,
            self.var_name_map
        )

        return result
