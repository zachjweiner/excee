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


from dataclasses import dataclass, field
from functools import cached_property
import re
from pathlib import Path
import numpy as np
import xarray as xr
import arviz as az
from excee.util import (
    ordered_intersection, ordered_union, read_pickle_from_h5,
    grouped_map, label_from_attrs
)
from excee.plot import (
    plot_autocorr_evolution, plot_trace_2d, plot_corner,
    compare_1d_posteriors, compare_2d_posteriors, plot_1d_posterior,
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

    from emcee.autocorr import integrated_time
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
    dset = dset.drop_vars(["chain", "sample", "draw"], errors="ignore")
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


def split_vector_vars(data, keep_dims=("chain", "draw", "sample")):
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
                if long_name := label_from_attrs(da):
                    prefix = long_name.replace("$", "")
                var.attrs["long_name"] = f"${prefix}_{{{idx}}}$"
        else:
            dset = da.copy()

        return dset

    das = [split_one(da) for da in data.values()]

    return xr.merge(das)


@dataclass
class SamplingResult:
    data: xr.Dataset
    best_fit: xr.Dataset = None
    fixed_parameters: dict = field(default_factory=dict)
    _autocorr_discard: int = field(default=100, repr=False)

    @classmethod
    def from_emcee_hdf(cls, backend):
        if isinstance(backend, str | Path):
            from emcee.backends import HDFBackend
            backend = HDFBackend(backend, read_only=True)

        # FIXME: this
        with backend.open("r") as f:
            sample_parameters = read_pickle_from_h5(f["sample_parameters"])
            fixed_parameters = read_pickle_from_h5(f["fixed_parameters"])
            log_prob_names = tuple(f.attrs["log_prob_names"])
            blob_names = tuple(f.attrs["blob_names"])
            var_name_map = read_pickle_from_h5(f["var_name_map"])

        try:
            best_fit = xr.load_dataset(
                backend.filename, engine="h5netcdf", group="best_fit")
        except (OSError, AttributeError):
            best_fit = None

        return cls.from_emcee(
            backend, sample_parameters, fixed_parameters, log_prob_names,
            blob_names, var_name_map, best_fit,
        )

    @classmethod
    def from_emcee(cls, backend, sample_parameters, fixed_parameters, log_prob_names,
                   blob_names, var_name_map, best_fit=None):
        var_names = [par.name for par in sample_parameters]

        _sample_map = {par.name: par.latex for par in sample_parameters}
        var_name_map = _sample_map | var_name_map
        _blob_names = log_prob_names + blob_names

        # TODO: remove usage of az.from_emcee
        from excee.sampling import sample_pars_to_par_names
        # FIXME: the below
        try:
            slices = sample_pars_to_par_names(sample_parameters).values()
        except AttributeError:
            slices = np.arange(len(sample_parameters))

        chain = backend.get_chain().transpose(2, 1, 0)
        coords = {
            "chain": np.arange(chain.shape[1]),
            "draw": np.arange(chain.shape[2]),
        }
        chain = {
            var_name: (("chain", "draw"), chain[idx])
            for idx, var_name in zip(slices, var_names)
        }

        if (blobs := backend.get_blobs()) is not None:
            blobs = blobs.transpose(2, 1, 0)
            blobs = {
                var_name: (("chain", "draw"), blobs[idx])
                for idx, var_name in enumerate(_blob_names)
            }
        else:
            blobs = {}

        blobs["log_prob"] = ("chain", "draw"), backend.get_log_prob().T
        data = xr.Dataset(chain | blobs, coords=coords)

        for key in var_names:
            data[key].attrs["kind"] = "sampled"
        for key in (*log_prob_names, "log_prob"):
            data[key].attrs["kind"] = "log_prob"
        for key in blob_names:
            data[key].attrs["kind"] = "derived"

        for key, val in data.items():
            val.attrs["long_name"] = var_name_map.get(key, key)

        return cls(data, best_fit=best_fit, fixed_parameters=fixed_parameters)

    @classmethod
    def from_cobaya(cls, path, run_key, repeat=True, truncate=True):
        from excee.cobaya_interop import get_cobaya_data
        data, fixed_parameters = get_cobaya_data(
            path, run_key, repeat=repeat, truncate=truncate
        )
        return cls(data, fixed_parameters=fixed_parameters)

    @classmethod
    def from_montepython(cls, path, repeat=True, truncate=True):
        from excee.mp_interop import get_montepython_data
        data = get_montepython_data(
            path, repeat=repeat, truncate=truncate
        )
        return cls(data)

    @cached_property
    def autocorr_time(self):
        ds = self.data  # .filter_by_attrs(kind="sampled")
        tau = autocorr_time(ds, discard=self._autocorr_discard)
        if not np.all(np.isfinite(tau)):
            from warnings import warn
            warn(f"nonfinite autocorrelation time: {tau}", stacklevel=2)
        tau = xr.DataArray(tau, coords={"p": list(ds.keys())})
        return tau

    def get_sample(self, discard_per_autocorr, thin_per_autocorr, *,
                   var_names=None, filter_std=None, tau=None,
                   split_vectors=False, filter_kw=None, **kwargs):
        if tau is None:
            tau = self.autocorr_time
            if var_names is not None:
                tau = tau.sel(p=var_names)
            tau = np.nanmax(tau.values)

        thin = round(thin_per_autocorr * tau)
        discard = round(discard_per_autocorr * tau)

        data = get_sample(self.data, discard, thin, **kwargs)

        if var_names is not None:
            data = data[var_names]

        if filter_kw is not None:
            data = data.filter_by_attrs(**filter_kw)

        if filter_std is not None:
            data = filter_outliers_dset(data, filter_std)

        if split_vectors:
            data = split_vector_vars(data)

        return data

    def get_random_sample(self, nsamples, rng=None, **kwargs):
        # FIXME: remove "sample" dimension but preserve coords?
        sample = self.get_sample(10, 1, flat=True, **kwargs)

        return get_random_sample(sample, "sample", nsamples, rng)

    @cached_property
    def best_sample(self):
        # N.B. *not* necessarily the best/optimal fit!
        idxmax = self.data.log_prob.argmax(...)
        return self.data[idxmax]

    def get_best_sample_array(self):
        return self.best_sample.filter_by_attrs(kind="sampled").to_array().values

    def get_bounds_array(self, clip=0.025):
        ds = self.get_sample(10, 1, split_vectors=True)
        ds = ds.filter_by_attrs(kind="sampled")
        return ds.quantile([clip, 1-clip]).to_array().values

    def summary(self, discard_per_autocorr, thin_per_autocorr, var_names=None,
                rng=False, filter_std=None, hdi_prob=0.95, filter_kw=None, **kwargs):
        _ds = self.data
        if filter_kw is not None:
            _ds = self.data.filter_by_attrs(**filter_kw)
        if var_names is None:
            var_names = list(_ds.keys())
        tau = autocorr_time(_ds, discard=self._autocorr_discard)

        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names, filter_kw=filter_kw,
            filter_std=filter_std, tau=np.nanmax(tau), rng=rng, split_vectors=True,
        )

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

    def plot_autocorr_evolution(self, n0=100, nn=20, var_names=None,
                                discard=200, thin=1, filter_kw=None, **kwargs):
        ds = self.data[var_names] if var_names is not None else self.data
        if filter_kw is not None:
            ds = ds.filter_by_attrs(**filter_kw)
        ds = split_vector_vars(ds)

        return plot_autocorr_evolution(
            ds, n0=n0, nn=nn, discard=discard, thin=thin, **kwargs)

    def plot_corner(self, discard_per_autocorr=10, thin_per_autocorr=1,
                    *, var_names=None, filter_kw=None, filter_std=None, tau=None,
                    rng=False, **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names, filter_kw=filter_kw,
            filter_std=filter_std, tau=tau, rng=rng, split_vectors=True,
        )

        return plot_corner(data, **kwargs)

    def plot_trace_2d(self, *, var_names=None, filter_kw=None, draw=None,
                      split_at_per_autocorr=10, ratio=1/4, **kwargs):
        ds = self.data[var_names] if var_names is not None else self.data
        if filter_kw is not None:
            ds = ds.filter_by_attrs(**filter_kw)
        if draw is not None:
            ds = ds.sel(draw=draw)
        ds = split_vector_vars(ds)

        if split_at_per_autocorr is not None:
            split_at = round(split_at_per_autocorr * np.nanmax(self.autocorr_time))
        else:
            split_at = None

        return plot_trace_2d(ds, split_at=split_at, ratio=ratio, **kwargs)

    def plot_1d_posterior(self, discard_per_autocorr=10, thin_per_autocorr=1,
                          *, var_names=None, filter_kw=None, filter_std=None,
                          tau=None, rng=False, **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names, filter_kw=filter_kw,
            filter_std=filter_std, tau=tau, rng=rng, split_vectors=True,
        )

        return plot_1d_posterior(data, **kwargs)

    @cached_property
    def covariance_matrix(self):
        # FIXME: arguments?
        # FIXME: xarray output
        sample = self.get_sample(
            discard_per_autocorr=10, thin_per_autocorr=1,
            flat=True, split_vectors=True)
        sample = sample.filter_by_attrs(kind="sampled")
        return np.cov(sample.to_array().values)

    @cached_property
    def errors(self):
        return np.sqrt(np.diagonal(self.covariance_matrix))

    @cached_property
    def correlation_matrix(self):
        sig_sig = np.outer(self.errors, self.errors)
        return self.covariance_matrix / sig_sig

    def project_sample(self, sample, func, nthreads=None, filter_kw=None, **kwargs):
        if filter_kw is None:
            filter_kw = {"kind": "sampled"}
        sample = sample.filter_by_attrs(**filter_kw)

        kw = self.fixed_parameters | kwargs

        return project_sample(sample, func, nthreads=nthreads, **kw)


def project_sample(sample, func, nthreads=None, **kwargs):
    from functools import partial
    func = partial(func, **kwargs)

    from multiprocessing import Pool

    with Pool(nthreads) as pool:
        result = grouped_map(sample, "sample", func, mapper=pool.map)

    return result


def _get_datasets_for_compare(results,
                              discard_per_autocorr=10, thin_per_autocorr=1,
                              sampled=True, log_prob=False, derived=False,
                              filter_std=None, var_names=None, rng=False, **kwargs):
    def _get_names(res):
        if var_names is not None:
            return ordered_intersection([var_names, list(res.data.keys())])

        names = []
        if sampled:
            names.extend(res.data.filter_by_attrs(kind="sampled").keys())
        if log_prob:
            names.extend(res.data.filter_by_attrs(kind="log_prob").keys())
        if derived:
            names.extend(res.data.filter_by_attrs(kind="derived").keys())

        return names

    datasets = [
        res.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=_get_names(res),
            filter_std=filter_std, rng=rng, split_vectors=True,
        )
        for res in results
    ]

    return datasets


def compare_results_1d(results, **kwargs):
    datasets = _get_datasets_for_compare(results, **kwargs)
    return compare_1d_posteriors(datasets, **kwargs)


def compare_results_2d(results, var_names=None, **kwargs):
    rowcols = ordered_union([kwargs.get("rows", []), kwargs.get("cols", [])])
    if rowcols:
        if var_names:
            raise ValueError("passing var_names and rows/cols")
        var_names = rowcols

    datasets = _get_datasets_for_compare(results, var_names=var_names, **kwargs)
    return compare_2d_posteriors(datasets, **kwargs)
