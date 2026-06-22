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
from functools import cached_property, partial
from itertools import count
import re
from pathlib import Path
import numpy as np
import xarray as xr
import arviz_stats as az
from excee.util import (
    ordered_intersection, ordered_union, read_pickle_from_h5,
    grouped_map, label_from_attrs
)
from excee.plot import (
    plot_autocorr_evolution, plot_trace_2d, plot_joint_dist,
    compare_1d_dists, compare_2d_dists, plot_1d_dists,
)
from excee.stats import (
    autocorr_time, autocorr_time_over_time, rank_normalized_autocorr_time,
    dwell_autocorr_time, autocorr_time_profile, rejection_profile,
    acceptance_fraction, rms_jump,
)


def get_random_sample(data, axis, num_samples, rng, reindex=False):
    rng = np.random.default_rng(None if rng is True else rng)
    slc = rng.choice(len(data[axis]), size=num_samples, replace=False)
    data = data.isel({axis: slc})
    if reindex:
        data = data.assign_coords(sample=np.arange(data.sample.size))

    return data


def flatten_chains(data, reindex=True, stacked_dims=("chain", "draw")):
    data = data.stack(sample=stacked_dims)
    if reindex:
        data = data.drop_vars(["sample", "draw", "chain"])
        data = data.assign_coords(sample=np.arange(data.sample.size))
    return data


def discard_and_thin(data, discard_per_autocorr, thin_per_autocorr, *,
                     autocorr_discard=100):
    if isinstance(data, xr.DataTree):
        return data.map_over_datasets(
            lambda node: discard_and_thin(
                node, discard_per_autocorr, thin_per_autocorr
            ) if "draw" in node.sizes else node
        )

    tau = np.nanmin([
        _tau if (_tau := da.attrs.get("autocorr_time")) is not None
        else autocorr_time(da, discard=autocorr_discard)[0].values
        for da in data.values()
    ])
    thin = max(1, round(thin_per_autocorr * tau))
    discard = round(discard_per_autocorr * tau)
    return data.sel(draw=slice(discard, None, thin))


def get_sample(data, discard, thin, flat=False, rng=False, reindex=True):
    if rng is False:  # 0 is a valid seed
        data = data.isel(draw=slice(discard, None, thin))
        if flat:
            data = flatten_chains(data)
    else:
        data = data.isel(draw=slice(discard, None))

        if flat:
            data = flatten_chains(data, reindex=reindex)
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

    flatten = "sample" not in dset.dims
    if flatten:
        dset = flatten_chains(dset, reindex=False)

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

    return expand_sample_to_chain_and_draw(dset) if flatten else dset


def split_vector_vars(data, keep_dims=("chain", "draw", "sample")):
    if set(data.dims) <= set(keep_dims):
        return data

    def split_one(da):
        dims_to_split = set(da.dims) - set(keep_dims)
        if len(dims_to_split) > 1:
            raise NotImplementedError("multi-dimensional splitting")
        elif dims_to_split:
            dim, = dims_to_split
            da[dim] = [f"{da.name}_{i}" for i in range(da[dim].size)]
            dset = da.to_dataset(dim, promote_attrs=True).copy()

            for name, var in dset.items():
                prefix, idx = re.findall("([a-zA-z]+)_([0-9]+)", name)[0]
                if long_name := label_from_attrs(da):
                    prefix = long_name.replace("$", "")
                var.attrs["long_name"] = f"${prefix}_{{{idx}}}$"
                var.attrs["kind"] = dset.attrs["kind"]
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

    @property
    def sampled_names(self):
        return list(self.data.filter_by_attrs(kind="sampled").keys())

    @property
    def log_prob_names(self):
        return list(self.data.filter_by_attrs(kind="log_prob").keys())

    @property
    def derived_names(self):
        return list(self.data.filter_by_attrs(kind="derived").keys())

    @classmethod
    def from_datatree(cls, dt, vkey="variable"):
        from excee.io import to_dataset, decompress
        data = dt["data"]
        if isinstance(data, xr.DataArray):
            if "sample" in data.sizes:
                data = decompress(data)
            data = to_dataset(data, dim=vkey)
        else:
            if isinstance(data, xr.DataTree):
                data = data.to_dataset()
            if "sample" in data.sizes:
                data = data.map(decompress)

        best_fit = dt.get("best_fit", None)
        if isinstance(best_fit, xr.DataArray):
            best_fit = to_dataset(best_fit, dim=vkey)
        elif isinstance(data, xr.DataTree):
            best_fit = best_fit.to_dataset()

        fixed_parameters = {
            key: None if val == "None" else val
            for key, val in dt.attrs.items()
        }

        return cls(data, best_fit=best_fit, fixed_parameters=fixed_parameters)

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

        dim_count = count()

        def get_dims(ary):
            if ary.ndim == 3:
                pre_dims = (f"dim_{next(dim_count)}",)
            elif ary.ndim == 2:
                pre_dims = ()
            else:
                raise NotImplementedError(f"{ary.ndims=}")

            return (*pre_dims, "chain", "draw")

        chain = {
            var_name: (get_dims(chain[idx]), chain[idx])
            for idx, var_name in zip(slices, var_names)
        }

        if (blobs := backend.get_blobs()) is not None:
            if np.ndim(blobs) == 2:
                blobs = blobs[..., None]
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
    def _autocorr_time_result(self):
        ds = split_vector_vars(self.data).to_dataarray("p")
        return autocorr_time(ds, discard=self._autocorr_discard)

    @cached_property
    def autocorr_time(self):
        return self._autocorr_time_result[0]

    @cached_property
    def coupling_penalty(self):
        return self._autocorr_time_result[1]

    def get_sample(self, discard_per_autocorr, thin_per_autocorr, *,
                   var_names=None, filter_std=None, tau=None,
                   split_vectors=False, filter_kw=None, **kwargs):
        if tau is None:
            tau = self.autocorr_time
            if var_names is not None:
                tau = tau.sel(p=var_names)

            tau = np.nanmin(tau.values)

        thin = max(1, round(thin_per_autocorr * tau))
        discard = round(discard_per_autocorr * tau)

        data = get_sample(self.data, discard, thin, **kwargs)

        if var_names is not None:
            data = data[var_names]

        if filter_kw is not None:
            data = data.filter_by_attrs(**filter_kw)

        if split_vectors:
            data = split_vector_vars(data)

        N = (
            data.sizes["sample"] if "sample" in data.sizes
            else data.sizes["chain"] * data.sizes["draw"]
        )
        taus = self.autocorr_time
        penalties = self.coupling_penalty
        ess = N * thin / taus
        for key in data:
            data[key].attrs["autocorr_time"] = taus.sel(p=key).values
            data[key].attrs["coupling_penalty"] = penalties.sel(p=key).values
            data[key].attrs["ess"] = ess.sel(p=key).values

        if filter_std is not None:
            data = filter_outliers_dset(data, filter_std)

        return data

    def get_random_sample(self, nsamples, rng=None, reindex=False, **kwargs):
        # FIXME: remove "sample" dimension but preserve coords?
        sample = self.get_sample(10, 1, flat=True, reindex=reindex, **kwargs)
        return get_random_sample(sample, "sample", nsamples, rng, reindex)

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
                rng=False, filter_std=None, ci_prob=0.95, filter_kw=None, **kwargs):
        _ds = self.data
        if filter_kw is not None:
            _ds = self.data.filter_by_attrs(**filter_kw)
        if var_names is None:
            var_names = list(_ds.keys())
        tau, penalty = autocorr_time(
            _ds.to_array("p"), discard=self._autocorr_discard,
        )

        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names, filter_kw=filter_kw,
            filter_std=filter_std, tau=np.nanmin(tau), rng=rng, split_vectors=True,
        )

        summary = az.summary(data, round_to="none", ci_prob=ci_prob, **kwargs)
        summary["tau"] = tau
        summary["coupling_penalty"] = penalty

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
            kind="stats", **kwargs)
        df2 = self.summary(
            discard_per_autocorr, thin_per_autocorr,
            kind="stats_median", **kwargs)
        merged = df2.merge(df1)

        return merged.set_index(df1.index)

    def plot_autocorr_evolution(self, n0=100, nn=20, var_names=None,
                                discard=200, thin=1, **kwargs):
        ds = self.data[var_names] if var_names is not None else self.data
        filter_kw = kwargs.get(
            "filter_kw",
            {"kind": "sampled"} if var_names is None else {}
        )
        if filter_kw:
            ds = ds.filter_by_attrs(**filter_kw)
        ds = split_vector_vars(ds)

        return plot_autocorr_evolution(
            ds, n0, nn, discard=discard, thin=thin, **kwargs)

    def plot_joint_dist(self, discard_per_autocorr=10, thin_per_autocorr=1,
                        *, var_names=None, filter_kw=None, filter_std=None,
                        tau=None, rng=False, **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names, filter_kw=filter_kw,
            filter_std=filter_std, tau=tau, rng=rng, split_vectors=True,
        )

        return plot_joint_dist(data, **kwargs)

    def plot_trace_2d(self, *, var_names=None, draw=None,
                      split_at_per_autocorr=10, ratio=1/4, **kwargs):
        ds = self.data[var_names] if var_names is not None else self.data
        filter_kw = kwargs.pop(
            "filter_kw",
            {"kind": "sampled"} if var_names is None else {}
        )
        if filter_kw:
            ds = ds.filter_by_attrs(**filter_kw)
        if draw is not None:
            ds = ds.sel(draw=draw)
        ds = split_vector_vars(ds)

        if split_at_per_autocorr is not None:
            split_at = round(split_at_per_autocorr * np.nanmin(self.autocorr_time))
        else:
            split_at = None

        return plot_trace_2d(ds, split_at=split_at, ratio=ratio, **kwargs)

    def plot_1d_dists(self, discard_per_autocorr=10, thin_per_autocorr=1,
                      *, var_names=None, filter_kw=None, filter_std=None,
                      tau=None, rng=False, **kwargs):
        data = self.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=var_names, filter_kw=filter_kw,
            filter_std=filter_std, tau=tau, rng=rng, split_vectors=True,
        )

        return plot_1d_dists(data, **kwargs)

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

    def project_sample(self, func, *, sample=None, nsamples=None,
                       rng=None, filter_kw=None, exclude_fixed=False, **kwargs):
        if filter_kw is None:
            filter_kw = {"kind": "sampled"}
        sample = (
            sample if sample is not None
            else self.get_random_sample(nsamples, rng=rng)
        )
        sampled = sample.filter_by_attrs(**filter_kw)
        kw = self.fixed_parameters | kwargs if not exclude_fixed else kwargs
        return sample, project_sample(sampled, func, **kw)

    def convergence_stats(self, discard_per_autocorr=10, thin_per_autocorr=0,
                          vkey="variable", profile_splits=20, n_tau_evo=20):
        ds = self.get_sample(discard_per_autocorr, thin_per_autocorr)
        stats = {}

        tau, coupling_penalty = autocorr_time(ds)
        rn_tau, rn_coupling_penalty = rank_normalized_autocorr_time(ds)
        mean_tau, _ = autocorr_time(ds.mean(dim="chain"))
        stats["autocorr"] = {
            "autocorr_time": tau,
            "coupling_penalty": coupling_penalty,
            "rank_normalized_autocorr_time": rn_tau,
            "rank_normalized_coupling_penalty": rn_coupling_penalty,
            "mean_chain_autocorr_time": mean_tau,
        } | {
            f"tau_{meth}": 1 / az.ess(ds, method=meth, relative=True)
            for meth in ("bulk", "tail")
        }

        dn = min(100, (ds.draw.max() - ds.draw.min()) / 2)
        ns = np.geomspace(ds.draw.min() + dn, ds.draw.max(), n_tau_evo).astype(int)
        tau_evo, cp_evo = autocorr_time_over_time(ds, ns)
        stats["autocorr_evo"] = {
            "autocorr_time_evolution": tau_evo,
            "coupling_penalty_evolution": cp_evo,
        }

        stats["proposal_efficiency"] = {
            "dwell_autocorr_time": dwell_autocorr_time(ds),
            "acceptance_fraction": acceptance_fraction(ds),
            "rms_jump": rms_jump(ds),
        }

        _thin = min(8, max(1, round(tau.to_dataarray().min().values / 4)))
        ds_thin = ds.isel(draw=slice(None, None, _thin))
        stats["profiles"] = {
            "tau_profile": _thin * autocorr_time_profile(ds_thin, profile_splits),
            "rejection_profile": rejection_profile(ds, profile_splits),
        }

        stats["rhat"] = {
            f"rhat_{meth}": az.rhat(ds, method=meth)
            for meth in ("rank", "folded", "identity")
        }

        def stacked_ary(data, kind_key):
            data = {key: val.to_dataarray(vkey) for key, val in data.items()}
            return xr.Dataset(data).to_dataarray(kind_key)

        stats = {key: stacked_ary(val, f"{key}_kind") for key, val in stats.items()}
        return xr.DataTree.from_dict(stats)

    def to_datatree(self, *, compressed=True, vkey="variable", include_stats=False,
                    discard_per_autocorr=10, thin_per_autocorr=1/2,
                    **kwargs):
        from excee.io import to_dataarray, compress

        if compressed:
            data = compress(to_dataarray(self.data, dim=vkey))
            tau = self.autocorr_time.sel(p=data[vkey])
            data.attrs["__autocorr_time"] = tau.values
        else:
            data = self.get_sample(discard_per_autocorr, thin_per_autocorr)
            data = to_dataarray(data, dim=vkey)

        dt = xr.DataTree.from_dict({"data": data})

        if self.best_fit is not None:
            dt["best_fit"] = to_dataarray(self.best_fit, dim=vkey)

        dt.attrs.update({
            k: v if v is not None else "None"
            for k, v in self.fixed_parameters.items()
        })

        if include_stats:
            dt["stats"] = self.convergence_stats(
                discard_per_autocorr, vkey=vkey, **kwargs)

        return dt


def project_sample(sample, func, pool=None, progress=True, progress_kwargs=None,
                   **kwargs):
    func = partial(func, **kwargs)
    mapper = pool.map if pool else map
    return grouped_map(
        sample, "sample", func, mapper=mapper,
        progress=progress, progress_kwargs=progress_kwargs,
    )


def _get_datasets_for_compare(results, discard_per_autocorr=10, thin_per_autocorr=1,
                              var_names=None, split_vectors=True, **kwargs):
    def _get_names(res):
        # FIXME: doesn't catch split vector var names
        return (
            ordered_intersection([var_names, list(res.data.keys())])
            if var_names else None
        )

    return [
        res.get_sample(
            discard_per_autocorr, thin_per_autocorr,
            var_names=_get_names(res), split_vectors=split_vectors,
            **kwargs,
        )
        for res in results
    ]


def compare_results_1d(results, var_names=None, sample_kw=None, **kwargs):
    sample_kw = sample_kw or {}
    datasets = _get_datasets_for_compare(results, var_names=var_names, **sample_kw)
    return compare_1d_dists(datasets, **kwargs)


def compare_results_2d(results, var_names=None, sample_kw=None, **kwargs):
    sample_kw = sample_kw or {}
    rowcols = ordered_union([kwargs.get("rows", []), kwargs.get("cols", [])])
    if rowcols:
        if var_names:
            raise ValueError("passing var_names and rows/cols")
        var_names = rowcols

    datasets = _get_datasets_for_compare(results, var_names=var_names, **sample_kw)
    return compare_2d_dists(datasets, **kwargs)


__all__ = [
    "autocorr_time",
    "autocorr_time_over_time",
    "expand_sample_to_chain_and_draw",
    "filter_outliers",
    "filter_outliers_dset",
    "get_random_sample",
    "get_sample",
    "split_vector_vars",
    "project_sample",
    "SamplingResult",
    "compare_results_1d",
    "compare_results_2d",
]
