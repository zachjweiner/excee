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


import numpy as np
import xarray as xr
import h5py
from excee.sampling import (
    SampleParameter, LogUniformSampleParameter, GaussianSampleParameter,
    ExpUniformSampleParameter, PowUniformSampleParameter, FixedParameter,
    GaussianLikelihood, LikelihoodSampler,
)
from excee.stats import autocorr_time, autocorr_time_over_time, eff_gaussian_tension
from excee.bandwidth import kde_bandwidth
from excee.plot import (
    plot_autocorr_evolution, plot_trace_2d, plot_joint_dist,
    compare_2d_dists, compare_1d_dists, plot_1d_dists, plot_violin, compare_violin,
    test_smoothing, get_1d_level, get_2d_level,
    get_inclusive_limits, get_inclusive_limits_from_2d_levels
)
from excee.analysis import (
    discard_and_thin, get_sample, filter_outliers, filter_outliers_dset,
    SamplingResult, compare_results_1d, compare_results_2d,
)

vlen_str_dt = h5py.string_dtype(encoding="utf-8")


def to_dataarray(ds, dim="variable"):
    da = ds.to_dataarray(dim)
    attrs = list({key for _da in ds.variables.values() for key in _da.attrs})
    for key in attrs:
        vals = np.array([ds[k].attrs.get(key, np.nan) for k in da[dim].values])
        if vals.dtype.kind in ("U", "S", "O") and isinstance(vals.flat[0], str):
            vals = vals.astype(vlen_str_dt)
        da.attrs["__"+key] = vals

    return da


def to_dataset(da, dim="variable"):
    ds = da.to_dataset(dim)
    restore_attrs = {
        key.replace("__", ""): val
        for key, val in da.attrs.items() if key.startswith("__")
    }
    for key in restore_attrs:
        da.attrs.pop("__"+key)
    for i, key in enumerate(ds):
        _attrs = {
            attr: val for attr, vals in restore_attrs.items()
            if (val := vals[i]) not in (np.nan, "nan")
        }
        ds[key].attrs.update(**_attrs)

    return ds


def compress(da):
    reduce_dims = [d for d in da.dims if d not in ("chain", "draw")]
    # .shift() inserts nan at draw=0, so changed(draw=0) is True
    changed = (da != da.shift(draw=1)).any(dim=reduce_dims)
    changed.loc[{"draw": da.draw[-1]}] = True

    mask = changed.stack(sample=["chain", "draw"])
    da = da.stack(sample=["chain", "draw"])
    da = da.isel(sample=mask)
    da = da.reset_index("sample")
    da = da.assign_coords(
        chain=("sample", da.chain.values),
        draw=("sample", da.draw.values)
    )

    return da


def decompress(da):
    chain = da.chain.values
    draw = da.draw.values

    same_chain = np.diff(chain, append=chain[-1] + 1) == 0
    repeats = np.where(
        same_chain,
        np.diff(draw, append=0),
        draw.max() + 1 - draw
    )

    da_expanded = da.isel(sample=np.repeat(np.arange(draw.size), repeats))

    chain = np.unique(chain)
    draw = np.arange(draw.min(), draw.max() + 1)

    sample_axis = da_expanded.dims.index("sample")
    new_dims = list(da_expanded.dims)
    new_dims[sample_axis:sample_axis+1] = ["chain", "draw"]
    new_shape = list(da_expanded.shape)
    new_shape[sample_axis:sample_axis+1] = [chain.size, draw.size]
    new_coords = da_expanded.coords | {"chain": chain, "draw": draw}

    return xr.DataArray(
        da_expanded.values.reshape(new_shape),
        dims=new_dims,
        coords=new_coords,
        attrs=da_expanded.attrs,
        name=da_expanded.name
    )


def decompress_dt(dt):
    def _decompress(da):
        return decompress(da) if "sample" in da.sizes else da
    return dt.map_over_datasets(lambda node: node.map(_decompress))


def restore_dsets(dt, vkey="variable"):
    data = {}
    for path, node in dt.subtree_with_keys:
        for vname, da in node.data_vars.items():
            data[f"{path}/{vname}"] = (
                to_dataset(da, dim=vkey) if vkey in da.sizes else da
            )

    return xr.DataTree.from_dict(data)


def extract_posterior(dt):
    dt = dt.match("*/data")
    return xr.DataTree.from_dict({
        node.parent.path: node.dataset
        for node in dt.subtree
        if node.has_data
    })


def load_result_tree(path, engine="h5netcdf", posterior_only=True, groups=None,
                     **kwargs):
    if groups is None:
        dt = xr.load_datatree(path, engine=engine, **kwargs)
    else:
        dt = xr.DataTree()
        for group in groups:
            dt[group] = xr.load_datatree(path, engine=engine, group=group, **kwargs)
    dt = decompress_dt(dt)
    dt = restore_dsets(dt)
    if posterior_only:
        dt = extract_posterior(dt)
    return dt


__all__ = [
    # analysis
    "SamplingResult",
    "load_result_tree",
    "autocorr_time",
    "autocorr_time_over_time",
    "discard_and_thin",
    "get_sample",
    "kde_bandwidth",
    "filter_outliers",
    "filter_outliers_dset",
    # plot
    "plot_joint_dist",
    "compare_2d_dists",
    "compare_1d_dists",
    "plot_1d_dists",
    "compare_violin",
    "compare_results_1d",
    "compare_results_2d",
    "plot_autocorr_evolution",
    "plot_trace_2d",
    "plot_violin",
    "test_smoothing",
    "get_1d_level",
    "get_2d_level",
    "get_inclusive_limits",
    "get_inclusive_limits_from_2d_levels",
    "eff_gaussian_tension",
    # sampling
    "SampleParameter",
    "LogUniformSampleParameter",
    "GaussianSampleParameter",
    "ExpUniformSampleParameter",
    "PowUniformSampleParameter",
    "FixedParameter",
    "GaussianLikelihood",
    "LikelihoodSampler",
]
