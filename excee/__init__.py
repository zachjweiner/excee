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


import xarray as xr
from excee.sampling import (
    SampleParameter, LogUniformSampleParameter, GaussianSampleParameter,
    ExpUniformSampleParameter, PowUniformSampleParameter, FixedParameter,
    GaussianLikelihood, LikelihoodSampler,
)
from excee.stats import autocorr_time, autocorr_time_over_time, eff_gaussian_tension
from excee.bandwidth import kde_bandwidth
from excee.plot import (
    plot_autocorr_evolution, plot_trace_2d, plot_joint_dist,
    compare_1d_dists, compare_2d_dists, test_smoothing, plot_1d_dists, plot_violin,
    get_2d_level,
)
from excee.analysis import (
    get_sample, filter_outliers, filter_outliers_dset,
    SamplingResult, compare_results_1d, compare_results_2d,
)


def restore_dsets(dt, vkey="variable"):
    data = {}
    for path, node in dt.subtree_with_keys:
        ds = node.dataset
        if ds is None or vkey not in ds.sizes:
            data[path] = ds
        else:
            for vname, da in ds.data_vars.items():
                data[f"{path}/{vname}"] = (
                    da.to_dataset(dim=vkey) if vkey in da.sizes
                    else da
                )

    return xr.DataTree.from_dict(data)


def assemble_posterior(dt, attrs=("long_name", "kind", "ess")):
    data_paths = {
        path for path, node in dt.match("*/data").subtree_with_keys
        if node.has_data
    }

    data = {}
    for path, node in dt.subtree_with_keys:
        ds = node.dataset
        if path in data_paths and ds is not None:
            for key in ds:
                _attrs = {
                    attr: node.parent[attr][key].values[()]
                    for attr in attrs if attr in node.parent
                }
                ds[key].attrs.update(**_attrs)
        data[path] = ds

    dt2 = xr.DataTree.from_dict(data)
    return dt2.filter(lambda node: node.name not in attrs)


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
    dt = restore_dsets(dt)
    dt = assemble_posterior(dt)
    if posterior_only:
        dt = extract_posterior(dt)
    return dt


__all__ = [
    "autocorr_time",
    "autocorr_time_over_time",
    "kde_bandwidth",
    "SampleParameter",
    "LogUniformSampleParameter",
    "GaussianSampleParameter",
    "ExpUniformSampleParameter",
    "PowUniformSampleParameter",
    "FixedParameter",
    "GaussianLikelihood",
    "LikelihoodSampler",
    "get_sample",
    "filter_outliers",
    "filter_outliers_dset",
    "plot_autocorr_evolution",
    "plot_trace_2d",
    "plot_joint_dist",
    "compare_1d_dists",
    "compare_2d_dists",
    "test_smoothing",
    "plot_1d_dists",
    "plot_violin",
    "compare_results_1d",
    "compare_results_2d",
    "get_2d_level",
    "SamplingResult",
    "load_result_tree",
    "eff_gaussian_tension",
]
