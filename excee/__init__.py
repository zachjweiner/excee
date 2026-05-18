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
from scipy.stats import norm
from excee.sampling import (
    SampleParameter, LogUniformSampleParameter, GaussianSampleParameter,
    ExpUniformSampleParameter, PowUniformSampleParameter, FixedParameter,
    GaussianLikelihood, LikelihoodSampler,
)
from excee.autocorr import autocorr_time, autocorr_time_over_time
from excee.plot import (
    plot_autocorr_evolution, plot_trace_2d, plot_joint_dist,
    compare_1d_dists, compare_2d_dists, plot_1d_dists, plot_violin,
    get_2d_level,
)
from excee.analysis import (
    get_sample, filter_outliers, filter_outliers_dset,
    SamplingResult, compare_results_1d, compare_results_2d,
)


@np.vectorize(signature="(n),(m)->(),()")
def _eff_gaussian_distance(x, y):
    n_less = np.sum(np.searchsorted(np.sort(x), y))
    p = n_less / (x.size * y.size)
    s = norm.ppf(p)

    # FIXME
    n_eff = (x.size * y.size) / (x.size + y.size)
    delta_p = np.sqrt((p * (1 - p)) / n_eff)
    return s, delta_p / norm.pdf(s)


def eff_gaussian_distance(x, y, *, sample_dims=("chain", "draw")):
    sample_dims = list(sample_dims)
    is_xr = all(
        isinstance(arg, (xr.DataArray, xr.Dataset, xr.DataTree))
        for arg in (x, y)
    )
    if is_xr:
        def regularize(z):
            return z if "sample" in z.dims else z.stack(sample=sample_dims)

        return xr.apply_ufunc(
            _eff_gaussian_distance,
            regularize(x), regularize(y),
            input_core_dims=[["sample"], ["sample"]],
            output_core_dims=[[], []],
            exclude_dims={"sample"},
            dataset_join="inner",
        )
    else:
        return _eff_gaussian_distance(x, y)


__all__ = [
    "autocorr_time",
    "autocorr_time_over_time",
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
    "plot_1d_dists",
    "plot_violin",
    "compare_results_1d",
    "compare_results_2d",
    "SamplingResult",
    "get_2d_level",
    "eff_gaussian_distance",
]
