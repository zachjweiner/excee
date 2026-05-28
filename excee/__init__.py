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
from excee.stats import autocorr_time, autocorr_time_over_time
from excee.bandwidth import kde_bandwidth
from excee.plot import (
    plot_autocorr_evolution, plot_trace_2d, plot_joint_dist,
    compare_1d_dists, compare_2d_dists, plot_1d_dists, plot_violin,
    get_2d_level,
)
from excee.analysis import (
    get_sample, filter_outliers, filter_outliers_dset,
    SamplingResult, compare_results_1d, compare_results_2d,
)


@np.vectorize(signature="(n),(m),()->(),(),()")
def _eff_gaussian_tension(x, y, quiet=False):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n, m = x.size, y.size

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    U = np.searchsorted(x_sorted, y_sorted, side="left") / n
    V = (m - np.searchsorted(y_sorted, x_sorted, side="right")) / m
    p = np.mean(U)

    # DeLong's exact empirical variance of the U-statistic
    var_U = np.var(U, ddof=1) if m > 1 else 0
    var_V = np.var(V, ddof=1) if n > 1 else 0
    var_p = var_U / m + var_V / n
    se_p = np.sqrt(var_p)

    ts = norm.ppf(np.clip(p + np.arange(-1, 2) * se_p, 0, 1))
    err_m, err_p = np.diff(ts)
    t_est = ts[1]

    n_eff = min(n, m)
    p_tail = min(p, 1 - p)
    min_ideal = 1 / p_tail
    expected_crossings = n_eff * p_tail
    min_marginal = np.exp(t_est**2 / 2)

    if n_eff < min_marginal and not quiet:
        import warnings
        warnings.warn(
            f"(n_x, n_y) = ({n}, {m}) is an insufficient sample size to robustly"
            f" quantify tension of estimated size {abs(t_est):.2f} sigma."
            f" Only ~{expected_crossings:.2g} samples are expected to span across"
            f" the distributions, so the estimate is likely biased and"
            f" underreporting its uncertainty."
            f" At least ~{min_marginal:.2e} samples are required for a marginal"
            f"estimate and ~{min_ideal:.2e} for strict theoretical accuracy.",
            category=UserWarning,
            stacklevel=2,
        )

    return t_est, err_m, err_p


def eff_gaussian_tension(x, y, *, quiet=False, sample_dims=("chain", "draw")):
    sample_dims = list(sample_dims)
    if any(
        isinstance(arg, (xr.DataArray, xr.Dataset, xr.DataTree))
        for arg in (x, y)
    ):
        def regularize(z):
            return z if "sample" in z.dims else z.stack(sample=sample_dims)
        return xr.apply_ufunc(
            _eff_gaussian_tension,
            regularize(x), regularize(y), quiet,
            input_core_dims=[["sample"], ["sample"], []],
            output_core_dims=[[], [], []],
            exclude_dims={"sample"},
            dataset_join="inner",
        )
    else:
        return _eff_gaussian_tension(x, y)


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
    "plot_1d_dists",
    "plot_violin",
    "compare_results_1d",
    "compare_results_2d",
    "SamplingResult",
    "get_2d_level",
    "eff_gaussian_tension",
]
