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
from excee.analysis import discard_and_thin, get_random_sample, project_sample
from excee.io import load_result_tree

__all__ = [
    # io
    "load_result_tree",
    # analysis
    "autocorr_time",
    "autocorr_time_over_time",
    "discard_and_thin",
    "get_random_sample",
    "project_sample",
    "kde_bandwidth",
    # plot
    "plot_joint_dist",
    "compare_2d_dists",
    "compare_1d_dists",
    "plot_1d_dists",
    "compare_violin",
    "plot_violin",
    "plot_autocorr_evolution",
    "plot_trace_2d",
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
