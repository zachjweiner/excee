__copyright__ = """
Copyright (c) 2026, ArviZ devs
Copyright (C) 2026 Zachary J Weiner
"""

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
from scipy.optimize import brentq
from arviz_stats.base import array_stats
from excee.autocorr import autocorr_time


def bw_isj(x, bounds=(None, None)):
    x_len = len(x)
    x_std = np.std(x)
    if x_std == 0:
        return 0.0

    grid_min = np.min(x) - 0.5 * x_std if bounds[0] is None else bounds[0]
    grid_max = np.max(x) + 0.5 * x_std if bounds[1] is None else bounds[1]
    grid_range = grid_max - grid_min

    grid_len = 256
    k = np.arange(grid_len)
    dct_weights = 2 * np.exp(-1j * k * np.pi / (2 * grid_len))
    dct_weights[0] = 1

    j = np.arange(6, 1, -1)
    # cumprod generates the double factorials
    c_n = (
        (1 + 0.5**(j + 1/2)) / 3
        * np.cumprod(np.arange(1, 12, 2))[j - 1] / np.sqrt(np.pi / 2)
    )
    p = 2.0 / (3.0 + 2.0 * j)
    f_m = np.pi**(2 * j) / 2

    indices = ((x - grid_min) * (grid_len / grid_range)).astype(np.intp)
    indices = np.clip(indices, 0, grid_len - 1)
    grid_relfreq = np.bincount(indices, minlength=grid_len) / x_len

    even_increasing = np.arange(0, grid_len, 2)
    odd_decreasing = np.arange(grid_len - 1, 0, -2)
    x_reordered = np.concatenate(
        (grid_relfreq[even_increasing], grid_relfreq[odd_decreasing])
    )
    a_k = np.real(dct_weights * np.fft.fft(x_reordered))

    k_sq = np.arange(1, grid_len, dtype=np.float64)**2
    a_sq = a_k[1:]**2
    K = k_sq * np.pi**2

    a_k_7 = a_sq * k_sq**7
    a_k_j = a_sq * k_sq**j[:, None]

    def fixed_point(t):
        f = np.sum(a_k_7 * np.exp(-K * t)) * np.pi**14 / 2
        for i in range(5):
            t_j = (c_n[i] / (x_len * f))**p[i]
            f = np.sum(a_k_j[i] * np.exp(-K * t_j)) * f_m[i]
        return t - (2 * np.sqrt(np.pi) * x_len * f)**(-2/5)

    try:
        bw = brentq(fixed_point, 0, 0.01, disp=False)
    except ValueError:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        h = (iqr / 1.34) if iqr > 0 else x_std
        return 0.9 * min(x_std, h) * x_len**(-1/5)

    return np.sqrt(bw) * grid_range


def robust_isj(data, N_eff=None, n_groups=13, seed=45397):
    data = np.asarray(data)
    if N_eff is None:
        tau = autocorr_time(data)[0]
        N_eff = data.size / tau
    else:
        tau = data.size / N_eff

    skip = max(1, int(np.round(tau)))
    data = data[..., ::skip]

    n_groups = min(data.shape[0], n_groups)
    rng = np.random.default_rng(seed)
    # shuffle chain axis if present
    groups = np.array_split(
        rng.permutation(data) if data.ndim != 1 else data,
        n_groups,
    )

    from excee.density import detect_boundaries
    bounded = detect_boundaries(data.ravel())
    bounds = (
        np.min(data) if bounded[0] else None,
        np.max(data) if bounded[1] else None,
    )
    bws = [
        bw_isj(group.ravel(), bounds=bounds)
        * (N_eff / group.size)**(-1/5)
        for group in groups
    ]
    n_75 = 3 * (n_groups - 1) // 4
    return sorted(bws)[n_75]


def _get_bw(data, bw="robust_isj", ess=None, dim=1, has_chain_axis=True, **kwargs):
    if dim not in (1, 2):
        raise NotImplementedError(f"{dim=}")
    s = -2 if has_chain_axis else -1
    N = np.prod(data.shape[s:])
    if ess is None:
        tau = autocorr_time(data, has_chain_axis=has_chain_axis)[0]
        ess = N / tau
    ess = np.broadcast_to(ess, data.shape[:s]).ravel()

    if bw == "robust_isj":
        _data = data.reshape(-1, *data.shape[s:])
        h = np.array([
            robust_isj(x, N_eff=N_eff, **kwargs)
            * (N_eff**(1/5-1/6) if dim == 2 else 1)
            for x, N_eff in zip(_data, ess)
        ])
    elif isinstance(bw, str):
        _data = data.reshape(-1, N)
        h = np.array([
            array_stats.get_bw(x, bw=bw, **kwargs)
            * (N_eff / N)**(-1/5)
            * (N_eff**(1/5-1/6) if dim == 2 else 1)
            for x, N_eff in zip(_data, ess)
        ])
    else:
        h = np.asarray(bw)

    return h.reshape(data.shape[:s])


def kde_bandwidth(x, chain_dim="chain", draw_dim="draw", has_chain_axis=None,
                  **kwargs):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        if has_chain_axis is None:
            has_chain_axis = chain_dim in x.dims and chain_dim is not None
        core_dims = [dim for dim in [chain_dim, draw_dim] if dim in x.dims]
        return xr.apply_ufunc(
            _get_bw, x,
            kwargs={"has_chain_axis": has_chain_axis, **kwargs},
            input_core_dims=[core_dims],
            output_core_dims=[[]],
            vectorize=False,
        )
    else:
        if has_chain_axis is None:
            has_chain_axis = x.ndim > 1
        return _get_bw(x, has_chain_axis=has_chain_axis, **kwargs)
