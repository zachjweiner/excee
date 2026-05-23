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
from excee.autocorr import autocorr_time

import logging
logger = logging.getLogger(__name__)


def bw_scott(x, ess=None, std=None):
    ess = ess if ess is not None else x.shape[-1]
    std = std if std is not None else np.std(x, axis=-1)
    return 1.06 * std * ess**(-1/5)


def bw_silverman(x, ess=None, std=None):
    ess = ess if ess is not None else x.shape[-1]
    std = std if std is not None else np.std(x, axis=-1)
    iqr = np.diff(np.quantile(x, [0.25, 0.75], axis=-1), axis=0).squeeze(axis=0)
    return 0.9 * np.minimum(std, iqr / 1.3489795) * ess**(-1/5)


def bw_isj(x, ess=None, bounds=None):
    ess = ess if ess is not None else x.size

    from excee.density import detect_boundaries
    if bounds is None:
        bounded = detect_boundaries(x)
        # FIXME: respect user bounds?
        bounds = (
            np.min(x) if bounded[0] else None,
            np.max(x) if bounded[1] else None,
        )

    std = np.std(x)
    grid_min = np.min(x) - 0.5 * std if bounds[0] is None else bounds[0]
    grid_max = np.max(x) + 0.5 * std if bounds[1] is None else bounds[1]
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
    grid_relfreq = np.bincount(indices, minlength=grid_len) / x.size

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
            t_j = (c_n[i] / (ess * f))**p[i]
            f = np.sum(a_k_j[i] * np.exp(-K * t_j)) * f_m[i]
        return t - (2 * np.sqrt(np.pi) * ess * f)**(-2/5)

    try:
        bw = brentq(fixed_point, 0, 0.01, disp=False)
    except ValueError:
        logger.warning("ISJ optimization failed")
        return bw_silverman(x, ess=ess, std=std)

    return np.sqrt(bw) * grid_range


def _get_bw(data, bw="isj", ess=None, dim=1, has_chain_axis=True, **kwargs):
    s = -2 if has_chain_axis else -1
    N = np.prod(data.shape[s:])
    if ess is None:
        tau = autocorr_time(data, has_chain_axis=has_chain_axis)[0]
        ess = N / tau
    ess = np.broadcast_to(ess, data.shape[:s])

    N_rescaling_exp = 1 / 5 - 1 / (4 + dim)

    if bw == "isj":
        _data = data.reshape(-1, *data.shape[s:])
        h = np.array([
            bw_isj(x.ravel(), ess=_ess, **kwargs)
            * _ess**N_rescaling_exp
            for x, _ess in zip(_data, ess.ravel())
        ])
        return h.reshape(data.shape[:s])
    elif bw in ("scott", "silverman",):
        _data = data.reshape(*data.shape[:s], -1)
        bw_func = bw_scott if bw == "scott" else bw_silverman
        return bw_func(_data, ess=ess) * ess**N_rescaling_exp
    else:
        return np.broadcast_to(np.asarray(bw), data.shape[:s])


def kde_bandwidth(x, *, chain_dim="chain", draw_dim="draw", has_chain_axis=None,
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
