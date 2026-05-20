__copyright__ = """
Copyright (c) 2012-2021 Dan Foreman-Mackey & emcee contributors
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


import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _integrated_time(x, c, tol, has_chain_axis):
    if not has_chain_axis:
        x = np.expand_dims(x, axis=-2)

    n_w, n_t = x.shape[-2:]
    n_pad = 2 * int(2**np.ceil(np.log2(n_t)))

    x = x - np.mean(x, axis=-1, keepdims=True)
    f = np.fft.rfft(x, n=n_pad, axis=-1)

    mean_power = np.mean(f.real**2 + f.imag**2, axis=-2)
    acf = np.fft.irfft(mean_power, n=n_pad, axis=-1)[..., :n_t]

    var = acf[..., 0:1].copy()
    acf /= var

    # ensemble coupling via mean-field trick
    if n_w > 1:
        f_mean = np.mean(f, axis=-2)
        blob_power = f_mean.real**2 + f_mean.imag**2
        cross_power = (n_w / (n_w - 1)) * (blob_power - mean_power / n_w)
        eccf = np.fft.irfft(cross_power, n=n_pad, axis=-1)[..., :n_t]
        eccf /= var
        taus_cross = 2 * np.cumsum(eccf, axis=-1) - eccf[..., 0:1]
    else:
        taus_cross = np.zeros_like(acf)

    taus = 2 * np.cumsum(acf, axis=-1) - 1
    cond = np.arange(n_t) >= c * taus
    windows = np.argmax(cond, axis=-1)

    # override windows where the condition is never met
    no_crossing = ~np.any(cond, axis=-1)
    windows = np.where(no_crossing, n_t - 1, windows)

    tau_est = np.take_along_axis(taus, windows[..., None], axis=-1)[..., 0]
    tau_cross_est = np.take_along_axis(
        taus_cross, windows[..., None], axis=-1,
    )[..., 0]
    coupling_penalty = (n_w - 1) * tau_cross_est / tau_est

    flag = tol * tau_est > n_t
    if np.any(flag):
        logger.warning(
            f"chain is fewer than {tol} autocorrelation times long"
            f" for {np.sum(flag)} parameter(s), with"
            f" max(tau) = {np.max(tau_est):.2f}"
            f" and N/tau = {n_t / np.max(tau_est):.2f} at worst"
        )

    return tau_est, coupling_penalty


def autocorr_time(x, discard=0, thin=1, c=5, tol=50,
                  chain_dim="chain", draw_dim="draw", has_chain_axis=None):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        if has_chain_axis is None:
            has_chain_axis = chain_dim in x.dims and chain_dim is not None
        core_dims = [dim for dim in [chain_dim, draw_dim] if dim in x.dims]
        taus, penalties = xr.apply_ufunc(
            _integrated_time,
            x.isel({draw_dim: slice(discard, None, thin)}),
            kwargs={"c": c, "tol": tol, "has_chain_axis": has_chain_axis},
            input_core_dims=[core_dims],
            output_core_dims=[[], []],
            vectorize=False,
        )
    else:
        x = np.asarray(x)[..., discard::thin]
        if has_chain_axis is None:
            has_chain_axis = x.ndim > 1

        taus, penalties = _integrated_time(x, c, tol, has_chain_axis=has_chain_axis)

    return thin * taus, penalties


def autocorr_time_over_time(x, ns, tol=0, draw_dim="draw", **kwargs):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        results = [
            autocorr_time(
                x.isel({draw_dim: slice(None, n)}),
                tol=tol, draw_dim=draw_dim, **kwargs,
            )
            for n in ns
        ]
        return tuple(
            xr.concat(x, dim="n").assign_coords(n=ns) for x in zip(*results)
        )
    else:
        results = [
            autocorr_time(x[..., :n], tol=tol, **kwargs)
            for n in ns
        ]
        return tuple(np.stack(x, axis=0) for x in zip(*results))
