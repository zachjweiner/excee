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

    n_t = x.shape[-1]
    n_pad = 2 * int(2**np.ceil(np.log2(n_t)))

    x = x - np.mean(x, axis=-1, keepdims=True)
    f = np.fft.rfft(x, n=n_pad, axis=-1)

    mean_power = np.mean(f.real**2 + f.imag**2, axis=-2)
    acf = np.fft.irfft(mean_power, n=n_pad, axis=-1)[..., :n_t]
    acf /= acf[..., 0:1]

    # window search
    taus = 2 * np.cumsum(acf, axis=-1) - 1
    cond = np.arange(n_t) >= c * taus
    windows = np.argmax(cond, axis=-1)

    # override windows where the condition is never met.
    no_crossing = ~np.any(cond, axis=-1)
    windows = np.where(no_crossing, n_t - 1, windows)

    tau_est = np.take_along_axis(taus, windows[..., np.newaxis], axis=-1)[..., 0]

    flag = tol * tau_est > n_t
    if np.any(flag):
        msg = (
            f"The chain is shorter than {tol} times the integrated "
            f"autocorrelation time for {np.sum(flag)} parameter(s). "
            f"Use this estimate with caution and run a longer chain!\n"
            f"N/{tol} = {n_t / tol:.0f};\ntau max: {np.max(tau_est):.2f}"
        )
        logger.warning(msg)

    return tau_est


def autocorr_time(x, discard=0, thin=1, c=5, tol=50,
                  chain_dim="chain", draw_dim="draw", has_chain_axis=None):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        if has_chain_axis is None:
            has_chain_axis = chain_dim in x.dims and chain_dim is not None
        core_dims = [dim for dim in [chain_dim, draw_dim] if dim in x.dims]
        return thin * xr.apply_ufunc(
            _integrated_time,
            x.isel({draw_dim: slice(discard, None, thin)}),
            kwargs={"c": c, "tol": tol, "has_chain_axis": has_chain_axis},
            input_core_dims=[core_dims],
            output_core_dims=[[]],
            vectorize=False,
        )
    else:
        x = np.asarray(x)[..., discard::thin]
        if has_chain_axis is None:
            has_chain_axis = x.ndim > 1

        return thin * _integrated_time(x, c, tol, has_chain_axis=has_chain_axis)


def autocorr_time_over_time(x, ns, tol=0, draw_dim="draw", **kwargs):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        results = [
            autocorr_time(
                x.isel(draw=slice(None, n)),
                tol=tol, draw_dim=draw_dim, **kwargs,
            )
            for n in ns
        ]
        res = xr.concat(results, dim="n")
        return res.assign_coords(n=ns)
    else:
        results = [
            autocorr_time(x[..., :n], tol=tol, **kwargs)
            for n in ns
        ]
        return np.stack(results, axis=0)
