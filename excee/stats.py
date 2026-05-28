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
from scipy.fft import next_fast_len
import xarray as xr

logger = logging.getLogger(__name__)


def _integrated_time(x, has_chain_axis, *, window_method="geyer", sokal_c=5):
    if not has_chain_axis:
        x = np.expand_dims(x, axis=-2)

    n_w, n_t = x.shape[-2:]
    n_pad = next_fast_len(2 * n_t)

    x = x - np.mean(x, axis=-1, keepdims=True)
    f = np.fft.rfft(x, n=n_pad, axis=-1)

    mean_power = np.mean(f.real**2 + f.imag**2, axis=-2)
    acf = np.fft.irfft(mean_power, n=n_pad, axis=-1)[..., :n_t]

    var = acf[..., :1].copy()
    acf /= var

    # ensemble coupling via mean-field trick
    if n_w > 1:
        f_mean = np.mean(f, axis=-2)
        blob_power = f_mean.real**2 + f_mean.imag**2
        cross_power = (n_w / (n_w - 1)) * (blob_power - mean_power / n_w)
        eccf = np.fft.irfft(cross_power, n=n_pad, axis=-1)[..., :n_t]
        eccf /= var
    else:
        tau_cross = 0

    if window_method == "sokal":
        taus = 2 * np.cumsum(acf, axis=-1) - 1
        windows = np.argmax(np.arange(n_t) - sokal_c * taus >= 0, axis=-1)
        windows = np.where(windows == 0, n_t - 1, windows)
        tau = np.take_along_axis(taus, windows[..., None], axis=-1)[..., 0]

        if n_w > 1:
            taus_crosses = 2 * np.cumsum(eccf, axis=-1) - eccf[..., :1]
            tau_cross = np.take_along_axis(
                taus_crosses, windows[..., None], axis=-1,
            )[..., 0]
    elif window_method == "geyer":
        n_pairs = n_t // 2
        acf_pairs = acf[..., 0:2*n_pairs:2] + acf[..., 1:2*n_pairs:2]
        ims = np.minimum.accumulate(acf_pairs, axis=-1)  # initial monotone sequence
        mask = ims > 0  # once zero always zero
        tau = 2 * np.sum(ims, where=mask, axis=-1) - 1

        if n_w > 1:
            eccf_pairs = eccf[..., 0:2*n_pairs:2] + eccf[..., 1:2*n_pairs:2]
            tau_cross = 2 * np.sum(eccf_pairs, where=mask, axis=-1) - eccf[..., 0]
    else:
        raise NotImplementedError(f"{window_method=}")

    coupling_penalty = (n_w - 1) * tau_cross / tau  # pylint: disable=E0601

    return tau, coupling_penalty


def autocorr_time(x, discard=0, thin=1,
                  chain_dim="chain", draw_dim="draw", has_chain_axis=None, **kwargs):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        if has_chain_axis is None:
            has_chain_axis = chain_dim in x.dims and chain_dim is not None
        core_dims = [dim for dim in [chain_dim, draw_dim] if dim in x.dims]
        tau, penalty = xr.apply_ufunc(
            _integrated_time,
            x.sel({draw_dim: slice(discard, None, thin)}),
            kwargs={"has_chain_axis": has_chain_axis, **kwargs},
            input_core_dims=[core_dims],
            output_core_dims=[[], []],
            vectorize=False,
        )
    else:
        x = np.asarray(x)[..., discard::thin]
        if has_chain_axis is None:
            has_chain_axis = x.ndim > 1

        tau, penalty = _integrated_time(x, has_chain_axis, **kwargs)

    return thin * tau, penalty


def autocorr_time_over_time(x, ns, draw_dim="draw", **kwargs):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        results = [
            autocorr_time(
                x.sel({draw_dim: slice(None, n)}),
                draw_dim=draw_dim, **kwargs,
            )
            for n in ns
        ]
        return tuple(
            xr.concat(x, dim="max_draw").assign_coords(max_draw=ns)
            for x in zip(*results)
        )
    else:
        results = [autocorr_time(x[..., :n], **kwargs) for n in ns]
        return tuple(np.stack(x, axis=0) for x in zip(*results))
