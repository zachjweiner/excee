__copyright__ = "Copyright (C) 2024 Zachary J Weiner"

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
from scipy.ndimage import gaussian_filter
from arviz_stats.base import array_stats

import logging
logger = logging.getLogger(__name__)


def get_bw(data, *args, **kwargs):
    shape = data.shape[1:]
    _data = np.atleast_2d(data.T)
    h = np.array([array_stats.get_bw(Z, *args, **kwargs) for Z in _data])
    return h.reshape(shape)


def _lcv(data, frac):
    # coefficient of L-variation
    n = data.shape[-1]
    k = max(3, int(n * frac))
    sorted_data = np.sort(data, axis=-1)
    tails = np.stack([sorted_data[..., :k+1], sorted_data[..., :-k-2:-1]], axis=-2)
    x = np.abs(tails[..., 1:] - tails[..., :1])

    # from scipy.stats import lmoment
    # l1, l2 = lmoment(x, order=[1, 2], axis=-1, sorted=True)
    # equivalent, but much less overhead:
    l1 = np.mean(x, axis=-1)
    weights = np.linspace(-1, 1, k)
    l2 = np.mean(weights * x, axis=-1)
    return l2 / l1


def lcv(data, frac, dim="sample", axis=-1):
    """
    Compute the coefficient of L-variation.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        The input data.
    frac : float
        Fraction of tails used in the calculation.
    dim : str, optional
        Dimension to reduce for xarray input.
    axis : int, optional
        Axis to reduce for NumPy input.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The L-CV for both tails. For NumPy input, the tail axis coincides with
        the reduction axis of the input.
    """
    if hasattr(data, "dims"):
        return xr.apply_ufunc(
            _lcv,
            data,
            kwargs={"frac": frac},
            input_core_dims=[[dim]],
            output_core_dims=[["tail"]],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    else:
        res = _lcv(np.moveaxis(data, axis, -1), frac)
        return np.moveaxis(res, -1, axis)


def detect_boundaries(data, lcv_threshold=0.22, lcv_frac=0.15):
    return lcv(data, frac=lcv_frac) > lcv_threshold


def compute_1d_density(sample, **kwargs):
    x, y, _ = array_stats.kde(np.asarray(sample), **kwargs)
    return x, y


def compute_2d_density(sample, *, weights=None, bins=256, smooth_factor=None,
                       bounds="auto", lcv_threshold=0.22, lcv_frac=0.15,
                       axes_scale="linear", pad_nstd=None, _cholesky=True):
    if weights is not None:
        raise NotImplementedError("weights")

    sample = np.asarray(sample)
    bins = np.asarray(bins) * np.ones(2, dtype=int)
    smooth_factor = 1 if smooth_factor is None else smooth_factor
    pad_nstd = pad_nstd if pad_nstd is not None else 2 if smooth_factor != 0 else 0
    axes_scale = [axes_scale]*2 if isinstance(axes_scale, str) else axes_scale
    if any(scale != "linear" for scale in axes_scale):
        raise NotImplementedError(f"{axes_scale=}")

    def get_lims(_samples):
        return np.array([np.min(_samples, axis=0), np.max(_samples, axis=0)]).T

    if bounds in (None, False, "auto"):
        bounds = [bounds, bounds]
    auto_bounds = detect_boundaries(sample.T, lcv_threshold, lcv_frac).T
    x_bounded, y_bounded = (
        auto if bound == "auto"
        else [False, False] if bound in (None, False) else bound
        for bound, auto in zip(bounds, auto_bounds)
    )
    x_lims, y_lims = get_lims(sample)
    bounds_x = np.where(x_bounded, x_lims, None)
    bounds_y = np.where(y_bounded, y_lims, None)
    logger.info(f"bounds_x = {tuple(bounds_x)}, bounds_y = {tuple(bounds_y)}")

    if any(bounds_x) and any(bounds_y):
        raise NotImplementedError("bounds in x and y")

    if any(x_bounded) and any(y_bounded) and _cholesky and smooth_factor != 0:
        logger.warning(
            "Simultaneous x and y boundaries detected. "
            "Skipping Cholesky rotation; smooth with caution.")
        _cholesky = False

    swap_axes = any(y_bounded) and not any(x_bounded) and _cholesky
    if swap_axes:
        sample = sample[:, ::-1]
        bounds, bins = bounds_y, bins[::-1]
    else:
        bounds = bounds_x

    L = np.linalg.cholesky(np.cov(sample.T)) if _cholesky else np.eye(2)
    logger.info(f"cholesky = [{L[0]}; {L[1]}]")
    samplez = sample @ np.linalg.inv(L).T
    z_lims = get_lims(samplez)
    z_spans = np.diff(z_lims, axis=1).squeeze()
    dz = z_spans / bins
    pad = pad_nstd * np.std(samplez, axis=0)
    n_pad = np.round(pad / dz).astype(int)
    pad = n_pad * dz  # pad by multiple of bin width

    bounds_z = [b / L[0, 0] if b is not None else None for b in bounds]

    def get_grid(lims, bounds, dx, pad):
        x_min, x_max = (
            lim + (sign * pad if bound is None else 0)
            for lim, bound, sign in zip(lims, bounds, [-1, 1])
        )
        n = int(np.round((x_max - x_min) / dx) + 1)
        centers, dx2 = np.linspace(x_min, x_max, n, retstep=True)
        assert abs(dx2 / dx - 1) < 1 / n / 2
        edges = np.concatenate([centers - dx/2, [centers[-1] + dx/2]])
        return edges, centers

    z1_edges, z1 = get_grid(z_lims[0], bounds_z, dz[0], pad[0])
    i0 = n_pad[0] if bounds_z[0] is None else 0
    inner_slc = slice(i0, i0 + bins[0] + 1), slice(n_pad[1], n_pad[1] + bins[1] + 1)

    z2_edges, z2 = get_grid(z_lims[1], [None, None], dz[1], pad[1])

    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")

    bws = get_bw(samplez, bw="scott") * sample.shape[0]**(1/5 - 1/6)
    logger.info(f"bandwidths = ({bws[0]}, {bws[1]})")

    pdf_Z, _, _ = np.histogram2d(
        samplez[:, 0], samplez[:, 1],
        bins=[z1_edges, z2_edges],
    )
    if bounds_z[0] is not None:
        pdf_Z[0, :] *= 2
    if bounds_z[1] is not None:
        pdf_Z[-1, :] *= 2

    if smooth_factor != 0:
        sigma = bws * smooth_factor / dz
        pdf_Z = gaussian_filter(pdf_Z, sigma=sigma, mode="mirror")
    pdf_Z /= samplez.shape[0] * np.prod(dz)

    Z = np.stack([Z1[inner_slc], Z2[inner_slc]], axis=-1)
    XY = Z @ L.T
    X, Y = XY[..., 0], XY[..., 1]
    pdf = pdf_Z[inner_slc] / np.linalg.det(L)

    return (Y.T, X.T, pdf.T) if swap_axes else (X, Y, pdf)
