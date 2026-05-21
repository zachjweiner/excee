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


from collections.abc import Iterable
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from arviz_stats.base import array_stats
from excee.bandwidth import get_bw

import logging
logger = logging.getLogger(__name__)


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


def compute_1d_density(sample, ess=None, bw_method="robust_isj", **kwargs):
    bw = get_bw(sample, bw=bw_method, ess=ess)
    logger.info(f"bandwidth = {bw}")

    x, y, _ = array_stats.kde(np.ravel(sample), bw=float(bw), **kwargs)
    return x, y


def compute_2d_density(sample, *, weights=None, bw_method="robust_isj",
                       bins=256, smooth=None, cholesky_whitening=True,
                       bounds=None, lcv_threshold=0.22, lcv_frac=0.15,
                       pad_nstd=None):
    if weights is not None:
        raise NotImplementedError("weights")

    sample = np.asarray(sample)
    _shape = sample.shape[1:]
    sample = sample.reshape(2, -1)
    bins = np.asarray(bins) * np.ones(2, dtype=int)
    smooth = 1 if smooth is None else smooth
    pad_nstd = pad_nstd if pad_nstd is not None else 2 if smooth != 0 else 0

    def get_lims(x):
        return np.stack([np.min(x, axis=-1), np.max(x, axis=-1)], axis=-1)

    xy_lims = get_lims(sample)

    def _twoify(x):
        return x if isinstance(x, Iterable) else (x, x)

    bounds = [_twoify(bnd) for bnd in _twoify(bounds)]
    bounds_detected = detect_boundaries(sample, lcv_threshold, lcv_frac)
    # only use max/min if boundary detected and input is None, else use input
    # bounds_* tuple elements are either boundary values or None if no boundary
    bounds_x, bounds_y = (
        [
            inpt if inpt is not None
            else lim if inpt is None and detected
            else None
            for (inpt, detected, lim) in zip(bnds, detections, lims)
        ]
        for (bnds, detections, lims) in zip(bounds, bounds_detected, xy_lims)
    )
    logger.info(f"bounds_x = {tuple(bounds_x)}, bounds_y = {tuple(bounds_y)}")

    x_bounded = [b is not None for b in bounds_x]
    y_bounded = [b is not None for b in bounds_y]

    if any(x_bounded) and any(y_bounded) and cholesky_whitening and smooth != 0:
        logger.warning(
            "Simultaneous x and y boundaries detected. "
            "Skipping Cholesky rotation; smooth with caution."
        )
        cholesky_whitening = False

    if cholesky_whitening:
        cov = np.cov(sample)
        L = (
            # flip to align y rather than x boundary
            np.flip(np.linalg.cholesky(np.flip(cov)))
            if any(y_bounded) and not any(x_bounded)
            else np.linalg.cholesky(cov)
        )
    else:
        L = np.eye(2)

    logger.info(f"cholesky = [{L[0]}; {L[1]}]")
    rot = np.linalg.inv(L)

    samplez = rot @ sample
    z_lims = get_lims(samplez)
    z_spans = np.diff(z_lims, axis=-1).squeeze()
    dz = z_spans / bins

    pad = pad_nstd * np.std(samplez, axis=-1)
    n_pad = np.round(pad / dz).astype(int)

    bounds_z = [
        [b * r if b is not None else None for b in bounds]
        for bounds, r in zip([bounds_x, bounds_y], np.diagonal(rot))
    ]

    def get_grid(lims, bounds, dx, n_pad, bins):
        pads = [0 if bound is not None else n_pad for bound in bounds]
        centers = lims[0] + dx * np.arange(-pads[0], bins + pads[1] + 1)
        edges = np.concatenate([centers - dx/2, [centers[-1] + dx/2]])
        inner_slc = slice(pads[0], pads[0] + bins + 1)
        return edges, centers, inner_slc

    z1_edges, z1, slc_z1 = get_grid(z_lims[0], bounds_z[0], dz[0], n_pad[0], bins[0])
    z2_edges, z2, slc_z2 = get_grid(z_lims[1], bounds_z[1], dz[1], n_pad[1], bins[1])
    inner_slc = (slc_z1, slc_z2)

    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")

    bws = get_bw(
        samplez.reshape(2, *_shape), bw=bw_method,
        has_chain_axis=len(_shape) == 2, dim=2,
    )
    logger.info(f"bandwidths = ({bws[0]}, {bws[1]})")

    pdf_Z, _, _ = np.histogram2d(
        samplez[0], samplez[1],
        bins=[z1_edges, z2_edges],
    )

    if bounds_z[0][0] is not None:
        pdf_Z[0, :] *= 2
    if bounds_z[0][1] is not None:
        pdf_Z[-1, :] *= 2
    if bounds_z[1][0] is not None:
        pdf_Z[:, 0] *= 2
    if bounds_z[1][1] is not None:
        pdf_Z[:, -1] *= 2

    if smooth != 0:
        sigma = bws * smooth / dz
        modes = [
            "mirror" if any(b is not None for b in bounds) else "constant"
            for bounds in bounds_z
        ]
        pdf_Z = gaussian_filter(pdf_Z, sigma=sigma, mode=modes)

    Z = np.stack([Z1[inner_slc], Z2[inner_slc]], axis=0)
    X, Y = (L @ Z.reshape(2, -1)).reshape(Z.shape)
    pdf = pdf_Z[inner_slc] / (sample.shape[-1] * np.prod(dz) * np.linalg.det(L))

    return X, Y, pdf
