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
from scipy.ndimage import gaussian_filter
from arviz_stats.base import array_stats

import logging
logger = logging.getLogger(__name__)


def get_bw(data, *args, **kwargs):
    shape = data.shape[1:]
    _data = np.atleast_2d(data.T)
    h = np.array([array_stats.get_bw(Z, *args, **kwargs) for Z in _data])
    return h.reshape(shape)


def autodetect_bounds(data, threshold):
    h = get_bw(data, bw="scott")
    _min = np.min(data, axis=0)
    _max = np.max(data, axis=0)
    f_bot = np.mean(data <= _min + h, axis=0)
    f_top = np.mean(data >= _max - h, axis=0)
    return np.array([
        np.where(f_bot >= threshold, _min, None),
        np.where(f_top >= threshold, _max, None)
    ]).T


def compute_1d_density(sample, **kwargs):
    x, y, _ = array_stats.kde(np.asarray(sample), **kwargs)
    return x, y


def compute_2d_density(sample, *, weights=None, bins=256,
                       bounds="auto", bound_threshold=0.015,
                       smooth_factor=None, use_kdepy=False,
                       pad_nstd=None, axes_scale="linear", _cholesky=True):
    if weights is not None:
        raise NotImplementedError("weights")

    sample = np.asarray(sample)
    bins = np.asarray(bins) * np.ones(2, dtype=int)
    smooth_factor = 1 if smooth_factor is None else smooth_factor
    pad_nstd = pad_nstd if pad_nstd is not None else 2 if smooth_factor != 0 else 0
    axes_scale = [axes_scale]*2 if isinstance(axes_scale, str) else axes_scale
    if any(scale != "linear" for scale in axes_scale):
        raise NotImplementedError(f"{axes_scale=}")

    if bounds in (None, "auto"):
        bounds = [bounds, bounds]
    auto_bounds = autodetect_bounds(sample, bound_threshold)
    bounds_x, bounds_y = (
        auto if bound == "auto" else [None, None] if bound is None else bound
        for bound, auto in zip(bounds, auto_bounds)
    )
    logger.info(f"bounds_x = {tuple(bounds_x)}, bounds_y = {tuple(bounds_y)}")
    has_x_bound = any(b is not None for b in bounds_x)
    has_y_bound = any(b is not None for b in bounds_y)
    if has_x_bound and has_y_bound:
        logger.warning(
            "Simultaneous x and y boundaries detected. "
            "Skipping Cholesky rotation; smooth with caution.")
        _cholesky = False

    swap_axes = has_y_bound
    if swap_axes:
        sample = sample[:, ::-1]
        bounds, bins = bounds_y, bins[::-1]
    else:
        bounds = bounds_x

    def get_lims(_samples):
        return np.array([np.min(_samples, axis=0), np.max(_samples, axis=0)]).T

    L = np.linalg.cholesky(np.cov(sample.T)) if _cholesky else np.eye(2)
    logger.info(f"cholesky = [{L[0]}; {L[1]}]")
    inner_samplez = sample @ np.linalg.inv(L).T
    inner_z_lims = get_lims(inner_samplez)
    inner_z_spans = np.diff(inner_z_lims, axis=1).squeeze()
    dz = inner_z_spans / bins
    pad = pad_nstd * np.std(inner_samplez, axis=0)
    n_pad = np.round(pad / dz).astype(int)
    pad = n_pad * dz  # pad by multiple of bin width

    bounds_z = [b / L[0, 0] if b is not None else None for b in bounds]
    mirrors = [
        np.array([2 * b, 0]) + np.array([-1, 1]) * inner_samplez
        for b in bounds_z if b is not None
    ]
    mass_multiplier = 1 + len(mirrors)
    samplez = np.vstack([inner_samplez, *mirrors])
    z_lims = get_lims(samplez)

    def get_grid(lims, bounds, dx, pad):
        x_min, x_max = (
            lim + (sign * pad if bound is None else 0)
            for lim, bound, sign in zip(lims, bounds, [-1, 1])
        )
        n = int(np.round((x_max - x_min) / dx) + 1)
        centers, dx2 = np.linspace(x_min, x_max, n, retstep=True)
        assert abs(dx2 / dx - 1) < 1e-12
        edges = np.concatenate([centers - dx/2, [centers[-1] + dx/2]])
        return edges, centers

    z1_edges, z1 = get_grid(z_lims[0], bounds_z, dz[0], pad[0])
    i0 = n_pad[0] if bounds_z[0] is None else bins[0]
    z1_inner_slc = slice(i0, i0 + bins[0] + 1)

    z2_edges, z2 = get_grid(z_lims[1], [None, None], dz[1], pad[1])

    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")

    bws = get_bw(samplez, bw="scott") * samplez.shape[0]**(1/5 - 1/6)
    logger.info(f"bandwidths = ({bws[0]}, {bws[1]})")

    if use_kdepy:
        if smooth_factor == 0.:
            raise ValueError("KDEpy without smoothing")
        from KDEpy import FFTKDE
        kde = FFTKDE(bw=1).fit(samplez / bws)
        grid_pts = np.stack([Z1, Z2], axis=-1).reshape(-1, 2) / bws
        pdf_Z = kde.evaluate(grid_pts).reshape(z1.shape[0], z2.shape[0])
        pdf_Z /= np.prod(bws)
    else:
        pdf_Z, _, _ = np.histogram2d(
            samplez[:, 0], samplez[:, 1],
            bins=[z1_edges, z2_edges],
        )
        if smooth_factor != 0:
            sigma = bws * smooth_factor / dz
            pdf_Z = gaussian_filter(pdf_Z, sigma=sigma)
        pdf_Z /= samplez.shape[0] * np.prod(dz)

    Z_inner = np.stack([Z1[z1_inner_slc, :], Z2[z1_inner_slc, :]], axis=-1)
    XY = Z_inner @ L.T
    X, Y = XY[..., 0], XY[..., 1]
    pdf = pdf_Z[z1_inner_slc, :] * mass_multiplier / np.linalg.det(L)

    return (Y.T, X.T, pdf.T) if swap_axes else (X, Y, pdf)
