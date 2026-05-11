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
                       pad_nstd=4, axes_scale="linear", _cholesky=True):
    if weights is not None:
        raise NotImplementedError("weights")

    sample = np.asarray(sample)
    bins = np.asarray(bins) * np.ones(2, dtype=int)
    smooth_factor = 1 if smooth_factor is None else smooth_factor
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

    L = np.linalg.cholesky(np.cov(sample.T)) if _cholesky else np.eye(2)
    interior_samplez = sample @ np.linalg.inv(L).T

    b_Z = [b / L[0, 0] if b is not None else None for b in bounds]
    mirrors = [
        np.array([2 * b, 0]) + np.array([-1, 1]) * interior_samplez
        for b in b_Z if b is not None
    ]
    mass_multiplier = 1 + len(mirrors)
    samplez = np.vstack([interior_samplez, *mirrors])

    def get_grid(x_min, x_max, n):
        centers, dx = np.linspace(x_min, x_max, n, retstep=True)
        edges = np.concatenate([centers - dx/2, [centers[-1] + dx/2]])
        return edges, centers, dx

    pad = pad_nstd * np.std(interior_samplez, axis=0)

    z1_min = b_Z[0] if b_Z[0] is not None else np.min(samplez[:, 0]) - pad[0]
    z1_max = b_Z[1] if b_Z[1] is not None else np.max(samplez[:, 0]) + pad[0]
    z1_width = z1_max - z1_min
    z1_edges, z1, dz1 = get_grid(
        z1_min - (z1_width if b_Z[0] is not None else 0),
        z1_max + (z1_width if b_Z[1] is not None else 0),
        bins[0] + 1 + bins[0] * len(mirrors)
    )
    i0 = 0 if b_Z[0] is None else bins[0]
    z1_interior_slc = slice(i0, i0 + bins[0] + 1)

    z2_edges, z2, dz2 = get_grid(
        np.min(samplez[:, 1]) - pad[1],
        np.max(samplez[:, 1]) + pad[1],
        bins[1] + 1,
    )

    Z1, Z2 = np.meshgrid(z1, z2, indexing="ij")

    hs = get_bw(samplez, bw="scott") * samplez.shape[0]**(1/5 - 1/6) * smooth_factor

    if use_kdepy:
        if smooth_factor == 0.:
            raise ValueError("KDEpy without smoothing")
        from KDEpy import FFTKDE
        kde = FFTKDE(bw=1).fit(samplez / hs)
        grid_pts = np.stack([Z1, Z2], axis=-1).reshape(-1, 2) / hs
        pdf_Z = kde.evaluate(grid_pts).reshape(z1.shape[0], z2.shape[0])
        pdf_Z /= np.prod(hs)
    else:
        pdf_Z, _, _ = np.histogram2d(
            samplez[:, 0], samplez[:, 1],
            bins=[z1_edges, z2_edges],
        )
        if smooth_factor != 0:
            sigma = hs / np.array([dz1, dz2])
            pdf_Z = gaussian_filter(pdf_Z, sigma=sigma)
        pdf_Z /= samplez.shape[0] * dz1 * dz2

    Z_interior = np.stack([Z1[z1_interior_slc, :], Z2[z1_interior_slc, :]], axis=-1)
    XY = Z_interior @ L.T
    X, Y = XY[..., 0], XY[..., 1]
    pdf = pdf_Z[z1_interior_slc, :] * mass_multiplier / np.linalg.det(L)

    return (Y.T, X.T, pdf.T) if swap_axes else (X, Y, pdf)
