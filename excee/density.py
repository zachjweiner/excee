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


def lcv(data, frac):
    # coefficient of L-variation
    n = data.shape[0]
    k = max(3, int(n * frac))
    sorted_data = np.sort(data, axis=0)
    tails = np.stack([sorted_data[:k+1], sorted_data[:-k-2:-1]], axis=1)
    x = np.abs(tails[1:] - tails[:1])

    # from scipy.stats import lmoment
    # l1, l2 = lmoment(x, order=[1, 2], axis=0, sorted=True)
    # equivalent, but much less overhead:
    l1 = np.mean(x, axis=0)
    weights = np.linspace(-1, 1, k)
    l2 = np.mean((weights * x.T).T, axis=0)
    return l2 / l1


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
    auto_bounds = detect_boundaries(sample, lcv_threshold, lcv_frac).T
    x_bounded, y_bounded = (
        auto if bound == "auto"
        else [False, False] if bound in (None, False) else bound
        for bound, auto in zip(bounds, auto_bounds)
    )
    x_lims, y_lims = get_lims(sample)
    bounds_x = np.where(x_bounded, x_lims, None)
    bounds_y = np.where(y_bounded, y_lims, None)
    logger.info(f"bounds_x = {tuple(bounds_x)}, bounds_y = {tuple(bounds_y)}")

    if any(x_bounded) and any(y_bounded) and _cholesky and smooth_factor != 0:
        logger.warning(
            "Simultaneous x and y boundaries detected. "
            "Skipping Cholesky rotation; smooth with caution.")
        _cholesky = False

    swap_axes = any(y_bounded)
    if swap_axes:
        sample = sample[:, ::-1]
        bounds, bins = bounds_y, bins[::-1]
    else:
        bounds = bounds_x

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
