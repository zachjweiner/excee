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
from scipy.stats import rankdata, norm
import xarray as xr
import arviz_stats as az

logger = logging.getLogger(__name__)


def _rank(x):
    return rankdata(x).reshape(x.shape)


def rank(x, dims=("chain", "draw")):
    core_dims = list(dims)

    return xr.apply_ufunc(
        _rank,
        x,
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )


def rank_normalize(x, dims=("chain", "draw")):
    core_dims = list(dims)
    ranks = rank(x, dims=core_dims)

    def _normalize(x):
        return norm.ppf((x - 3/8) / (np.size(x) + 1/4))

    return xr.apply_ufunc(
        _normalize,
        ranks,
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        vectorize=True,
    )


def _integrated_time(x, has_chain_axis, *, window_method="geyer", sokal_c=5):
    if not has_chain_axis:
        x = np.expand_dims(x, axis=-2)

    n_w, n_t = x.shape[-2:]
    if n_t <= 1:
        nan = np.full(x.shape[:-2], np.nan)
        return nan, nan

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


def rank_normalized_autocorr_time(x, chain_dim="chain", draw_dim="draw", **kwargs):
    return autocorr_time(
        rank_normalize(x, dims=[chain_dim, draw_dim]),
        chain_dim=chain_dim, draw_dim=draw_dim, **kwargs
    )


def dwell_autocorr_time(x, draw_dim="draw"):
    rejected = x.diff(dim=draw_dim) == 0

    def _tau_dwell(rej_1d):
        N = len(rej_1d) + 1
        # indices of accepted steps
        acc_idx = np.where(~rej_1d)[0]
        # length of each block is difference of acceptance indices
        # pad with -1 and N-1 to capture the edges
        D = np.diff(np.concatenate(([-1], acc_idx, [N-1])))
        return np.sum(D**2) / N

    tau_ideal = xr.apply_ufunc(
        _tau_dwell,
        rejected,
        input_core_dims=[[draw_dim]],
        output_core_dims=[[]],
        vectorize=True,
    )

    return tau_ideal


def autocorr_time_profile(x, n_splits=20, chain_dim="chain", draw_dim="draw"):
    edges = np.linspace(0, 1, n_splits+1)
    centers = (edges[1:] + edges[:-1]) / 2
    qs = x.quantile(edges, dim=[chain_dim, draw_dim])

    def indicate(z, i):
        left = qs.isel(quantile=i)
        right = qs.isel(quantile=i+1)
        mask = (z >= left) & ((z < right) if i < n_splits - 1 else (z <= right))
        return mask.astype(float)

    taus = [
        1 / az.ess(indicate(x, i), method="mean", relative=True)
        for i in range(n_splits)
    ]
    taus = xr.concat(taus, dim="quantile_center")
    return taus.assign_coords(quantile_center=centers)


def rejection_profile(x, n_splits=20, chain_dim="chain", draw_dim="draw"):
    rejected = x.diff(dim=draw_dim) == 0
    x_t = x.shift({draw_dim: 1})
    edges = np.linspace(0, 1, n_splits+1)
    centers = (edges[1:] + edges[:-1]) / 2
    qs = x_t.quantile(edges, dim=[chain_dim, draw_dim])

    def get_mask(z, i):
        left = qs.isel(quantile=i)
        right = qs.isel(quantile=i+1)
        mask = (z >= left) & ((z < right) if i < n_splits - 1 else (z <= right))
        return mask

    rates = [
        rejected.where(get_mask(x_t, i)).mean(dim=[chain_dim, draw_dim])
        for i in range(n_splits)
    ]
    rates = xr.concat(rates, dim="quantile_center")
    return rates.assign_coords(quantile_center=centers)


def acceptance_fraction(x, *, average_chains=False, draw_dim="draw"):
    accepted = x.diff(dim=draw_dim) != 0
    dims = ["chain", draw_dim] if average_chains else [draw_dim]
    return accepted.mean(dim=dims)


def rms_jump(x, *, average_chains=False, draw_dim="draw"):
    diff_sq = x.diff(dim=draw_dim)**2
    dims = ["chain", draw_dim] if average_chains else [draw_dim]
    return np.sqrt(diff_sq.mean(dim=dims) / x.var(dim=dims))


def ecdf(x, dims="draw"):
    core_dims = [dims] if isinstance(dims, str) else list(dims)
    x_sorted = xr.apply_ufunc(
        np.sort,
        x,
        input_core_dims=[core_dims],
        output_core_dims=[["ecdf_prob"]],
        vectorize=True,
        kwargs={"axis": None},
    )
    n_obs = x_sorted.sizes["ecdf_prob"]
    return x_sorted.assign_coords(ecdf_prob=np.linspace(1/n_obs, 1, n_obs))


def ranked_ecdf(x, ecdf_dims="draw", rank_dims=("chain", "draw")):
    rank_dims = [rank_dims] if isinstance(rank_dims, str) else list(rank_dims)
    ranks = rank(x, dims=rank_dims)
    N = np.prod([x.sizes[d] for d in rank_dims])
    uniform_ranks = ranks / N
    return ecdf(uniform_ranks, dims=ecdf_dims)


def _hdi_sample(x, prob):
    x_sorted = np.sort(x, axis=None)
    n = x_sorted.size
    idx_interval = int(np.floor(prob * n))
    widths = x_sorted[idx_interval:] - x_sorted[:-idx_interval]
    min_idx = np.argmin(widths)
    return x_sorted[[min_idx, min_idx + idx_interval]]


def _hdi_kde(x, pdf, prob):
    sort_idx = np.argsort(-pdf)
    sorted_cdf = np.cumsum(pdf[sort_idx] * np.gradient(x)[sort_idx])
    threshold = np.interp(prob, sorted_cdf / sorted_cdf[-1], pdf[sort_idx])

    thresh_dist = np.pad(pdf - threshold, 1, constant_values=-1)
    x_pad = np.pad(x, 1, mode="edge")
    idx = np.where(np.diff(thresh_dist >= 0))[0]

    dx = x_pad[idx+1] - x_pad[idx]
    df = thresh_dist[idx+1] - thresh_dist[idx]
    roots = x_pad[idx] - thresh_dist[idx] * (dx / df)
    return roots.reshape(-1, 2).squeeze()


def _hdi_kde_from_sample(x, prob):
    from excee.density import compute_1d_density
    x, pdf = compute_1d_density(x, 4096, 1)
    return _hdi_kde(x, pdf, prob)


def hdi(x, prob, *, method="kde", dims=("chain", "draw")):
    func = _hdi_kde_from_sample if method == "kde" else _hdi_sample
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        core_dims = [dims] if isinstance(dims, str) else list(dims)
        return xr.apply_ufunc(
            func,
            x,
            input_core_dims=[core_dims],
            output_core_dims=[["side"]],
            vectorize=True,
            kwargs={"prob": prob},
        )
    else:
        shape = np.shape(x)[:-1]
        hdis = np.array([func(x_i, prob) for x_i in x.reshape(-1, x.shape[-1])])
        return hdis.reshape((*shape, 2))


def _eti(x, prob, method="inverted_cdf", **kwargs):
    quantiles = np.array([(1 - prob)/2, (1 + prob)/2])
    return np.quantile(x, quantiles, method=method, **kwargs)


def eti(x, prob, dims=("chain", "draw"), **kwargs):
    if isinstance(x, (xr.DataArray, xr.Dataset)):
        core_dims = [dims] if isinstance(dims, str) else list(dims)
        return xr.apply_ufunc(
            _eti,
            x,
            input_core_dims=[core_dims],
            output_core_dims=[["side"]],
            vectorize=False,
            kwargs={"prob": prob} | kwargs,
        )
    else:
        return _eti(x, prob, **kwargs)


@np.vectorize(signature="(n),(m),()->(),(),()")
def _eff_gaussian_tension(x, y, quiet=False):
    x = np.ravel(x)
    y = np.ravel(y)
    n, m = x.size, y.size

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    U = np.searchsorted(x_sorted, y_sorted, side="left") / n
    V = (m - np.searchsorted(y_sorted, x_sorted, side="right")) / m
    p = np.mean(U)

    # DeLong's exact empirical variance of the U-statistic
    var_U = np.var(U, ddof=1) if m > 1 else 0
    var_V = np.var(V, ddof=1) if n > 1 else 0
    var_p = var_U / m + var_V / n
    se_p = np.sqrt(var_p)

    ts = norm.ppf(np.clip(p + np.arange(-1, 2) * se_p, 0, 1))
    err_m, err_p = np.diff(ts)
    t_est = ts[1]

    n_eff = min(n, m)
    p_tail = min(p, 1 - p)
    min_ideal = 1 / p_tail
    expected_crossings = n_eff * p_tail
    min_marginal = np.exp(t_est**2 / 2)

    if n_eff < min_marginal and not quiet:
        import warnings
        warnings.warn(
            f"(n_x, n_y) = ({n}, {m}) is an insufficient sample size to robustly"
            f" quantify tension of estimated size {abs(t_est):.2f} sigma."
            f" Only ~{expected_crossings:.2g} samples are expected to span across"
            f" the distributions, so the estimate is likely biased and"
            f" underreporting its uncertainty."
            f" At least ~{min_marginal:.2e} samples are required for a marginal"
            f"estimate and ~{min_ideal:.2e} for strict theoretical accuracy.",
            category=UserWarning,
            stacklevel=2,
        )

    return t_est, err_m, err_p


def eff_gaussian_tension(x, y, *, quiet=False, sample_dims=("chain", "draw")):
    sample_dims = list(sample_dims)
    if any(
        isinstance(arg, (xr.DataArray, xr.Dataset, xr.DataTree))
        for arg in (x, y)
    ):
        def regularize(z):
            return z if "sample" in z.dims else z.stack(sample=sample_dims)
        return xr.apply_ufunc(
            _eff_gaussian_tension,
            regularize(x), regularize(y), quiet,
            input_core_dims=[["sample"], ["sample"], []],
            output_core_dims=[[], [], []],
            exclude_dims={"sample"},
            dataset_join="inner",
        )
    else:
        return _eff_gaussian_tension(x, y)
