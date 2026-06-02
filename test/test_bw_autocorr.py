__copyright__ = "Copyright (C) 2026 Zachary J Weiner"

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
from scipy import stats
import xarray as xr
from excee import kde_bandwidth, autocorr_time
import pytest

BW_METHODS = ("scott", "silverman", "isj")


def generate_synthetic_mcmc(dist, n_draws, tau, seed=None):
    rng = np.random.default_rng(seed)
    p_accept = 2 / (tau + 1)
    indep_samples = dist.sample(n_draws, rng=rng)
    run_lengths = rng.geometric(p=p_accept, size=n_draws)
    chain = np.repeat(indep_samples, run_lengths)
    return chain[:n_draws]


def generate_chain(dists, n_chains, n_draws, taus, seed=None):
    rng = np.random.default_rng(seed)
    taus = np.broadcast_to(taus, len(dists))
    return np.stack([
        np.stack([
            generate_synthetic_mcmc(dist, n_draws, tau, seed=rng)
            for _ in range(n_chains)
        ])
        for dist, tau in zip(dists, taus)
    ])


@pytest.fixture(scope="module")
def normal_mcmc_data():
    return generate_chain([stats.Normal()]*3, 8, 10000, [7, 4, 6], seed=8231)


@pytest.fixture(scope="module")
def dummy_ess(normal_mcmc_data):
    n_vars = normal_mcmc_data.shape[0]
    rng = np.random.default_rng(123)
    return rng.uniform(800.0, 1200.0, size=n_vars)


@pytest.fixture
def ref_bw(normal_mcmc_data, dummy_ess, bw, has_chain_axis):
    s = -2 if has_chain_axis else -1
    out_shape = normal_mcmc_data.shape[:s]
    manual_bw = np.zeros(out_shape)

    for idx in np.ndindex(out_shape):
        # to test autocorr fallback, only pass ess when has_chain_axis=True
        kwargs = {"ess": dummy_ess[idx[0]]} if has_chain_axis else {}
        manual_bw[idx] = kde_bandwidth(
            normal_mcmc_data[idx],
            bw=bw,
            has_chain_axis=has_chain_axis,
            **kwargs
        )
    return manual_bw


@pytest.fixture
def ref_tau(normal_mcmc_data, has_chain_axis):
    s = -2 if has_chain_axis else -1
    out_shape = normal_mcmc_data.shape[:s]
    manual_tau = np.zeros(out_shape)

    for idx in np.ndindex(out_shape):
        manual_tau[idx] = autocorr_time(
            normal_mcmc_data[idx],
            has_chain_axis=has_chain_axis,
        )[0]
    return manual_tau


@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_numpy_tau(normal_mcmc_data, has_chain_axis, ref_tau):
    tau_res = autocorr_time(
        normal_mcmc_data,
        has_chain_axis=has_chain_axis,
    )[0]
    assert tau_res.shape == ref_tau.shape
    np.testing.assert_allclose(tau_res, ref_tau, rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_numpy_bw(normal_mcmc_data, dummy_ess, bw, has_chain_axis, ref_bw):
    kwargs = {"ess": dummy_ess} if has_chain_axis else {}
    bw_res = kde_bandwidth(
        normal_mcmc_data,
        bw=bw,
        has_chain_axis=has_chain_axis,
        **kwargs
    )
    assert bw_res.shape == ref_bw.shape
    np.testing.assert_allclose(bw_res, ref_bw, rtol=1e-12)


@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_dataarray_tau(normal_mcmc_data, has_chain_axis, ref_tau):
    da = xr.DataArray(
        normal_mcmc_data,
        dims=["variable", "chain", "draw"],
        coords={"variable": ["a", "b", "c"]}
    )

    tau_res = autocorr_time(da, chain_dim="chain" if has_chain_axis else None)[0]
    assert isinstance(tau_res, xr.DataArray)
    assert tau_res.shape == ref_tau.shape
    np.testing.assert_allclose(tau_res, ref_tau, rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_dataarray_bw(normal_mcmc_data, dummy_ess, bw, has_chain_axis, ref_bw):
    da = xr.DataArray(
        normal_mcmc_data,
        dims=["variable", "chain", "draw"],
        coords={"variable": ["a", "b", "c"]}
    )

    kwargs = {"ess": dummy_ess} if has_chain_axis else {}
    bw_res = kde_bandwidth(
        da, bw=bw,
        chain_dim="chain" if has_chain_axis else None, draw_dim="draw",
        **kwargs
    )

    assert isinstance(bw_res, xr.DataArray)
    expected_dims = ("variable",) if has_chain_axis else ("variable", "chain")
    assert bw_res.dims == expected_dims
    np.testing.assert_allclose(bw_res.values, ref_bw, rtol=1e-12)


@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_dataset_tau(normal_mcmc_data, has_chain_axis, ref_tau):
    ds = xr.Dataset({
        "var_1": (["chain", "draw"], normal_mcmc_data[0]),
        "var_2": (["chain", "draw"], normal_mcmc_data[1]),
    })

    expected_dims = () if has_chain_axis else ("chain",)

    tau_res = autocorr_time(ds, chain_dim="chain" if has_chain_axis else None)[0]

    assert isinstance(tau_res, xr.Dataset)
    assert tuple(tau_res.sizes) == expected_dims
    assert tau_res["var_1"].shape == ref_tau[0].shape
    assert tau_res["var_2"].shape == ref_tau[1].shape
    np.testing.assert_allclose(tau_res["var_1"], ref_tau[0], rtol=1e-12)
    np.testing.assert_allclose(tau_res["var_2"], ref_tau[1], rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_dataset_bw(normal_mcmc_data, bw, has_chain_axis):
    ds = xr.Dataset({
        "var_1": (["chain", "draw"], normal_mcmc_data[0]),
        "var_2": (["chain", "draw"], normal_mcmc_data[1]),
    })

    expected_dims = () if has_chain_axis else ("chain",)

    bw_res = kde_bandwidth(ds, bw=bw, chain_dim="chain" if has_chain_axis else None)
    # can't pass structured ess with dataset input
    ref_bw = kde_bandwidth(normal_mcmc_data, bw=bw, has_chain_axis=has_chain_axis)

    assert isinstance(bw_res, xr.Dataset)
    assert tuple(bw_res.sizes) == expected_dims
    assert bw_res["var_1"].dims == expected_dims
    assert bw_res["var_1"].shape == ref_bw[0].shape
    assert bw_res["var_2"].shape == ref_bw[1].shape
    np.testing.assert_allclose(bw_res["var_1"].values, ref_bw[0], rtol=1e-12)
    np.testing.assert_allclose(bw_res["var_2"].values, ref_bw[1], rtol=1e-12)


def test_missing_chain_dim_tau(normal_mcmc_data):
    da = xr.DataArray(normal_mcmc_data[0, 0], dims=["draw"])

    da_tau = autocorr_time(da)[0]
    np_tau = autocorr_time(normal_mcmc_data[0, 0])[0]

    np.testing.assert_allclose(da_tau.values, np_tau, rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
def test_missing_chain_dim_bw(normal_mcmc_data, bw):
    da = xr.DataArray(normal_mcmc_data[0, 0], dims=["draw"])

    da_bw = kde_bandwidth(da, bw=bw)
    np_bw = kde_bandwidth(normal_mcmc_data[0, 0], bw=bw)

    np.testing.assert_allclose(da_bw.values, np_bw, rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
def test_dimensionality_rescaling(normal_mcmc_data, dummy_ess, bw):
    x = normal_mcmc_data[0]
    ess = dummy_ess[0]

    bw_1d = kde_bandwidth(x, bw=bw, ess=ess, dim=1)
    bw_2d = kde_bandwidth(x, bw=bw, ess=ess, dim=2)

    np.testing.assert_allclose(bw_2d, bw_1d * ess**(1/5-1/6), rtol=1e-12)


@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_explicit_bandwidth(normal_mcmc_data, has_chain_axis):
    bw = 0.5
    res = kde_bandwidth(normal_mcmc_data, bw=bw, has_chain_axis=has_chain_axis)

    s = -2 if has_chain_axis else -1
    expected_shape = normal_mcmc_data.shape[:s]
    assert res.shape == expected_shape
    np.testing.assert_allclose(res, np.full(expected_shape, bw))


@pytest.fixture(scope="module")
def bounded_mcmc_data():
    return generate_chain(
        [stats.truncate(stats.Normal(), a, np.inf) for a in np.arange(0, 4)],
        8, 10000, 5,
        seed=8211,
    )


def test_isj_bounded(bounded_mcmc_data):
    res_scott = kde_bandwidth(bounded_mcmc_data, bw="scott")
    res_isj = kde_bandwidth(bounded_mcmc_data, bw="isj")
    res_isj_bad_ess = kde_bandwidth(
        bounded_mcmc_data, bw="isj", thin=False,
        ess=np.prod(bounded_mcmc_data.shape[-2:]),
    )
    res_isj_bad_bounds = kde_bandwidth(
        bounded_mcmc_data, bw="isj", thin=False,
        bounds=[None, None],
    )

    assert all(res_isj / res_scott > 1/2), res_isj / res_scott
    assert all(res_isj / res_isj_bad_ess > 4), res_isj / res_isj_bad_ess
    assert all(res_isj / res_isj_bad_bounds > 5), res_isj / res_isj_bad_bounds


fit_ns = np.round(np.logspace(5, 6.75, 20)).astype(int)


@pytest.fixture(scope="module")
def random_data_for_fits():
    rng = np.random.default_rng(523)
    return rng.normal(size=max(fit_ns))


@pytest.mark.parametrize("bw", BW_METHODS)
def test_normal(bw, random_data_for_fits):
    kw = {"bounds": (None, None)} if bw == "isj" else {}  # just for speed
    bws = np.array([
        kde_bandwidth(random_data_for_fits[:n], bw=bw, thin=False, ess=n, **kw)
        for n in fit_ns
    ])

    res = stats.linregress(np.log(fit_ns), np.log(bws))
    an_intercepts = {
        "scott": 1.06,
        "silverman": 0.9,
        "isj": (4/3)**(1/5),
    }

    power_err = np.abs(res.slope / (-1/5) - 1)
    intercept_err = np.abs(np.exp(res.intercept) / an_intercepts[bw] - 1)
    rtol = {
        "scott": 5e-3,
        "silverman": 1e-2,
        "isj": 5e-2,
    }
    assert power_err < rtol[bw], (res.slope, power_err)
    assert intercept_err < rtol[bw], (np.exp(res.intercept), intercept_err)


@pytest.mark.parametrize(
    ("n_chain", "n_draw"), [(1, 10**6), (4, 10**5), (100, 10**4)]
)
def test_autocorr(n_chain, n_draw):
    rng = np.random.default_rng(4231)
    inpt_taus = np.array([3, 7, 10])
    data = generate_chain([stats.Normal()]*3, n_chain, n_draw, inpt_taus, seed=rng)
    taus = autocorr_time(data)[0]
    np.testing.assert_allclose(taus, inpt_taus, rtol=5e-2)
