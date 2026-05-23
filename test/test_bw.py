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
import xarray as xr
from excee import kde_bandwidth
import pytest

BW_METHODS = ("scott", "silverman", "isj", "robust_isj")


@pytest.fixture
def random_data():
    rng = np.random.default_rng(42)
    return rng.normal(loc=0, scale=1, size=(3, 4, 500))


@pytest.fixture
def dummy_ess(random_data):
    n_vars = random_data.shape[0]
    rng = np.random.default_rng(123)
    return rng.uniform(800.0, 1200.0, size=n_vars)


@pytest.fixture
def ref_bw(random_data, dummy_ess, bw, has_chain_axis):
    s = -2 if has_chain_axis else -1
    out_shape = random_data.shape[:s]
    manual_bw = np.zeros(out_shape)

    for idx in np.ndindex(out_shape):
        # to test autocorr fallback, only pass ess when has_chain_axis=True
        kwargs = {"ess": dummy_ess[idx[0]]} if has_chain_axis else {}
        manual_bw[idx] = kde_bandwidth(
            random_data[idx],
            bw=bw,
            has_chain_axis=has_chain_axis,
            **kwargs
        )
    return manual_bw


@pytest.mark.parametrize("bw", BW_METHODS)
@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_numpy_vectorization(random_data, dummy_ess, bw, has_chain_axis, ref_bw):
    kwargs = {"ess": dummy_ess} if has_chain_axis else {}

    bw_res = kde_bandwidth(
        random_data,
        bw=bw,
        has_chain_axis=has_chain_axis,
        **kwargs
    )
    assert bw_res.shape == ref_bw.shape
    np.testing.assert_allclose(bw_res, ref_bw, rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_dataarray(random_data, dummy_ess, bw, has_chain_axis, ref_bw):
    da = xr.DataArray(
        random_data,
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


@pytest.mark.parametrize("bw", BW_METHODS)
@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_dataset(random_data, bw, has_chain_axis):
    ds = xr.Dataset({
        "var_1": (["chain", "draw"], random_data[0]),
        "var_2": (["chain", "draw"], random_data[1]),
    })

    chain_dim = "chain" if has_chain_axis else None
    bw_res = kde_bandwidth(ds, bw=bw, chain_dim=chain_dim, draw_dim="draw")

    assert isinstance(bw_res, xr.Dataset)

    expected_dims = () if has_chain_axis else ("chain",)
    assert bw_res["var_1"].dims == expected_dims

    # can't pass structured ess with dataset input
    ref_bw = kde_bandwidth(random_data, bw=bw, has_chain_axis=has_chain_axis)

    np.testing.assert_allclose(bw_res["var_1"].values, ref_bw[0], rtol=1e-12)
    np.testing.assert_allclose(bw_res["var_2"].values, ref_bw[1], rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
def test_missing_chain_dim(random_data, bw):
    da = xr.DataArray(random_data[0, 0], dims=["draw"])

    da_bw = kde_bandwidth(da, bw=bw, chain_dim="chain", draw_dim="draw")
    np_bw = kde_bandwidth(random_data[0, 0], bw=bw, has_chain_axis=False)

    np.testing.assert_allclose(da_bw.values, np_bw, rtol=1e-12)


@pytest.mark.parametrize("bw", BW_METHODS)
def test_dimensionality_rescaling(random_data, dummy_ess, bw):
    x = random_data[0]
    ess = dummy_ess[0]

    bw_1d = kde_bandwidth(x, bw=bw, ess=ess, dim=1)
    bw_2d = kde_bandwidth(x, bw=bw, ess=ess, dim=2)

    np.testing.assert_allclose(bw_2d, bw_1d * ess**(1/5-1/6), rtol=1e-12)


@pytest.mark.parametrize("has_chain_axis", [True, False])
def test_explicit_bandwidth(random_data, has_chain_axis):
    explicit_bw = 0.5
    res = kde_bandwidth(random_data, bw=explicit_bw, has_chain_axis=has_chain_axis)

    expected_shape = (3,) if has_chain_axis else (3, 4)
    assert res.shape == expected_shape
    np.testing.assert_allclose(res, np.full(expected_shape, 0.5))
