__copyright__ = "Copyright (C) 2023 Zachary J Weiner"

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
import excee as pc
from emcee.backends import TempHDFBackend
import pytest


def _fun(pars, **kwargs):
    return - pars["a"]**2 / 2


def _fun_blobs(pars, **kwargs):
    return {"log_prob": pars["a"]}, {"x": 1}


pars = [pc.SampleParameter("a", -10, 10, "a")]


def test_backend_handling():
    pars_2 = [pc.SampleParameter("a", 0, 10, "a")]

    sampler = pc.LikelihoodSampler(pars, _fun)

    with TempHDFBackend() as backend:
        sampler(10, 10, backend=backend, progress=False)
        sampler(10, 10, backend=backend, progress=False)

        with pytest.raises(ValueError):
            sampler(20, 10, backend=backend, progress=False)

        sampler_2 = pc.LikelihoodSampler(pars_2, _fun)

        with pytest.raises(RuntimeError):
            sampler_2(10, 10, backend=backend, progress=False)

        sampler_3 = pc.LikelihoodSampler(pars, _fun, kwargs={"c": 1})

        with pytest.raises(RuntimeError):
            sampler_3(10, 10, backend=backend, progress=False)

        sampler_4 = pc.LikelihoodSampler(pars, _fun_blobs)

        with pytest.raises(RuntimeError):
            sampler_4(10, 10, backend=backend, progress=False)


def test_resume(nwalkers=10, nsteps=20, seed=52380):
    # https://github.com/dfm/emcee/issues/313
    # https://github.com/dfm/emcee/pull/376

    np.random.seed(seed)
    sampler = pc.LikelihoodSampler(pars, _fun)

    res_no_resume = sampler(nwalkers, nsteps, progress=False)

    np.random.seed(seed)
    sampler = pc.LikelihoodSampler(pars, _fun)

    with TempHDFBackend() as backend:
        sampler(nwalkers, nsteps//2, backend=backend, progress=False)
        res_resume = sampler(nwalkers, nsteps//2, backend=backend, progress=False)

        assert np.all(res_no_resume.sampler.chain == res_resume.sampler.chain)


if __name__ == "__main__":
    test_resume()
    test_backend_handling()
