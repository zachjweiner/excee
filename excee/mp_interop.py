__copyright__ = "Copyright (C) 2025 Zachary J Weiner"

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


import re
import numpy as np
import xarray as xr


def parse_param_spec(file):
    content = file.read_text()
    param_spec = r"data.parameters\[['\"](.*?)['\"]\].*\'{}\']$"
    sampled = re.findall(param_spec.format("cosmo"), content, flags=re.MULTILINE)
    sampled += re.findall(param_spec.format("nuisance"), content, flags=re.MULTILINE)
    blobs = re.findall(param_spec.format("derived"), content, flags=re.MULTILINE)

    return sampled, blobs


def chain_to_xr(file, cols, repeat=True):
    dat = np.loadtxt(file).T
    if repeat:
        w = dat[cols.index("weight")].astype(int)
        dat = np.repeat(dat, w, axis=-1)
    n = dat.shape[1]
    chain = dict(zip(cols, dat))
    return xr.Dataset(
        {key: ("draw", val) for key, val in chain.items()},
        {"draw": np.arange(n)},
    )


def get_montepython_data(direc, repeat=True, truncate=True):
    sampled, blobs = parse_param_spec(next(direc.glob("*.param")))
    # NB: weights is used by excee, not weight
    cols = ["weight", "chi2", *sampled, *blobs]

    long_names = {}

    chains = {
        file.stem: chain_to_xr(file, cols, repeat=True)
        for file in direc.glob("*[0-9]*.txt")
    }
    if truncate:
        # select subset of chains that maximize total number of draws
        # useful if some chains terminated early
        sizes = {key: chain.sizes["draw"] for key, chain in chains.items()}
        sizes = dict(sorted(sizes.items(), key=lambda item: item[1]))
        n = len(sizes)
        ndraws = {key: (n - i) * size for i, (key, size) in enumerate(sizes.items())}
        min_size = sizes[max(ndraws, key=ndraws.get)]

        chains = [
            chain for chain in chains.values()
            if chain.sizes["draw"] >= min_size
        ]
    else:
        chains = list(chains.values())

    ds = xr.concat(chains, dim=xr.DataArray(np.arange(len(chains)), dims=("chain",)))
    for key in ds:
        if key in long_names:
            ds[key].attrs["long_name"] = long_names[key]
        ds[key].attrs["kind"] = (
            "sampled" if key in sampled
            else "derived" if key in blobs and not key.startswith("chi2")
            else "log_prob"
        )

    ds["chi2"] *= -1/2
    ds = ds.rename({
        "chi2": "log_prob",
    })

    if truncate:
        x = ds.to_array().values.transpose(2, 1, 0)
        inan = np.argmax(np.isnan(x), axis=0)
        icut = np.min(inan[inan > 0])
        ds = ds.where(ds.draw < icut, drop=True)

    # FIXME: best fit?
    return ds
