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


import re
import numpy as np
import xarray as xr
import excee as xc
import yaml


def parse_prior(name, prior, latex):
    latex = f"${latex}$"
    match prior:
        case {"dist": "norm", "loc": mean, "scale": std}:
            return xc.GaussianSampleParameter(name, mean, std, latex)
        case {"min": low, "max": high}:
            return xc.SampleParameter(name, low, high, latex)
        case _:
            return None


def cobaya_to_params(spec):
    sample_parameters = []
    fixed_parameters = {}
    blobs = {}

    for name, param in spec.items():
        match param:
            case {"value": val}:
                fixed_parameters |= {name: val}
            case {"prior": prior, "latex": latex}:
                sample_parameters.append(parse_prior(name, prior, latex))
            case {"latex": latex}:
                blobs |= {name: latex}
            case val:
                if isinstance(val, float | int):
                    fixed_parameters |= {name: val}
                else:
                    raise ValueError(f"{name} unparsed")

    return sample_parameters, fixed_parameters, blobs


def chain_to_xr(file, repeat=True):
    cols = re.split(r"\s+", file.read_text().splitlines()[0])[1:]
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


def get_cobaya_data(direc, run_key, repeat=True, truncate=True):
    config = yaml.safe_load((direc / f"chains/{run_key}.updated.yaml").read_text())
    sp, fp, blobs = cobaya_to_params(config["params"])
    var_names = {par.name for par in sp}
    long_names = {par.name: par.latex for par in sp} | blobs
    long_names = {k: f"${v.strip('$')}$" for k, v in long_names.items()}

    chains = [
        chain_to_xr(file, repeat=repeat)
        for file in (direc / "chains").glob(f"{run_key}.[0-9]*.txt")
    ]
    ds = xr.concat(chains, dim=xr.DataArray(np.arange(len(chains)), dims=("chain",)))
    for key in ds:
        if key in long_names:
            ds[key].attrs["long_name"] = long_names[key]
        ds[key].attrs["kind"] = (
            "sampled" if key in var_names
            else "derived" if key in blobs
            else "log_prob"
        )

    if truncate:
        x = ds.to_array().values.transpose(2, 1, 0)
        inan = np.argmax(np.isnan(x), axis=0)
        icut = np.min(inan[inan > 0])
        ds = ds.where(ds.draw < icut, drop=True)

    # FIXME: best fit?
    return ds, fp
