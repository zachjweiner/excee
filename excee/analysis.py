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


from functools import partial
import re
import numpy as np
import xarray as xr
from excee.util import grouped_map, label_from_attrs
from excee.stats import autocorr_time


def discard_and_thin(data, discard_per_autocorr, thin_per_autocorr, *,
                     autocorr_discard=100):
    if isinstance(data, xr.DataTree):
        return data.map_over_datasets(
            lambda node: discard_and_thin(
                node, discard_per_autocorr, thin_per_autocorr
            ) if "draw" in node.sizes else node
        )

    tau = np.nanmin([
        _tau if (_tau := da.attrs.get("autocorr_time")) is not None
        else autocorr_time(da, discard=autocorr_discard)[0].values
        for da in data.values()
    ])
    thin = max(1, round(thin_per_autocorr * tau))
    discard = round(discard_per_autocorr * tau)
    return data.sel(draw=slice(discard, None, thin))


def get_random_sample(data, size, rng, reindex=False):
    if {"chain", "draw"} <= set(data.dims):
        data = data.stack(sample=["chain", "draw"])
    rng = np.random.default_rng(None if rng is True else rng)
    slc = rng.choice(len(data["sample"]), size=size, replace=False)
    data = data.isel({"sample": slc})
    if reindex:
        data = data.assign_coords(sample=np.arange(data.sample.size))

    return data


def project_sample(sample, func, *, filter_kw=None, exclude_attrs=False,
                   pool=None, progress=True, progress_kwargs=None, **kwargs):
    if filter_kw is None:
        filter_kw = {"kind": "sampled"}
        sample = sample.filter_by_attrs(**filter_kw)

    kw = kwargs if exclude_attrs else sample.attrs | kwargs
    func = partial(func, **kw)

    mapper = pool.map if pool else map

    projection = grouped_map(
        sample, "sample", func, mapper=mapper,
        progress=progress, progress_kwargs=progress_kwargs,
    )
    return sample, projection


def split_vector_vars(data, keep_dims=("chain", "draw", "sample")):
    if set(data.dims) <= set(keep_dims):
        return data

    def split_one(da):
        dims_to_split = set(da.dims) - set(keep_dims)
        if len(dims_to_split) > 1:
            raise NotImplementedError("multi-dimensional splitting")
        elif dims_to_split:
            dim, = dims_to_split
            da[dim] = [f"{da.name}_{i}" for i in range(da[dim].size)]
            dset = da.to_dataset(dim, promote_attrs=True).copy()

            for name, var in dset.items():
                prefix, idx = re.findall("([a-zA-z]+)_([0-9]+)", name)[0]
                if long_name := label_from_attrs(da):
                    prefix = long_name.replace("$", "")
                var.attrs["long_name"] = f"${prefix}_{{{idx}}}$"
                var.attrs["kind"] = dset.attrs["kind"]
        else:
            dset = da.copy()

        return dset

    das = [split_one(da) for da in data.values()]
    return xr.merge(das)


def get_best_sample(ds, key="log_prob"):
    # N.B. *not* necessarily the best/optimal fit!
    idxmax = ds[key].argmax(...)
    return ds[idxmax]


def get_bounds_array(self, clip=0.025):
    ds = self.get_sample(10, 1, split_vectors=True)
    ds = ds.filter_by_attrs(kind="sampled")
    return ds.quantile([clip, 1-clip]).to_array().values


def flatten_chains(data, reindex=False, stacked_dims=("chain", "draw")):
    data = data.stack(sample=stacked_dims)
    if reindex:
        data = data.drop_vars(["sample", "draw", "chain"])
        data = data.assign_coords(sample=np.arange(data.sample.size))
    return data


def expand_sample_to_chain_and_draw(dset):
    n = dset.sizes["sample"]
    dset = dset.drop_vars(["chain", "sample", "draw"], errors="ignore")
    dset = dset.rename_dims({"sample": "draw"})
    dset = dset.assign_coords(draw=np.arange(n))
    dset = dset.expand_dims({"chain": [1]}, axis=0)
    return dset


__all__ = [
    "discard_and_thin",
    "split_vector_vars",
    "get_random_sample",
    "project_sample",
]
