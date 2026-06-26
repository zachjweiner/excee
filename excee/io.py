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


from itertools import count
from pathlib import Path
import numpy as np
import xarray as xr
import h5py

vlen_str_dt = h5py.string_dtype(encoding="utf-8")


def to_dataarray(ds, dim="variable"):
    da = ds.to_dataarray(dim)
    attrs = list({key for _da in ds.variables.values() for key in _da.attrs})
    for key in attrs:
        vals = np.array([ds[k].attrs.get(key, np.nan) for k in da[dim].values])
        if vals.dtype.kind in ("U", "S", "O") and isinstance(vals.flat[0], str):
            vals = vals.astype(vlen_str_dt)
        da.attrs["__"+key] = vals

    return da


def to_dataset(da, dim="variable"):
    ds = da.to_dataset(dim)
    restore_attrs = {
        key.replace("__", ""): val
        for key, val in da.attrs.items() if key.startswith("__")
    }
    for key in restore_attrs:
        da.attrs.pop("__"+key)
    for i, key in enumerate(ds):
        _attrs = {
            attr: val for attr, vals in restore_attrs.items()
            if (
                (val := vals[i]) != "nan"
                and not (isinstance(val, float) and np.isnan(val))
            )
        }
        ds[key].attrs.update(**_attrs)

    return ds


def restore_dsets(dt, vkey="variable"):
    data = {}
    for path, node in dt.subtree_with_keys:
        for vname, da in node.data_vars.items():
            data[f"{path}/{vname}"] = (
                to_dataset(da, dim=vkey) if vkey in da.sizes else da
            )

    return xr.DataTree.from_dict(data)


def _compress_da(da):
    reduce_dims = [d for d in da.dims if d not in ("chain", "draw")]
    # .shift() inserts nan at draw=0, so changed(draw=0) is True
    changed = (da != da.shift(draw=1)).any(dim=reduce_dims)
    changed.loc[{"draw": da.draw[-1]}] = True

    mask = changed.stack(sample=["chain", "draw"])
    da = da.stack(sample=["chain", "draw"])
    da = da.isel(sample=mask)
    da = da.reset_index("sample")
    da = da.assign_coords(
        chain=("sample", da.chain.values),
        draw=("sample", da.draw.values)
    )

    return da


def compress(data):
    def _compress(da):
        return _compress_da(da) if {"chain", "draw"} <= set(da.sizes) else da

    if isinstance(data, xr.DataTree):
        return data.map_over_datasets(lambda node: node.map(_compress))
    elif isinstance(data, xr.Dataset):
        return data.map(_compress)
    else:
        return _compress(data)


def _decompress_da(da):
    chain = da.chain.values
    draw = da.draw.values

    same_chain = np.diff(chain, append=chain[-1] + 1) == 0
    repeats = np.where(
        same_chain,
        np.diff(draw, append=0),
        draw.max() + 1 - draw
    )

    da_expanded = da.isel(sample=np.repeat(np.arange(draw.size), repeats))

    chain = np.unique(chain)
    draw = np.arange(draw.min(), draw.max() + 1)

    sample_axis = da_expanded.dims.index("sample")
    new_dims = list(da_expanded.dims)
    new_dims[sample_axis:sample_axis+1] = ["chain", "draw"]
    new_shape = list(da_expanded.shape)
    new_shape[sample_axis:sample_axis+1] = [chain.size, draw.size]
    new_coords = da_expanded.coords | {"chain": chain, "draw": draw}

    return xr.DataArray(
        da_expanded.values.reshape(new_shape),
        dims=new_dims,
        coords=new_coords,
        attrs=da_expanded.attrs,
        name=da_expanded.name
    )


def decompress(data):
    def _decompress(da):
        return _decompress_da(da) if "sample" in da.sizes else da

    if isinstance(data, xr.DataTree):
        return data.map_over_datasets(lambda node: node.map(_decompress))
    elif isinstance(data, xr.Dataset):
        return data.map(_decompress)
    else:
        return _decompress(data)


def extract_posterior(dt):
    dt = dt.match("*/data") or dt.match("data")  # for compat with a single leaf
    return xr.DataTree.from_dict({
        node.parent.path: node.dataset
        for node in dt.subtree
        if node.has_data
    })


def load_result_tree(path, engine="h5netcdf", posterior_only=True, groups=None,
                     **kwargs):
    if groups is None:
        dt = xr.load_datatree(path, engine=engine, **kwargs)
    else:
        dt = xr.DataTree()
        for group in groups:
            dt[group] = xr.load_datatree(path, engine=engine, group=group, **kwargs)
    dt = decompress(dt)
    dt = restore_dsets(dt)
    if posterior_only:
        dt = extract_posterior(dt)
    return dt


def construct_dt(data, best_fit=None, fixed_parameters=None,
                 compressed=False, vkey=None, encode_attrs=False):
    if vkey:
        data = to_dataarray(data, vkey)
    if compressed:
        data = compress(data)
    dt = xr.DataTree.from_dict({"data": data})

    if best_fit is not None:
        if vkey:
            best_fit = to_dataarray(best_fit, vkey)
        dt["best_fit"] = best_fit

    if fixed_parameters is not None:
        if encode_attrs:
            fixed_parameters = {
                k: v if v is not None else "None"
                for k, v in fixed_parameters.items()
            }
        dt.attrs.update(fixed_parameters)

    return dt


def construct_dt_for_storage(data, best_fit=None, fixed_parameters=None,
                             compressed=True, vkey="variable", encode_attrs=True):
    return construct_dt(
        data, best_fit=best_fit, fixed_parameters=fixed_parameters,
        compressed=compressed, vkey=vkey, encode_attrs=encode_attrs,
    )


def deconstruct_dt(dt, vkey="variable"):
    data = dt["data"]
    data = decompress(data)
    if isinstance(data, xr.DataArray):
        data = to_dataset(data, dim=vkey)
    elif isinstance(data, xr.DataTree):
        data = data.to_dataset()

    best_fit = dt.get("best_fit", None)
    if isinstance(best_fit, xr.DataArray):
        best_fit = to_dataset(best_fit, dim=vkey)
    elif isinstance(best_fit, xr.DataTree):
        best_fit = best_fit.to_dataset()

    fixed_parameters = {
        key: None if val == "None" else val
        for key, val in dt.attrs.items()
    }

    return data, best_fit, fixed_parameters


def load_emcee(backend, sample_parameters, fixed_parameters, log_prob_names,
               blob_names, var_name_map, best_fit=None):
    var_names = [par.name for par in sample_parameters]

    _sample_map = {par.name: par.latex for par in sample_parameters}
    var_name_map = _sample_map | var_name_map
    _blob_names = log_prob_names + blob_names

    from excee.sampling import sample_pars_to_par_names
    # FIXME: the below
    try:
        slices = sample_pars_to_par_names(sample_parameters).values()
    except AttributeError:
        slices = np.arange(len(sample_parameters))

    chain = backend.get_chain().transpose(2, 1, 0)
    coords = {
        "chain": np.arange(chain.shape[1]),
        "draw": np.arange(chain.shape[2]),
    }

    dim_count = count()

    def get_dims(ary):
        if ary.ndim == 3:
            pre_dims = (f"dim_{next(dim_count)}",)
        elif ary.ndim == 2:
            pre_dims = ()
        else:
            raise NotImplementedError(f"{ary.ndims=}")

        return (*pre_dims, "chain", "draw")

    chain = {
        var_name: (get_dims(chain[idx]), chain[idx])
        for idx, var_name in zip(slices, var_names)
    }

    if (blobs := backend.get_blobs()) is not None:
        if np.ndim(blobs) == 2:
            blobs = blobs[..., None]
        blobs = blobs.transpose(2, 1, 0)
        blobs = {
            var_name: (("chain", "draw"), blobs[idx])
            for idx, var_name in enumerate(_blob_names)
        }
    else:
        blobs = {}

    blobs["log_prob"] = ("chain", "draw"), backend.get_log_prob().T
    data = xr.Dataset(chain | blobs, coords=coords)

    for key in var_names:
        data[key].attrs["kind"] = "sampled"
    for key in (*log_prob_names, "log_prob"):
        data[key].attrs["kind"] = "log_prob"
    for key in blob_names:
        data[key].attrs["kind"] = "derived"

    for key, val in data.items():
        val.attrs["long_name"] = var_name_map.get(key, key)

    return construct_dt(data, best_fit=best_fit, fixed_parameters=fixed_parameters)


def load_emcee_hdf(backend):
    if isinstance(backend, str | Path):
        from emcee.backends import HDFBackend
        backend = HDFBackend(backend, read_only=True)

    from excee.util import read_pickle_from_h5
    with backend.open("r") as f:
        sample_parameters = read_pickle_from_h5(f["sample_parameters"])
        fixed_parameters = read_pickle_from_h5(f["fixed_parameters"])
        log_prob_names = tuple(f.attrs["log_prob_names"])
        blob_names = tuple(f.attrs["blob_names"])
        var_name_map = read_pickle_from_h5(f["var_name_map"])

    try:
        best_fit = xr.load_dataset(
            backend.filename, engine="h5netcdf", group="best_fit")
    except (OSError, AttributeError):
        best_fit = None

    return load_emcee(
        backend, sample_parameters, fixed_parameters, log_prob_names,
        blob_names, var_name_map, best_fit,
    )


def load_cobaya(path, run_key, repeat=True, truncate=True):
    from excee.cobaya_interop import get_cobaya_data
    data, fixed_parameters = get_cobaya_data(
        path, run_key, repeat=repeat, truncate=truncate
    )
    return construct_dt(data, fixed_parameters=fixed_parameters)


def load_montepython(path, repeat=True, truncate=True):
    from excee.mp_interop import get_montepython_data
    data = get_montepython_data(
        path, repeat=repeat, truncate=truncate
    )
    return construct_dt(data)


__all__ = [
    "to_dataarray",
    "to_dataset",
    "restore_dsets",
    "compress",
    "decompress",
    "extract_posterior",
    "construct_dt",
    "construct_dt_for_storage",
    "deconstruct_dt",
    "load_result_tree",
    "load_emcee",
    "load_emcee_hdf",
    "load_cobaya",
    "load_montepython",
]
