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


import textwrap
from xarray.plot.utils import _get_units_from_attrs


def _init_kwargs_dict(kwargs):
    return {} if kwargs is None else kwargs.copy()


def ordered_union(lists):
    flat = [x for list in lists for x in list]
    return tuple(dict.fromkeys(flat).keys())


def ordered_intersection(lists):
    intersection = set.intersection(*(set(x) for x in lists))
    union = ordered_union([list(x) for x in lists])
    return [x for x in union if x in intersection]


def union_dicts(dicts):
    from functools import reduce
    from operator import ior
    return reduce(ior, dicts, {})


def dataset_to_dict(data):
    return dict(zip(data.keys(), data.to_array().values))


def grouped_map(self, dim, func, mapper=None, input_to_flat_dict=True,
                progress=None, progress_kwargs=None):
    """
    Return *func* applied to *self* grouped by dimension *dim*, optionally using
    the *map* method of *pool*.
    """

    mapper = mapper or map
    groups = self.groupby(dim)
    iterator = groups._iter_grouped()
    if input_to_flat_dict:
        iterator = (dataset_to_dict(i.squeeze()) for i in iterator)
    if progress:
        from tqdm.auto import tqdm
        progress_kwargs = progress_kwargs or {}
        progress_kwargs.setdefault("total", len(groups))
        iterator = tqdm(iterator, **progress_kwargs)
    applied = mapper(func, iterator)

    return groups._combine(applied)


def write_pickle_to_h5(file, obj, name):
    import pickle
    pickled_obj = pickle.dumps(obj)

    from h5py import string_dtype
    dt = string_dtype(length=len(pickled_obj))

    file.create_dataset(name, data=pickled_obj, dtype=dt)


def read_pickle_from_h5(dset):
    import pickle
    return pickle.loads(dset[()])


def label_from_attrs(da, extra="", wrap=False):
    if da is None:
        return ""

    name: str = "{}"
    if "long_name" in da.attrs:
        name = name.format(da.attrs["long_name"])
    elif "standard_name" in da.attrs:
        name = name.format(da.attrs["standard_name"])
    elif da.name is not None:
        name = name.format(da.name)
    else:
        name = ""

    units = _get_units_from_attrs(da)

    label = name + extra + units

    if wrap:
        # Treat `name` differently if it's a latex sequence
        if name.startswith("$") and (name.count("$") % 2 == 0):
            return "$\n$".join(
                textwrap.wrap(label, 60, break_long_words=False)
            )
        else:
            return "\n".join(textwrap.wrap(label, 30))
    else:
        return label


def get_long_names(data):
    return [label_from_attrs(da) for da in data.values()]
    # return [da.attrs.get("long_name", key) for key, da in data.items()]


def render_prior(par):
    import excee.sampling as xcs
    if isinstance(par, xcs.GaussianSampleParameter):
        name = par.latex
        prior = rf"\mathcal{{N}}({par.mean}, {par.std})"
    elif isinstance(par, xcs.LogUniformSampleParameter):
        name = f"\\ln {par.latex}"
        prior = rf"\mathcal{{U}}({par.low}, {par.high})"
    elif isinstance(par, xcs.ExpUniformSampleParameter):
        name = f"\\exp {par.latex}"
        prior = rf"\mathcal{{U}}({par.low}, {par.high})"
    elif isinstance(par, xcs.PowUniformSampleParameter):
        name = f"{par.latex}^{par.power}"
        prior = rf"\mathcal{{U}}({par.low**par.power}, {par.high**par.power})"
    elif type(par) is xcs.SampleParameter:
        name = par.latex
        prior = rf"\mathcal{{U}}({par.low}, {par.high})"
    else:
        raise NotImplementedError(f"{type(par)}")
    name = name.replace("$", "")
    return f"${name} \\sim {prior}$"


def print_priors_from_file(path):
    from h5py import File
    with File(path) as f:
        for p in read_pickle_from_h5(f["sample_parameters"]):
            print(render_prior(p))  # noqa: T201


__all__ = [
    "_init_kwargs_dict",
    "get_long_names",
    "write_pickle_to_h5",
    "read_pickle_from_h5",
    "render_prior",
    "print_priors_from_file",
]
