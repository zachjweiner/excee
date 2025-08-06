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


__all__ = [
    "write_pickle_to_h5",
    "read_pickle_from_h5",
]
