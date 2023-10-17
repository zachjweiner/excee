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


def ordered_union(lists):
    return tuple(dict.fromkeys(sum(lists, [])).keys())


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


def grouped_map(self, dim, func, mapper=None):
    """
    Return *func* applied to *self* grouped by dimension *dim*, optionally using
    the *map* method of *pool*.
    """

    mapper = mapper or map
    groups = self.groupby(dim)
    applied = mapper(func, groups._iter_grouped())

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


__all__ = [
    "write_pickle_to_h5",
    "read_pickle_from_h5",
]
