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
from excee.util import get_long_names, _init_kwargs_dict
from excee.autocorr import autocorr_time_over_time

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def plot_autocorr_evolution(data, n0=100, nn=20, **kwargs):
    ns = np.geomspace(kwargs.get("discard", 0) + n0, data.sizes["draw"], nn)
    ns = ns.astype(int)
    tau = autocorr_time_over_time(data, ns, **kwargs)[0]

    _names = get_long_names(data)
    labels = [
        fr"{name}: {round(t.values[()]) if np.isfinite(t.values) else 'NAN'}"
        for t, name in zip(tau.isel(n=-1, drop=True).values(), _names)
    ]

    fig, ax = plt.subplots()
    ax.loglog(ns, tau.to_array().values.T, ".-", label=labels)
    ax.legend(title=r"$\tau_f$", loc="center left", bbox_to_anchor=(1, 0.5))
    return fig, ax


def plot_trace_2d(data, width=8, height=2, split_at=None, ratio=None,
                  cbar_kwargs=None, subplots_layout="tight", **kwargs):

    n = len(data)
    ncol = 1 if split_at is None else 2
    if ratio is None:
        ratio = 1 if split_at is None else split_at / data.sizes["draw"]

    fig, axes = plt.subplots(
        n, ncol, figsize=(width, n*height),
        sharex="col", sharey=True, squeeze=False,
        width_ratios=None if split_at is None else (ratio, 1),
        layout=subplots_layout,
    )

    cbar_kwargs = _init_kwargs_dict(cbar_kwargs)
    cbar_kwargs.setdefault("aspect", 10)
    if split_at is None:
        cbar_kwargs.setdefault("pad", 0.025)

    cbar_kwargs_left = cbar_kwargs.copy()
    cbar_kwargs_left.setdefault("location", "left")

    cbar_kwargs_right = cbar_kwargs.copy()
    cbar_kwargs_right.setdefault("label", None)
    cbar_kwargs_right.setdefault("pad", ratio * 0.1)

    for row, key in enumerate(data):
        arr = data[key].transpose("chain", "draw")
        if split_at is not None:
            arr.isel(draw=slice(split_at)).plot(
                ax=axes[row, 0],
                cbar_kwargs=cbar_kwargs_left, **kwargs,
            )
            arr.isel(draw=slice(split_at, None)).plot(
                ax=axes[row, 1],
                cbar_kwargs=cbar_kwargs_right, **kwargs,
            )
        else:
            arr.plot(ax=axes[row, 0], cbar_kwargs=cbar_kwargs, **kwargs)

    for ax in axes.flat:
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    if split_at is None:
        wspace = 0
        for ax in axes[:, 0]:
            ax.set_ylabel("chain")
        axes[-1, 0].set_xlabel("draw")
    else:
        wspace = 0.025
        # fig.supxlabel("draw", y=0.125, va="top")
        for ax in axes.flat:
            ax.yaxis.set_ticklabels([])

    if subplots_layout == "tight":
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=wspace)

    return fig, axes
