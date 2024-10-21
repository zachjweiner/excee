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


import numpy as np
from excee.analysis import _init_kwargs_dict
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from arviz.stats.density_utils import _find_hdi_contours, _fast_kde_2d


def get_levels(sigmas):
    return 1.0 - np.exp(-0.5 * sigmas**2)


def plot_density_levels(ax, X, Y, pdf, color, *, levels=None,
                        fill_contours=True, shade_background=True,
                        gapcolor=None, gap_linestyle="--",
                        contour_kwargs=None, contourf_kwargs=None, alpha_xx=0.5):
    contour_kwargs = _init_kwargs_dict(contour_kwargs)
    contourf_kwargs = _init_kwargs_dict(contourf_kwargs)
    contourf_kwargs.setdefault("antialiased", False)

    if levels is None:
        levels = get_levels(np.arange(1, 3))

    V = _find_hdi_contours(pdf, levels[::-1])

    if shade_background:
        base_color = ax.get_facecolor()
        base_cmap = LinearSegmentedColormap.from_list(
            "base_cmap", [base_color, base_color], N=2
        )
        ax.contourf(
            X, Y, pdf, [V.min(), pdf.max()],
            cmap=base_cmap,
            antialiased=False,
        )

    ax.contour(
        X, Y, pdf, V[:],
        colors=[color],
        **contour_kwargs,
    )
    if gapcolor is not None:
        kw = contour_kwargs | dict(linestyles=[gap_linestyle])
        ax.contour(
            X, Y, pdf, V[:],
            colors=[gapcolor],
            **kw,
        )

    if fill_contours:
        rgba_color = colorConverter.to_rgba(color)
        contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
        for i, l in enumerate(levels):
            contour_cmap[i][-1] *= (i + 1 + alpha_xx) / (len(levels) + alpha_xx)

        ax.contourf(
            X, Y, pdf, np.concatenate([V, [pdf.max()]]),
            colors=contour_cmap,
            **contourf_kwargs,
        )


def plot_kde(ax, x, y, color="k", gridsize=(256, 256), **kwargs):
    density, xmin, xmax, ymin, ymax = _fast_kde_2d(x, y, gridsize=gridsize)

    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, gridsize[0]),
        np.linspace(ymin, ymax, gridsize[1]),
        indexing="ij",
    )
    return plot_density_levels(
        ax, X, Y, density, color=color, **kwargs
    )
