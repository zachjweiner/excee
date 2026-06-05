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
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from excee.util import label_from_attrs, _init_kwargs_dict

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

std_quantiles = (0.15865525, 0.5, 0.84134475)


def quantiles_from_log_pdf(log_pdf, x, quantiles):
    pdf = np.exp(log_pdf - log_pdf.max())
    slc = np.where(pdf > 1e-20)  # FIXME: smarter way?
    pdf = pdf[slc]
    x = x[slc]
    pdf /= simpson(pdf, x=x)
    cdf = CubicSpline(x, pdf).antiderivative()

    return np.array([cdf.solve(q, extrapolate=False).squeeze() for q in quantiles])


def _exponent(x):
    return np.floor(np.log10(np.abs(x))).astype(int)


def format_measurement(quantiles, err_prec=2, rescale_thresh=2,
                       label=None, style="paren"):
    q_lo, q_mid, q_hi = quantiles
    q_m, q_p = q_mid - q_lo, q_hi - q_mid

    _exps = _exponent([q_m, q_p])
    if (
        max(*(_exps + 1), 0) <= min(err_prec, rescale_thresh)
        and max(*(-_exps - 1), 0) <= rescale_thresh
    ):
        rescale_exp = 0
    else:
        rescale_exp = max(*_exps, _exponent(q_mid))

    m_digits, p_digits = np.maximum(0, err_prec - (_exps - rescale_exp) - 1)
    mid_digits = max(m_digits, p_digits)

    m_str = f"{{:.{m_digits}f}}".format(q_m / 10.**rescale_exp)
    p_str = f"{{:.{p_digits}f}}".format(q_p / 10.**rescale_exp)
    mid_str = f"{{:.{mid_digits}f}}".format(q_mid / 10.**rescale_exp)
    meas = fr"{mid_str}_{{-{m_str}}}^{{+{p_str}}}"

    lhs = f"{label} = " if label else ""
    if rescale_exp != 0:
        if style == "paren":
            meas = lhs + fr"$\left( {meas} \right) \times 10^{{{rescale_exp}}}$"
        elif style == "multiply" and label is not None:
            meas = fr"$10^{{{-rescale_exp}}}$\,{{{label}}} = ${meas}$"
        else:
            raise ValueError()
    else:
        meas = (f"{label} = " if label else "") + f"${meas}$"

    return meas


def measurement_from_sample(sample, quantiles=std_quantiles, weights=None,
                            **kwargs):
    qs = np.quantile(sample, quantiles, weights=weights, method="inverted_cdf")
    return format_measurement(qs, **kwargs)


def measurement_from_log_pdf(log_pdf, x, quantiles=std_quantiles, **kwargs):
    qs = quantiles_from_log_pdf(log_pdf, x, quantiles)
    return format_measurement(qs, **kwargs)


def add_stacked_title(ax, arys, title_quantiles=std_quantiles,
                      *, kind="sample", weights=None, colors=None, label=None,
                      title_loc="center", title_kwargs=None,
                      title_stack_pad_frac=0.2, include_long_names=True):
    if weights is not None:
        raise NotImplementedError("weights")

    title_kwargs = _init_kwargs_dict(title_kwargs)
    title_kwargs.setdefault("fontsize", plt.rcParams["axes.titlesize"])
    meas_kwargs = {
        "err_prec": title_kwargs.pop("err_prec", 2),
        "rescale_thresh": title_kwargs.pop("rescale_thresh", 2),
        "style": title_kwargs.pop("style", "paren"),
    }
    if colors is None:
        colors = ["k"]*len(arys)

    xycoords = None
    for x, color in zip(arys[::-1], colors[::-1]):
        if x is None:
            continue
        label = label or label_from_attrs(x) if include_long_names else None
        if kind == "sample":
            _x = np.asarray(x).ravel()
            title = measurement_from_sample(
                _x, title_quantiles, weights=weights, label=label,
                **meas_kwargs,
            )
        elif kind == "pdf":
            coord, pdf = x.coords[x.dims[0]], x
            title = measurement_from_log_pdf(
                np.log(pdf), coord, title_quantiles, label=label,
                **meas_kwargs,
            )
        else:
            raise RuntimeError(f"{kind=}")

        title_kwargs["color"] = color
        if xycoords is None:
            xycoords = ax.set_title(title, loc=title_loc, **title_kwargs)
        else:
            xycoords = ax.annotate(
                title, (0, 1 + title_stack_pad_frac), xycoords=xycoords,
                va="bottom", ha="left", **title_kwargs,
            )
