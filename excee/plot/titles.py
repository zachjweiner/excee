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
from excee.util import label_from_attrs
from excee.stats import compute_ci
from excee.density import detect_boundaries

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def decide_ci_kind(data, default):
    lower_bounded, upper_bounded = detect_boundaries(data)
    return (
        "upper_limit" if lower_bounded and not upper_bounded
        else "lower_limit" if upper_bounded and not lower_bounded
        else default
    )


def parse_ci_input(data, ci_kind, default_ci_kind, ci_prob):
    ci_kind = (
        decide_ci_kind(np.ravel(data), default_ci_kind)
        if ci_kind == "auto"
        else ci_kind
    )
    if ci_prob is None:
        ci_prob = 0.9544997361036416 if "limit" in ci_kind else 0.6826894921370859

    return ci_kind, ci_prob


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


def format_limit(value, side="upper", label=None, err_prec=3, rescale_thresh=2,
                 style=None):
    op = "<" if side == "upper" else ">"
    label = label or ""
    exp = _exponent(value)
    if abs(exp) > rescale_thresh:
        mantissa = value / 10.**exp
        val_str = rf"{mantissa:.{err_prec}f} \times 10^{{{exp}}}"
    else:
        val_str = f"{value:#.{err_prec}g}"
    return f"{label or ''} ${op} {val_str}$"


def make_ci_str(ary, *, input_kind="sample", use_kde=True,
                ci_kind="eti", default_ci_kind="hdi",
                ci_prob=None, label=None, include_long_names=True, **kwargs):
    if input_kind == "sample":
        ci_kind, ci_prob = parse_ci_input(ary, ci_kind, default_ci_kind, ci_prob)
    elif ci_kind == "auto":
        raise NotImplementedError("auto ci_kind with densities")

    format_kwargs = {
        key: kwargs.pop(key)
        for key in ("err_prec", "rescale_thresh", "style")
        if key in kwargs
    }
    if include_long_names:
        format_kwargs["label"] = label or label_from_attrs(ary)

    ci = compute_ci(
        ary, input_kind, ci_kind=ci_kind, ci_prob=ci_prob, use_kde=use_kde, **kwargs
    )

    if ci_kind in ("eti", "hdi"):
        return format_measurement(ci, **format_kwargs)
    elif "limit" in ci_kind:
        return format_limit(ci, side=ci_kind.replace("_limit", ""), **format_kwargs)
    else:
        raise NotImplementedError(f"{ci_kind=}")


def add_stacked_title(ax, titles, *, colors=None, title_loc="center",
                      title_stack_pad_frac=0.2, **kwargs):
    kwargs.setdefault("fontsize", plt.rcParams["axes.titlesize"])
    if colors is None:
        colors = ["k" for _ in titles]

    xycoords = None
    for title, color in zip(titles[::-1], colors[::-1]):
        if not title:
            continue

        kwargs["color"] = color
        if xycoords is None:
            xycoords = ax.set_title(title, loc=title_loc, **kwargs)
        else:
            xycoords = ax.annotate(
                title, (0, 1 + title_stack_pad_frac), xycoords=xycoords,
                va="bottom", ha="left", **kwargs,
            )
