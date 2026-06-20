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


from typing import Literal
from collections.abc import Mapping
from numpy.typing import ArrayLike
import xarray as xr
from matplotlib.typing import (
    ColorType as _ColorType,
    LineStyleType as _LineStyleType
)

type DataSpec = xr.Dataset | xr.DataTree | Mapping[str, ArrayLike]
"""
Acceptable container formats for samples.
"""

type BroadcastableToVars[T] = T | dict[str, T]
"""
Argument that broadcasts across variables.
Either specified as a single value of type ``T`` for all variables or as a
:class:`collections.abc.Mapping` of variable names to individual values
(with default values applied to missing keys).
"""

type BroadcastableToDatasets[T] = T | list[T]
"""
Argument that broadcasts across datasets.
Either specified as a single value of type ``T`` for all datasets or as a
:class:`list` of values that apply to each passed dataset.
"""

type BroadcastableToVarsAndDatasets[T] = BroadcastableToDatasets[
    BroadcastableToVars[T]
]
"""
Argument that broadcasts across both variables and datasets, composing
:type:`BroadcastableToDatasets` with :type:`BroadcastableToVars`.
"""

type BoundsTuple = tuple[float | None, float | None]
"""
Lower and upper boundaries of (explicit) prior support.
Values of ``None`` indicate no explicit boundary is to be accounted for.
"""

type LimitsSpecifiers = tuple[float, float] | Literal["auto"] | None
"""
Specification for axis limits.

Can be a :class:`tuple` of :class:`float`\\ s for explicit limits,
``"auto"`` for inclusive limits determined automatically from the data,
or ``None`` to leave limits unmodified.
"""

type AxesScale = Literal["linear", "log"]
"""
Scaling applied to plot axes.
"""

type CIKind = Literal["eti", "hdi", "upper_limit", "lower_limit", "auto"]
"""
Types of credible intervals available for plotting.

Includes  equal-tailed interval (``"eti"``), highest density interval (``"hdi"``),
one-sided limits (``"upper_limit"`` and ``"lower_limit"``),
and ``"auto"`` for automatic detection.
"""

type ColorType = _ColorType
"""
Alias for :py:type:`matplotlib.typing.ColorType`.
"""

type LineStyleType = _LineStyleType
"""
Alias for :py:type:`matplotlib.typing.LineStyleType`.
"""
